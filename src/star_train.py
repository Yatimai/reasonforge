#!/usr/bin/env python3
"""
STaR-SQL Pipeline
=================

Implementation of STaR-SQL with CV prompt (schema + cell values, no CoT).

Architecture:
1. Generate: Question + Schema → SQL (k candidates)
2. Evaluate: Execute SQL, check result match
3. Collect correct examples → SFT data
4. Injection: For failed questions (0/k correct), inject gold SQL as training
5. Train SFT on correct + injected examples (from BASE model each iteration)
6. Iterate

Key features:
- CV prompt: schema with sample cell values (no Chain-of-Thought)
- k-candidates sampling for diverse generation
- Injection: Learn from failed questions using gold SQL
- Train from base model each iteration (prevents overfitting)
- Fixed sample: same questions each iteration (STaR-SQL paper)

Stack: Ministral-8B + LLaMA-Factory + LoRA + vLLM
"""

import json
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sqlite3
import re
import shutil
import gc
import os
import time
import sys

from loguru import logger
from tqdm import tqdm
import torch


# =============================================================================
# GPU MEMORY MANAGEMENT
# =============================================================================

def clear_gpu_memory():
    """
    Aggressively clear GPU memory.

    Called between training and inference to ensure vLLM can load the model.
    """
    logger.info("Clearing GPU memory...")

    # Clear Python garbage collector
    gc.collect()

    # Clear PyTorch CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Log memory status
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            free_in_cache = reserved - allocated  # Available in PyTorch cache
            truly_free = total - reserved  # Not yet reserved by PyTorch
            logger.info(f"   GPU {i}: {allocated:.1f}GB used, {free_in_cache:.1f}GB cached, {truly_free:.1f}GB free / {total:.1f}GB total")

    gc.collect()
    logger.info("   GPU memory cleared")


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_PATH = Path("configs/config.yaml")

DATA_DIR = Path("data")
SPIDER_DIR = DATA_DIR / "spider"
LLAMAFACTORY_DATA_DIR = DATA_DIR / "llamafactory"
MODELS_DIR = Path("models")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation."""
    question: str
    schema: str
    db_id: str
    reasoning: str
    predicted_sql: str
    gold_sql: str
    is_correct: bool
    execution_result: Optional[str] = None
    error_message: Optional[str] = None
    difficulty_weight: int = 1  # For difficulty resampling: higher = harder question


# =============================================================================
# IMPORTS FROM CENTRALIZED MODULES
# =============================================================================

# Import from centralized modules to ensure consistency
from prompts import (
    SYSTEM_PROMPT, format_prompt,
    get_stop_tokens, get_max_tokens, get_temperature
)
from sql_utils import (
    execute_sql,
    parse_response,
    normalize_sql_result,
    check_result_match,
    load_spider_data as _load_spider_data_base,
)


def evaluate_single_item(args) -> dict:
    """
    Evaluate a single item (for parallel execution).

    This function is designed to be called by ProcessPoolExecutor.
    Returns a dict with all evaluation results for this item.
    """
    item, response_candidates, schemas, db_base_path, k_candidates = args

    try:
        gold_sql = item['query']
        db_path = db_base_path / item['db_id'] / f"{item['db_id']}.sqlite"

        # Execute gold SQL once
        gold_success, gold_results, _ = execute_sql(gold_sql, db_path)

        # Handle k=1 case (backward compatibility: response is string, not list)
        if k_candidates == 1:
            candidates = [response_candidates] if isinstance(response_candidates, str) else response_candidates
        else:
            candidates = response_candidates

        # Safety check for None/empty candidates
        if not candidates:
            candidates = [""]

        # Evaluate ALL candidates
        n_correct_candidates = 0
        candidate_results = []

        for candidate in candidates:
            if candidate is None:
                candidate = ""
            reasoning, predicted_sql = parse_response(candidate)

            # Execute predicted SQL
            pred_success, pred_results, pred_error = execute_sql(predicted_sql, db_path)

            # Check correctness
            is_correct = False
            if pred_success and gold_success:
                is_correct = check_result_match(pred_results, gold_results)

            if is_correct:
                n_correct_candidates += 1

            candidate_results.append({
                'reasoning': reasoning,
                'predicted_sql': predicted_sql,
                'is_correct': is_correct,
                'exec_result': str(pred_results) if pred_success else None,
                'error': pred_error,
            })

        # Difficulty weight: more failures = harder question = higher weight
        # Fix: 0 correct = hardest = max weight (k_candidates), not 1
        difficulty_weight = max(1, k_candidates - n_correct_candidates) if n_correct_candidates > 0 else k_candidates

        return {
            'success': True,
            'item': item,
            'gold_sql': gold_sql,
            'candidate_results': candidate_results,
            'difficulty_weight': difficulty_weight,
        }
    except Exception as e:
        return {
            'success': False,
            'item': item,
            'error': str(e),
        }


# normalize_sql_result and check_result_match are imported from sql_utils


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spider_data() -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
    """
    Load Spider train and dev data.

    Returns:
        Tuple of (train_data, dev_data, schemas)
    """
    train_data, dev_data, schemas = _load_spider_data_base(SPIDER_DIR)
    logger.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev examples")
    return train_data, dev_data, schemas


# =============================================================================
# GENERATION (via LLaMA-Factory CLI)
# =============================================================================

def generate_with_vllm_subprocess(
    prompts: List[str],
    model_path: str = "mistralai/Ministral-8B-Instruct-2410",
    temperature: float = None,  # Uses get_temperature("inference") if None
    lora_path: str = None,
    n_candidates: int = 1,
) -> List[List[str]]:
    """
    Generate responses using vLLM in a SUBPROCESS.

    This ensures GPU memory is fully released when the subprocess exits,
    which is necessary because vLLM doesn't properly release CUDA memory
    even with del + empty_cache().

    Args:
        prompts: List of prompts to generate responses for
        model_path: Base model path (HuggingFace ID or local path)
        temperature: Sampling temperature
        lora_path: Optional path to LoRA adapter to apply
        n_candidates: Number of candidates to generate per prompt (k for multi-candidates)

    Returns:
        List of lists: For each prompt, a list of n_candidates responses
    """
    import tempfile

    # Save prompts to temp file
    prompts_file = Path(tempfile.mktemp(suffix=".json"))
    outputs_file = Path(tempfile.mktemp(suffix=".json"))

    with open(prompts_file, 'w') as f:
        json.dump(prompts, f)

    # Build LoRA config for script
    lora_config = ""
    lora_request_code = "None"
    if lora_path:
        lora_config = f'''
from vllm.lora.request import LoRARequest
lora_request = LoRARequest("sql_adapter", 1, "{lora_path}")
'''
        lora_request_code = "lora_request"

    # Get centralized config
    stop_tokens = get_stop_tokens()
    max_tokens = get_max_tokens()
    if temperature is None:
        temperature = get_temperature("inference")

    # Create subprocess script
    # Note: SYSTEM_PROMPT is injected from the centralized prompts.py module
    script = f'''
import json
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
{lora_config}

# Centralized prompt (injected from prompts.py to ensure consistency)
SYSTEM_PROMPT = "{SYSTEM_PROMPT}"

# Load prompts
with open("{prompts_file}") as f:
    prompts = json.load(f)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("{model_path}", trust_remote_code=True)
llm = LLM(
    model="{model_path}",
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    enable_lora={"True" if lora_path else "False"},
    max_lora_rank=16,
    max_num_seqs=16,
)

sampling_params = SamplingParams(
    temperature={temperature},
    max_tokens={max_tokens},
    n={n_candidates},
    stop={stop_tokens},
)

# Format prompts with chat template
formatted_prompts = []
for prompt in prompts:
    messages = [
        {{"role": "system", "content": SYSTEM_PROMPT}},
        {{"role": "user", "content": prompt}}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    formatted_prompts.append(formatted)

# Generate
print(f"Generating {{len(formatted_prompts)}} responses with n={n_candidates} candidates each...")
outputs = llm.generate(formatted_prompts, sampling_params, lora_request={lora_request_code})
# Return list of lists: for each prompt, all n candidates
responses = [[output.text for output in out.outputs] for out in outputs]

# Save outputs
with open("{outputs_file}", 'w') as f:
    json.dump(responses, f)

print("Done!")
'''

    # Run in subprocess (no timeout - let it run to completion)
    logger.info(f"Starting vLLM subprocess for {len(prompts)} prompts...")
    logger.info(f"   Using Python: {sys.executable}")
    result = subprocess.run(
        [sys.executable, "-c", script],
        stdout=None,  # Inherit parent's stdout (visible in terminal)
        stderr=None,  # Inherit parent's stderr (errors visible)
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"vLLM subprocess failed with return code {result.returncode}")
        # Return empty responses matching expected format
        if n_candidates == 1:
            return ["" for _ in prompts]
        return [[""] * n_candidates for _ in prompts]

    logger.info("vLLM generation completed")

    # Load outputs
    with open(outputs_file) as f:
        responses = json.load(f)

    # Cleanup temp files
    prompts_file.unlink()
    outputs_file.unlink()

    logger.info("vLLM subprocess completed")

    # Force GPU memory release after subprocess exit
    # vLLM doesn't always release CUDA memory immediately
    logger.info("   Waiting for GPU memory release...")
    time.sleep(5)  # Wait for CUDA driver to fully release

    # Multiple cleanup attempts
    for attempt in range(3):
        clear_gpu_memory()
        time.sleep(2)

        # Check if GPU is actually free
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            if allocated < 1.0:  # Less than 1GB
                logger.info(f"   GPU memory verified released ({allocated:.2f}GB)")
                break
            else:
                logger.warning(f"   GPU still has {allocated:.2f}GB allocated, attempt {attempt+1}/3")
    else:
        logger.warning("   GPU memory may not be fully released")

    # For backward compatibility: if n_candidates=1, flatten to list of strings
    if n_candidates == 1:
        return [r[0] if r else "" for r in responses]
    return responses


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_batch(
    data: List[Dict],
    schemas: Dict[str, str],
    model_path: str,
    sample_size: int = 500,
    base_model: str = "mistralai/Ministral-8B-Instruct-2410",
    k_candidates: int = 1,
    iteration: int = 1,
) -> List[GenerationResult]:
    """
    Evaluate model on a batch of examples with k-candidates sampling.

    With k > 1 (multi-candidates), generates k SQL queries per question
    and keeps the FIRST correct one. This dramatically improves accuracy
    while using authentic model-generated SQL (CV prompt: no reasoning).

    Args:
        data: List of examples to evaluate
        schemas: Dict mapping db_id to schema string
        model_path: Either base model ID or path to LoRA adapter
        sample_size: Number of examples to sample
        base_model: Base model for LoRA inference
        k_candidates: Number of candidates to generate per question (default 1)

    Returns list of GenerationResult with SQL and correctness.
    """
    import random

    # Sample data
    if len(data) > sample_size:
        data = random.sample(data, sample_size)

    # Prepare prompts
    prompts = []
    for item in data:
        schema = schemas.get(item['db_id'], "")
        prompt = format_prompt(item['question'], schema)
        prompts.append(prompt)

    # Determine if we're using a LoRA adapter or base model
    lora_path = None
    actual_model = model_path

    # Check if model_path is a LoRA adapter (local directory with adapter files)
    # Only check local paths (starting with ./ or /) to avoid confusing HF model IDs
    is_local_path = model_path.startswith('./') or model_path.startswith('/')
    if is_local_path and Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
        logger.info(f"   Using LoRA adapter: {model_path}")
        lora_path = model_path
        actual_model = base_model
    else:
        logger.info(f"   Using base model: {model_path}")

    # Temperature for diverse candidate generation (STaR-SQL recommendation)
    temperature = get_temperature("training")  # 0.7 for diversity
    logger.info(f"   Temperature: {temperature}")

    # Generate (with k candidates if k > 1)
    logger.info(f"Generating {len(prompts)} prompts × {k_candidates} candidates (temp={temperature})...")
    responses = generate_with_vllm_subprocess(
        prompts, actual_model, temperature=temperature,
        lora_path=lora_path, n_candidates=k_candidates
    )

    # Evaluate each - PARALLEL execution for speed
    results = []
    db_base_path = SPIDER_DIR / "database"

    # Prepare args for parallel execution
    eval_args = [
        (item, response_candidates, schemas, db_base_path, k_candidates)
        for item, response_candidates in zip(data, responses)
    ]

    # Use ProcessPoolExecutor for parallel SQL evaluation (8 workers)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_workers = min(8, len(eval_args))  # Max 8 workers
    logger.info(f"   Evaluating {len(eval_args)} items with {n_workers} parallel workers...")

    eval_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(evaluate_single_item, arg): i for i, arg in enumerate(eval_args)}

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), desc="Evaluating", total=len(futures), dynamic_ncols=True, mininterval=0.5):
            eval_results.append((futures[future], future.result()))

    # Sort by original order and process results
    eval_results.sort(key=lambda x: x[0])

    for _, eval_result in eval_results:
        item = eval_result['item']

        if eval_result['success']:
            gold_sql = eval_result['gold_sql']
            candidate_results = eval_result['candidate_results']
            difficulty_weight = eval_result['difficulty_weight']

            # Add ALL candidates as separate results
            for cand in candidate_results:
                results.append(GenerationResult(
                    question=item['question'],
                    schema=schemas.get(item['db_id'], ""),
                    db_id=item['db_id'],
                    reasoning=cand['reasoning'],
                    predicted_sql=cand['predicted_sql'],
                    gold_sql=gold_sql,
                    is_correct=cand['is_correct'],
                    execution_result=cand['exec_result'],
                    error_message=cand['error'],
                    difficulty_weight=difficulty_weight,
                ))
        else:
            # Error case - categorize for easier debugging
            error = eval_result['error']
            error_preview = error[:100] if error else "Unknown"
            if 'timeout' in error.lower() or 'interrupt' in error.lower():
                logger.warning(f"⏱️ TIMEOUT: {error_preview}")
            elif 'syntax' in error.lower() or 'near' in error.lower():
                logger.warning(f"SYNTAX: {error_preview}")
            elif 'no such' in error.lower():
                logger.warning(f"SCHEMA: {error_preview}")
            else:
                logger.warning(f"⚠️ ERROR: {error_preview}")
            results.append(GenerationResult(
                question=item.get('question', ''),
                schema=schemas.get(item.get('db_id', ''), ""),
                db_id=item.get('db_id', ''),
                reasoning="",
                predicted_sql="",
                gold_sql=item.get('query', ''),
                is_correct=False,
                execution_result=None,
                error_message=eval_result['error'],
                difficulty_weight=1,
            ))

    # Log k-candidates stats (per QUESTION, not per candidate)
    if k_candidates > 1:
        # Group by question to count questions with at least 1 correct
        from collections import defaultdict
        questions_correct = defaultdict(bool)
        for r in results:
            if r.is_correct:
                questions_correct[r.question] = True

        n_questions = len(results) // k_candidates
        n_questions_correct = sum(1 for q, correct in questions_correct.items() if correct)
        total_correct = sum(1 for r in results if r.is_correct)

        q_pct = (n_questions_correct / n_questions * 100) if n_questions > 0 else 0
        c_pct = (total_correct / len(results) * 100) if results else 0
        logger.info(f"   k={k_candidates}: {n_questions_correct}/{n_questions} questions with ≥1 correct ({q_pct:.1f}%)")
        logger.info(f"   Total correct candidates: {total_correct}/{len(results)} ({c_pct:.1f}%)")

    return results


def evaluate_dev_self_consistency(
    dev_data: List[Dict],
    schemas: Dict[str, str],
    model_path: str,
    base_model: str,
    k: int = 16,
) -> float:
    """
    Self-consistency evaluation on dev set (k=16, majority vote on results).

    Args:
        dev_data: Dev dataset
        schemas: Database schemas
        model_path: Current model (merged or base)
        base_model: Base model for reference
        k: Number of candidates (default 16)

    Returns:
        Dev accuracy (0.0 to 1.0)
    """
    from collections import Counter, defaultdict

    logger.info(f"\nEvaluating on dev set with self-consistency (k={k})...")

    # Generate k candidates per dev example
    results = evaluate_batch(
        data=dev_data,
        schemas=schemas,
        model_path=model_path,
        sample_size=len(dev_data),
        base_model=base_model,
        k_candidates=k,
        iteration=0,
    )

    # Group results by question
    question_candidates = defaultdict(list)
    question_info = {}

    for r in results:
        question_candidates[r.question].append(r)
        question_info[r.question] = {'db_id': r.db_id, 'gold_sql': r.gold_sql}

    correct = 0
    total = len(question_candidates)
    db_base_path = SPIDER_DIR / "database"

    for question, candidates in question_candidates.items():
        info = question_info[question]
        db_path = db_base_path / info['db_id'] / f"{info['db_id']}.sqlite"

        # Execute gold SQL
        gold_success, gold_results, _ = execute_sql(info['gold_sql'], db_path)
        if not gold_success or gold_results is None:
            continue

        # Normalize gold result (convert to strings to handle mixed types)
        gold_normalized = tuple(sorted([tuple(str(v) for v in row) for row in gold_results]))

        # Count execution results (normalized)
        result_counts = Counter()

        for r in candidates:
            if r.predicted_sql:
                pred_success, pred_results, _ = execute_sql(r.predicted_sql, db_path)
                if pred_success and pred_results is not None:
                    # Normalize result for counting (convert to strings to handle mixed types)
                    result_key = tuple(sorted([tuple(str(v) for v in row) for row in pred_results]))
                    result_counts[result_key] += 1

        if not result_counts:
            continue

        # Majority vote: most common result
        most_common_result = result_counts.most_common(1)[0][0]

        # Check if majority result matches gold
        if most_common_result == gold_normalized:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"   Dev accuracy (SC k={k}): {accuracy:.1%} ({correct}/{total})")

    return accuracy


# =============================================================================
# INJECTION (gold SQL for failed questions)
# =============================================================================

@dataclass
class FailedQuestion:
    """A question where all k candidates failed."""
    question: str
    schema: str
    db_id: str
    gold_sql: str
    n_failures: int  # L = number of copies based on difficulty


def inject_gold_sql(
    failed_questions: List[FailedQuestion],
    model_path: str = None,  # Unused with CV prompt
    base_model: str = None,  # Unused with CV prompt
) -> List[Dict]:
    """
    Inject gold SQL for failed questions (0/k correct).

    CV prompt simplification: No vLLM call needed.
    We just create (question, gold_sql) pairs directly.
    This teaches the model the correct answer for hard questions.

    Args:
        failed_questions: Questions where all k candidates were incorrect
        model_path: Unused (kept for API compatibility)
        base_model: Unused (kept for API compatibility)

    Returns:
        List of SFT examples (instruction, input, output)
    """
    if not failed_questions:
        return []

    logger.info(f"\nInjecting gold SQL for {len(failed_questions)} failed questions...")

    # CV prompt: just create (question → gold_sql) pairs directly
    # No vLLM generation needed - we already have the correct answer
    sft_data = []
    for fq in failed_questions:
        # The instruction is the prompt (question + schema)
        instruction = format_prompt(fq.question, fq.schema)

        # CV prompt: output is just the gold SQL
        output = fq.gold_sql

        example = {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

        # Add L copies based on difficulty (n_failures)
        for _ in range(fq.n_failures):
            sft_data.append(example.copy())

    logger.info(f"   Injected {len(sft_data)} examples (from {len(failed_questions)} failed questions)")
    return sft_data


# =============================================================================
# DATA PREPARATION FOR LLAMAFACTORY
# =============================================================================

def prepare_sft_data(results: List[GenerationResult], difficulty_resampling: bool = False) -> List[Dict]:
    """
    Prepare SFT data from correct examples with difficulty resampling.

    Format for LLaMA-Factory:
    {"instruction": "...", "input": "", "output": "..."}

    Difficulty resampling: duplicate hard questions based on difficulty_weight.
    - Easy question (4/4 correct): weight=1 -> 1 copy
    - Hard question (1/4 correct): weight=3 -> 3 copies
    """
    sft_data = []
    total_weight = 0
    n_correct = 0

    for r in results:
        if r.is_correct:
            n_correct += 1
            instruction = format_prompt(r.question, r.schema)
            # CV prompt: just the SQL, no reasoning
            output = r.predicted_sql

            example = {
                "instruction": instruction,
                "input": "",
                "output": output,
            }

            # Difficulty resampling: add multiple copies for hard questions
            weight = r.difficulty_weight if difficulty_resampling else 1
            for _ in range(weight):
                sft_data.append(example.copy())
            total_weight += weight

    if difficulty_resampling and total_weight > n_correct:
        logger.info(f"Prepared {total_weight} SFT examples ({n_correct} unique, resampled by difficulty)")
    else:
        logger.info(f"Prepared {len(sft_data)} SFT examples from {len(results)} total")
    return sft_data


def save_llamafactory_dataset(data: List[Dict], name: str):
    """Save dataset in LLaMA-Factory format with proper dataset_info.json."""
    LLAMAFACTORY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save data
    data_path = LLAMAFACTORY_DATA_DIR / f"{name}.json"
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Update dataset_info.json with proper column mapping
    info_path = LLAMAFACTORY_DATA_DIR / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {}

    # SFT format (alpaca style)
    info[name] = {
        "file_name": f"{name}.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }

    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved {len(data)} examples to {data_path}")


# =============================================================================
# TRAINING (via LLaMA-Factory CLI)
# =============================================================================

def train_sft(base_model: str = "mistralai/Ministral-8B-Instruct-2410"):
    """
    Run SFT training via LLaMA-Factory CLI in a subprocess.

    The subprocess ensures GPU memory is fully released after training,
    which is critical for vLLM inference to load the model.
    """
    logger.info(f"Starting SFT training in subprocess with model: {base_model}")

    # Clear any existing GPU memory before training
    clear_gpu_memory()

    # Generate dynamic LLaMA-Factory config with correct model
    # Determine template based on model name
    if "qwen" in base_model.lower():
        template = "qwen"
    elif "llama" in base_model.lower():
        template = "llama3"
    elif "mistral" in base_model.lower():
        template = "mistral"
    else:
        template = "default"

    dynamic_config = {
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.15,
        "lora_target": "all",
        "use_dora": False,
        "dataset": "spider_correct",
        "dataset_dir": "./data/llamafactory",
        "template": template,
        "cutoff_len": 2048,
        "max_samples": 10000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": "./models/sft_lora",
        "logging_steps": 10,
        "save_steps": 500,
        "plot_loss": True,
        "overwrite_output_dir": True,
        "save_only_model": False,
        "report_to": "wandb",
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 2,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "ddp_timeout": 180000000,
        "flash_attn": "auto",
        "gradient_checkpointing": True,
    }

    # Write dynamic config
    dynamic_config_path = Path("configs/llamafactory/sft_dynamic.yaml")
    dynamic_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dynamic_config_path, 'w') as f:
        yaml.dump(dynamic_config, f)

    logger.info(f"   Using template: {template}")

    # Run training in subprocess - this ensures full memory release on exit
    cmd = f"llamafactory-cli train {dynamic_config_path}"
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        logger.error(f"Training failed with return code {result.returncode}")
        raise RuntimeError(f"LLaMA-Factory training failed with return code {result.returncode}")

    logger.info("SFT training completed")

    # Clear GPU memory after training (subprocess should have released it, but be safe)
    clear_gpu_memory()


def merge_model(adapter_path: str, output_dir: str, base_model: str = "mistralai/Ministral-8B-Instruct-2410"):
    """Merge LoRA weights into base model using LLaMA-Factory."""
    logger.info(f"Merging adapter from {adapter_path} to {output_dir}")

    # Determine template based on model name
    if "qwen" in base_model.lower():
        template = "qwen"
    elif "llama" in base_model.lower():
        template = "llama3"
    elif "mistral" in base_model.lower():
        template = "mistral"
    else:
        template = "default"

    # Create a temporary export config
    export_config = {
        "model_name_or_path": base_model,
        "adapter_name_or_path": adapter_path,
        "template": template,
        "finetuning_type": "lora",
        "export_dir": output_dir,
        "export_size": 2,
        "export_legacy_format": False,
    }

    config_path = Path("configs/llamafactory/export_temp.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(export_config, f)

    cmd = f"llamafactory-cli export {config_path}"
    subprocess.run(cmd, shell=True, check=True)

    # Cleanup temp config
    config_path.unlink()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class STaRPipeline:
    """
    STaR-SQL Pipeline with CV prompt (no CoT).

    Iteration:
    1. Evaluate current model → collect correct + errors (k candidates)
    2. SFT on correct examples (CV prompt: just SQL)
    3. Repeat
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.train_data, self.dev_data, self.schemas = load_spider_data()

        # Check for saved state (auto-resume)
        self.state_file = MODELS_DIR / "pipeline_state.json"
        saved_state = self._load_state()

        if saved_state:
            self.current_model = saved_state['current_model']
            self.iteration = saved_state['iteration']
            self.history = saved_state['history']
            logger.info(f"RESUMING from iteration {self.iteration + 1}")
            logger.info(f"   Model: {self.current_model}")
        else:
            self.current_model = self.config['model']['name']
            self.iteration = 0
            self.history = []

        # SFT data (correct examples only, fresh each iteration per STaR-SQL paper)
        self.all_sft_data = []

        # Fixed sample for all iterations (STaR-SQL paper style)
        import random
        sample_size = self.config['self_improvement']['eval_sample_size']
        if len(self.train_data) > sample_size:
            self.fixed_sample = random.sample(self.train_data, sample_size)
            logger.info(f"Fixed sample of {sample_size} questions for all iterations (STaR-SQL style)")
        else:
            self.fixed_sample = list(self.train_data)
            logger.info(f"Using all {len(self.fixed_sample)} training questions")

        # Track best model
        self.best_accuracy = 0.0
        self.best_checkpoint_path = None

        # STaR config: train_from_base (paper recommendation)
        self.train_from_base = self.config.get('star', {}).get('train_from_base', True)
        self.base_model = self.config['model']['name']  # Always keep reference to base

        # Difficulty resampling: duplicate hard questions in training data
        self.difficulty_resampling = self.config.get('star', {}).get('difficulty_resampling', False)
        if self.difficulty_resampling:
            logger.info("Difficulty resampling ENABLED - hard questions will be oversampled")

    def _load_state(self) -> Optional[Dict]:
        """Load saved pipeline state for auto-resume."""
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return None

    def _save_state(self):
        """Save pipeline state after each iteration."""
        import json
        state = {
            'iteration': self.iteration,
            'current_model': self.current_model,
            'history': self.history,
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"State saved: iteration {self.iteration}, model {self.current_model}")

    def run_iteration(self) -> Dict:
        """Run one iteration of STaR-SQL (SFT + Injection)."""
        self.iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {self.iteration}")
        logger.info(f"{'='*60}")

        # Step 1: Evaluate on UNSEEN data (variety)
        logger.info("\nStep 1: Evaluating current model on unseen data...")

        # Clear GPU memory before inference
        clear_gpu_memory()

        sample_size = self.config['self_improvement']['eval_sample_size']

        # STaR-SQL paper: use same questions each iteration
        # The model generates better SQL as it improves across iterations
        self.remaining_data = list(self.fixed_sample)
        logger.info(f"   Using fixed {len(self.fixed_sample)} questions (STaR-SQL style)")

        # Get k_candidates from config (default 8 as per STaR-SQL paper)
        k_candidates = self.config.get('star', {}).get('k_candidates', 8)
        if k_candidates > 1:
            logger.info(f"   Using k={k_candidates} multi-candidates (first correct wins)")

        results = evaluate_batch(
            data=self.remaining_data,  # Only unseen data
            schemas=self.schemas,
            model_path=self.current_model,
            sample_size=sample_size,
            base_model=self.base_model,
            k_candidates=k_candidates,
            iteration=self.iteration,  # For B-STaR dynamic temperature
        )

        # Note: With fixed sample (STaR-SQL paper), we use the same questions each iteration
        # The model improves its answers on the same questions over time

        # Calculate metrics (per question, not per candidate)
        # With k>1, we have multiple results per question - count question as correct if ANY candidate is correct
        from collections import defaultdict
        questions_correct = defaultdict(bool)
        for r in results:
            if r.is_correct:
                questions_correct[r.question] = True
            elif r.question not in questions_correct:
                questions_correct[r.question] = False

        correct = sum(1 for q, is_correct in questions_correct.items() if is_correct)
        total = len(questions_correct)
        accuracy = correct / total if total > 0 else 0

        logger.info(f"   Result Match: {accuracy:.1%} ({correct}/{total})")

        # Step 2: Prepare training data (STaR-SQL style)
        logger.info("\nStep 2: Preparing training data...")

        # Get correct examples for SFT (with difficulty resampling as per STaR-SQL paper)
        sft_data = prepare_sft_data(results, difficulty_resampling=True)
        n_unique = sum(1 for r in results if r.is_correct)
        logger.info(f"   Correct examples: {n_unique} (training samples: {len(sft_data)})")

        # Step 2b: Injection for failed questions (gold SQL for 0/k correct)
        # Identify questions where ALL k candidates failed
        question_stats = defaultdict(lambda: {"correct": 0, "total": 0, "schema": "", "gold_sql": "", "db_id": ""})
        for r in results:
            question_stats[r.question]["total"] += 1
            if r.is_correct:
                question_stats[r.question]["correct"] += 1
            # Keep track of schema and gold_sql for injection
            question_stats[r.question]["schema"] = r.schema
            question_stats[r.question]["gold_sql"] = r.gold_sql
            question_stats[r.question]["db_id"] = r.db_id

        # Collect failed questions (0 correct out of k)
        failed_questions = []
        for question, stats in question_stats.items():
            if stats["correct"] == 0:
                failed_questions.append(FailedQuestion(
                    question=question,
                    schema=stats["schema"],
                    db_id=stats["db_id"],
                    gold_sql=stats["gold_sql"],
                    n_failures=stats["total"],  # L = k (all failed)
                ))

        n_failed = len(failed_questions)
        logger.info(f"   Failed questions (0/{k_candidates} correct): {n_failed}")

        # Inject gold SQL for failed questions
        injected_data = []
        if n_failed > 0:
            injected_data = inject_gold_sql(
                failed_questions=failed_questions,
                model_path=self.current_model,
                base_model=self.base_model,
            )
            logger.info(f"   Injected examples: {len(injected_data)}")

        # Combine: correct examples + injected examples
        self.all_sft_data = sft_data + injected_data
        logger.info(f"   Total SFT data: {len(self.all_sft_data)} (correct: {len(sft_data)}, injected: {len(injected_data)})")

        # Save dataset
        save_llamafactory_dataset(self.all_sft_data, "spider_correct")

        # Step 3: Train SFT on correct examples
        if len(self.all_sft_data) >= 100:
            logger.info("\nStep 3: SFT Training...")

            # STaR paper: "always return to the original pre-trained model for re-initialization"
            # This prevents catastrophic forgetting and overfitting
            if self.train_from_base:
                logger.info(f"   Training from BASE model: {self.base_model}")
                logger.info(f"   On ACCUMULATED data: {len(self.all_sft_data)} examples")
            else:
                logger.info(f"   ➡️ Continuing from checkpoint (train_from_base=False)")

            train_sft(base_model=self.base_model)

            # Merge LoRA adapter into base model (avoids vLLM LoRA bugs with 12B+ models)
            logger.info("\nMerging LoRA adapter into base model...")
            adapter_path = str(MODELS_DIR / "sft_lora")
            merged_path = str(MODELS_DIR / f"merged_iter_{self.iteration}")
            merge_model(adapter_path, merged_path, base_model=self.base_model)
            self.current_model = merged_path
            logger.info(f"   Merged model saved to: {merged_path}")

        # Step 4: Evaluate on dev set with self-consistency
        logger.info("\nStep 4: Evaluating on dev set...")
        clear_gpu_memory()
        dev_accuracy = evaluate_dev_self_consistency(
            dev_data=self.dev_data,
            schemas=self.schemas,
            model_path=self.current_model,
            base_model=self.base_model,
            k=16,
        )

        # Record history
        result = {
            "iteration": self.iteration,
            "train_accuracy": accuracy,
            "train_correct": correct,
            "train_total": total,
            "dev_accuracy": dev_accuracy,
            "sft_examples": len(sft_data),
            "injected_examples": len(injected_data),
            "total_training_examples": len(self.all_sft_data),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(result)

        # Save history to file
        self._save_history()

        # Track best model (based on DEV accuracy)
        if dev_accuracy > self.best_accuracy:
            self.best_accuracy = dev_accuracy
            self.best_checkpoint_path = self.current_model
            logger.info(f"   🏆 New best model: {dev_accuracy:.1%} on dev")

        # Save state for auto-resume
        self._save_state()

        return result

    def _save_history(self):
        """Save iteration history to file."""
        history_path = MODELS_DIR / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"   History saved to {history_path}")

    def run(self, max_iterations: int = 5):
        """
        Run the full STaR-SQL pipeline for exactly max_iterations.

        STaR-SQL = SFT on correct examples (CV prompt: just SQL, no CoT).
        """
        k_candidates = self.config.get('star', {}).get('k_candidates', 8)
        logger.info("=" * 60)
        logger.info("Starting STaR-SQL Pipeline")
        logger.info(f"Model: {self.config['model']['name']}")
        logger.info(f"Running {max_iterations} iterations")
        logger.info(f"k-candidates: {k_candidates}")
        logger.info("=" * 60)

        previous_accuracy = 0.0

        for i in range(max_iterations):
            result = self.run_iteration()
            current_accuracy = result['train_accuracy']

            # Log improvement
            improvement = current_accuracy - previous_accuracy
            logger.info(f"\nIteration {result['iteration']}: {current_accuracy:.1%} ({improvement:+.1%})")

            previous_accuracy = current_accuracy


        # Final summary
        self._print_summary()

    def _print_summary(self):
        """Print training summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        for h in self.history:
            train_acc = h.get('train_accuracy', h.get('accuracy', 0))
            dev_acc = h.get('dev_accuracy', 0)
            logger.info(f"Iter {h['iteration']}: Train {train_acc:.1%}, Dev {dev_acc:.1%} (SC k=16)")

        if self.history:
            best = max(self.history, key=lambda x: x.get('dev_accuracy', 0))
            logger.info(f"\nBest: Iteration {best['iteration']} with {best.get('dev_accuracy', 0):.1%} on dev")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    pipeline = STaRPipeline()
    pipeline.run(max_iterations=5)


if __name__ == "__main__":
    main()
