#!/usr/bin/env python3
"""
Mini-STaR Retrain on Errors

Iter 1: 100% injection (all errors by definition)
Iter 2+: STaR (model-generated correct) + injection (gold for failures)

Always train LoRA on BASE model (latest merged) to prevent catastrophic forgetting.

Usage:
    python src/star_retrain.py
    python src/star_retrain.py --base_model models/merged_iter_3
    python src/star_retrain.py --errors data/errors/my_errors.json
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SPIDER_DIR = PROJECT_ROOT / "data" / "spider"
DATA_DIR = PROJECT_ROOT / "data" / "llamafactory"
ERRORS_DIR = PROJECT_ROOT / "data" / "errors"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sql_utils import execute_sql, parse_response, check_result_match, load_spider_data
from prompts import format_prompt, SYSTEM_PROMPT


# =============================================================================
# AUTO-DETECTION
# =============================================================================

def find_latest_merged() -> Optional[Path]:
    """Find the latest merged model."""
    # Try merged_iter_* first
    merged = sorted(MODELS_DIR.glob("merged_iter_*"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if merged:
        return merged[0]

    # Try retrain_iter*
    retrain = sorted(MODELS_DIR.glob("retrain_iter*"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if retrain:
        return retrain[0]

    return None


def find_latest_errors() -> Optional[Path]:
    """Find the latest errors file."""
    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    errors = sorted(ERRORS_DIR.glob("errors_*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    return errors[0] if errors else None


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    base_model: Optional[Path] = None      # Auto-detect if None
    errors_file: Optional[Path] = None     # Auto-detect if None
    iterations: int = 3
    k_candidates: int = 8
    temperature: float = 0.7
    lora_rank: int = 16
    lora_alpha: int = 32
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-4


# =============================================================================
# GENERATION (subprocess to avoid GPU memory issues)
# =============================================================================

def generate_with_vllm_subprocess(
    prompts: List[str],
    model_path: str,
    temperature: float = 0.7,
    n_candidates: int = 8,
) -> List[List[str]]:
    """Generate candidates using vLLM in a subprocess."""

    import tempfile
    import os

    # Save prompts to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(prompts, f)
        prompts_file = f.name

    # Output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_file = f.name

    # Script to run in subprocess
    script = f'''
import json
import os
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SYSTEM_PROMPT = """{SYSTEM_PROMPT}"""

with open("{prompts_file}") as f:
    prompts = json.load(f)

llm = LLM(
    model="{model_path}",
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
)

tokenizer = AutoTokenizer.from_pretrained("{model_path}", trust_remote_code=True)

sampling_params = SamplingParams(
    temperature={temperature},
    max_tokens=512,
    n={n_candidates},
    stop=["</s>"],
)

formatted = []
for p in prompts:
    messages = [{{"role": "system", "content": SYSTEM_PROMPT}}, {{"role": "user", "content": p}}]
    formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

outputs = llm.generate(formatted, sampling_params)

results = []
for output in outputs:
    candidates = [o.text for o in output.outputs]
    results.append(candidates)

with open("{output_file}", "w") as f:
    json.dump(results, f)

del llm
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
'''

    # Run subprocess
    logger.info(f"Generating {n_candidates} candidates for {len(prompts)} prompts...")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Generation failed: {result.stderr}")
        raise RuntimeError(f"Generation failed: {result.stderr}")

    # Load results
    with open(output_file) as f:
        responses = json.load(f)

    # Cleanup
    os.unlink(prompts_file)
    os.unlink(output_file)

    return responses


# =============================================================================
# EVALUATION
# =============================================================================

def find_correct_candidate(
    candidates: List[str],
    gold_sql: str,
    db_id: str,
) -> Optional[str]:
    """Find first correct candidate (if any)."""

    db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

    # Execute gold SQL
    gold_success, gold_results, _ = execute_sql(gold_sql, db_path)
    if not gold_success:
        return None

    # Try each candidate
    for candidate in candidates:
        _, predicted_sql = parse_response(candidate)
        if not predicted_sql.strip():
            continue

        pred_success, pred_results, _ = execute_sql(predicted_sql, db_path)
        if pred_success and pred_results is not None and gold_results is not None:
            if check_result_match(pred_results, gold_results):
                return predicted_sql

    return None


# =============================================================================
# DATA FORMATTING
# =============================================================================

def make_sft_example(question: str, schema: str, sql: str) -> Dict:
    """Create LlamaFactory SFT example."""
    return {
        "instruction": SYSTEM_PROMPT,
        "input": format_prompt(question, schema),
        "output": sql,
    }


def save_sft_dataset(examples: List[Dict], name: str) -> Path:
    """Save dataset for LlamaFactory."""
    output_path = DATA_DIR / f"{name}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Saved {len(examples)} examples to {output_path}")
    return output_path


# =============================================================================
# TRAINING
# =============================================================================

def train_lora(
    base_model: str,
    dataset_name: str,
    output_dir: str,
    config: Config,
) -> None:
    """Train LoRA adapter using LlamaFactory."""
    import yaml

    train_config = {
        "model_name_or_path": base_model,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": 0.1,
        "lora_target": "all",
        "dataset": dataset_name,
        "dataset_dir": str(DATA_DIR),
        "template": "mistral",
        "cutoff_len": 2048,
        "max_samples": 100000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "output_dir": output_dir,
        "logging_steps": 50,
        "save_steps": 500,
        "save_total_limit": 1,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "bf16": True,
        "gradient_checkpointing": True,
        "report_to": "none",
    }

    # Update dataset_info.json
    dataset_info_path = DATA_DIR / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": f"{dataset_name}.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}
    }

    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Save config
    config_path = PROJECT_ROOT / "configs" / f"train_{dataset_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(train_config, f)

    # Run training
    logger.info(f"Training LoRA on {base_model}...")
    cmd = [
        "llamafactory-cli", "train",
        str(config_path)
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Training failed")


def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str,
) -> None:
    """Merge LoRA adapter into base model."""
    import yaml

    logger.info(f"Merging LoRA into {output_path}...")

    config = {
        "model_name_or_path": base_model,
        "adapter_name_or_path": lora_path,
        "template": "mistral",
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False,
    }

    config_path = PROJECT_ROOT / "configs" / "merge_temp.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    cmd = ["llamafactory-cli", "export", str(config_path)]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Merge failed")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mini-STaR Retrain on Errors")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path (default: auto-detect latest merged)")
    parser.add_argument("--errors", type=str, default=None,
                        help="Errors file path (default: auto-detect latest)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations (default: 3)")
    parser.add_argument("--k", type=int, default=8,
                        help="Candidates per question (default: 8)")
    args = parser.parse_args()

    # Config with auto-detection
    config = Config()
    config.iterations = args.iterations
    config.k_candidates = args.k

    # Auto-detect or use provided base_model
    if args.base_model:
        config.base_model = Path(args.base_model)
    else:
        config.base_model = find_latest_merged()

    # Auto-detect or use provided errors file
    if args.errors:
        config.errors_file = Path(args.errors)
    else:
        config.errors_file = find_latest_errors()

    # Validate
    if config.base_model is None or not config.base_model.exists():
        logger.error(f"Base model not found: {config.base_model}")
        logger.error("Use --base_model to specify path")
        sys.exit(1)

    if config.errors_file is None or not config.errors_file.exists():
        logger.error(f"Errors file not found: {config.errors_file}")
        logger.error("Use --errors to specify path")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Mini-STaR Retrain on Errors")
    logger.info("=" * 60)
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Errors file: {config.errors_file}")
    logger.info(f"Iterations: {config.iterations}, k={config.k_candidates}")

    # Load data
    logger.info(f"Loading errors from {config.errors_file}")
    with open(config.errors_file) as f:
        errors = json.load(f)

    # Handle both formats: list or dict with 'errors' key
    if isinstance(errors, dict) and 'errors' in errors:
        # Flatten errors from categories
        all_errors = []
        for category, error_list in errors['errors'].items():
            for e in error_list:
                e['error_category'] = category
                all_errors.append(e)
        errors = all_errors

    logger.info(f"Loaded {len(errors)} errors")

    # Load schemas
    _, _, schemas = load_spider_data(SPIDER_DIR)

    current_model = str(config.base_model)
    history = []

    for iteration in range(1, config.iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")

        # Prepare prompts
        prompts = [
            format_prompt(e["question"], schemas[e["db_id"]])
            for e in errors
        ]

        # Generate candidates
        logger.info(f"Generating with model: {current_model}")
        responses = generate_with_vllm_subprocess(
            prompts=prompts,
            model_path=current_model,
            temperature=config.temperature,
            n_candidates=config.k_candidates,
        )

        # Evaluate and collect training data
        sft_data = []
        star_count = 0
        injection_count = 0

        for error, candidates in tqdm(zip(errors, responses), total=len(errors), desc="Evaluating"):
            question = error["question"]
            db_id = error["db_id"]
            gold_sql = error.get("gold_sql", error.get("query", ""))
            schema = schemas[db_id]

            # Try to find correct candidate
            correct_sql = find_correct_candidate(candidates, gold_sql, db_id)

            if correct_sql:
                # STaR: use model-generated correct SQL
                sft_data.append(make_sft_example(question, schema, correct_sql))
                star_count += 1
            else:
                # Injection: use gold SQL
                sft_data.append(make_sft_example(question, schema, gold_sql))
                injection_count += 1

        star_pct = (star_count / len(errors) * 100) if errors else 0
        inj_pct = (injection_count / len(errors) * 100) if errors else 0
        logger.info(f"STaR: {star_count} ({star_pct:.1f}%)")
        logger.info(f"Injection: {injection_count} ({inj_pct:.1f}%)")

        # Save dataset
        dataset_name = f"retrain_iter{iteration}"
        save_sft_dataset(sft_data, dataset_name)

        # Train LoRA on BASE model (always!)
        lora_output = str(MODELS_DIR / f"retrain_lora_iter{iteration}")
        train_lora(
            base_model=str(config.base_model),  # Always train on base!
            dataset_name=dataset_name,
            output_dir=lora_output,
            config=config,
        )

        # Merge LoRA into base
        merged_path = MODELS_DIR / f"retrain_iter{iteration}"
        merge_lora(
            base_model=str(config.base_model),
            lora_path=lora_output,
            output_path=str(merged_path),
        )

        # Update current model for next iteration
        current_model = str(merged_path)

        # Save history
        iter_stats = {
            "iteration": iteration,
            "model": current_model,
            "total_errors": len(errors),
            "star_count": star_count,
            "injection_count": injection_count,
            "star_ratio": (star_count / len(errors)) if errors else 0,
            "timestamp": datetime.now().isoformat(),
        }
        history.append(iter_stats)

        with open(MODELS_DIR / "retrain_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Iteration {iteration} complete. Model: {current_model}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final model: {current_model}")
    logger.info("\nHistory:")
    for h in history:
        logger.info(f"  Iter {h['iteration']}: STaR={h['star_count']}, Inj={h['injection_count']}")

    logger.info(f"\nTo evaluate: python src/evaluation/eval_dev.py --model_path {current_model} --k 16 --self_consistency")


if __name__ == "__main__":
    main()
