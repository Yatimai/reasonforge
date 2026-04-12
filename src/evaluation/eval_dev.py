#!/usr/bin/env python3
"""
Unified evaluation script for Spider dataset.

Supports:
- Any model (base or fine-tuned)
- Two strategies: "first_correct" (pass@k) or "self_consistency" (vote)
- Explicit temperature control
- Error collection for analysis

Usage:
    # BASELINE: k=1, greedy (temp=0.0)
    python src/evaluation/eval_dev.py --model_path mistralai/Mistral-Nemo-Instruct-2407 --k 1 --temperature 0.0 --no_self_consistency

    # PRODUCTION (default): k=5, self-consistency (temp=0.7)
    python src/evaluation/eval_dev.py --model_path ./models/merged_iter_5

    # BENCHMARK MAX: k=16
    python src/evaluation/eval_dev.py --model_path ./models/merged_iter_5 --k 16
"""

import argparse
import json
import sqlite3
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

from tqdm import tqdm
from loguru import logger

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SPIDER_DIR = PROJECT_ROOT / "data" / "spider"

# Add src to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Setup logging
logger.add(str(PROJECT_ROOT / "logs" / "eval_{time}.log"), rotation="10 MB")


# =============================================================================
# IMPORTS FROM CENTRALIZED MODULES
# =============================================================================

from prompts import SYSTEM_PROMPT, format_prompt, get_stop_tokens, get_max_tokens, get_temperature
from sql_utils import (
    execute_sql,
    parse_response,
    normalize_sql_result,
    check_result_match,
    load_spider_data as _load_spider_data_base,
    normalize_sql,
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spider_data() -> Tuple[List[Dict], List[Dict], Dict[str, str]]:
    """Load Spider train and dev data using centralized function."""
    train_data, dev_data, schemas = _load_spider_data_base(SPIDER_DIR)
    logger.info(f"Loaded {len(train_data)} train, {len(dev_data)} dev examples")
    return train_data, dev_data, schemas


# =============================================================================
# GENERATION WITH vLLM
# =============================================================================

def generate_with_vllm(
    prompts: List[str],
    model_path: str,
    k_candidates: int = 1,
    temperature: float = None,
) -> List[List[str]]:
    """
    Generate k candidates per prompt using vLLM.
    Returns list of lists: for each prompt, k candidate responses.
    """
    import os
    os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    logger.info(f"Loading model: {model_path}")

    stop_tokens = get_stop_tokens()
    max_tokens = get_max_tokens()

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Temperature: use provided or default based on k
    if temperature is not None:
        temp = temperature
    elif k_candidates > 1:
        temp = get_temperature("training")  # 0.7 for diversity
    else:
        temp = get_temperature("inference")  # 0.1 for consistency

    sampling_params = SamplingParams(
        temperature=temp,
        max_tokens=max_tokens,
        n=k_candidates,
        stop=stop_tokens,
    )

    logger.info(f"Generating {k_candidates} candidates per prompt (temp={temp})...")

    # Format prompts with chat template
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)

    outputs = llm.generate(formatted_prompts, sampling_params)

    # Extract responses
    all_responses = []
    for output in outputs:
        candidates = [o.text for o in output.outputs]
        all_responses.append(candidates)

    # Cleanup
    del llm
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_responses


# =============================================================================
# EVALUATION STRATEGIES
# =============================================================================

def evaluate_first_correct(item: Dict, candidates: List[str], schemas: Dict[str, str]) -> Dict:
    """
    Strategy: First Correct Wins (pass@k)
    If ANY candidate is correct, the question is considered correct.
    """
    question = item['question']
    gold_sql = item['query']
    db_id = item['db_id']
    db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

    gold_success, gold_results, gold_error = execute_sql(gold_sql, db_path)

    if not gold_success:
        logger.warning(f"Gold SQL failed for {db_id}: {gold_sql[:100]}... Error: {gold_error}")
        return {
            'question': question,
            'db_id': db_id,
            'gold_sql': gold_sql,
            'is_correct': False,
            'predicted_sql': None,
            'error': 'Gold SQL failed to execute',
            'gold_error': True,
        }

    # Try each candidate
    for i, candidate in enumerate(candidates):
        reasoning, predicted_sql = parse_response(candidate)

        if not predicted_sql.strip():
            continue

        pred_success, pred_results, pred_error = execute_sql(predicted_sql, db_path)

        if pred_success and check_result_match(pred_results, gold_results):
            return {
                'question': question,
                'db_id': db_id,
                'gold_sql': gold_sql,
                'is_correct': True,
                'predicted_sql': predicted_sql,
                'candidate_idx': i + 1,
                'reasoning': reasoning,
            }

    # No correct candidate
    reasoning, predicted_sql = parse_response(candidates[0]) if candidates else ("", "")
    return {
        'question': question,
        'db_id': db_id,
        'gold_sql': gold_sql,
        'is_correct': False,
        'predicted_sql': predicted_sql,
        'error': 'No correct candidate found',
    }


def evaluate_self_consistency(item: Dict, candidates: List[str], schemas: Dict[str, str]) -> Dict:
    """
    Strategy: Self-Consistency (vote)
    Execute all candidates, vote for most common result.
    """
    question = item['question']
    gold_sql = item['query']
    db_id = item['db_id']
    db_path = SPIDER_DIR / "database" / db_id / f"{db_id}.sqlite"

    gold_success, gold_results, gold_error = execute_sql(gold_sql, db_path)

    if not gold_success:
        logger.warning(f"Gold SQL failed for {db_id}: {gold_sql[:100]}... Error: {gold_error}")
        return {
            'question': question,
            'db_id': db_id,
            'gold_sql': gold_sql,
            'is_correct': False,
            'predicted_sql': None,
            'error': 'Gold SQL failed to execute',
            'gold_error': True,
        }

    # Execute all candidates and collect results
    results_map = {}  # normalized_result -> (sql, count, actual_results)

    for candidate in candidates:
        reasoning, predicted_sql = parse_response(candidate)

        if not predicted_sql.strip():
            continue

        pred_success, pred_results, _ = execute_sql(predicted_sql, db_path)

        if pred_success and pred_results is not None:
            # Normalize results for comparison
            try:
                norm_key = str(sorted([str(row) for row in pred_results]))
            except:
                norm_key = str(pred_results)

            if norm_key in results_map:
                results_map[norm_key] = (results_map[norm_key][0], results_map[norm_key][1] + 1, pred_results)
            else:
                results_map[norm_key] = (predicted_sql, 1, pred_results)

    if not results_map:
        reasoning, predicted_sql = parse_response(candidates[0]) if candidates else ("", "")
        return {
            'question': question,
            'db_id': db_id,
            'gold_sql': gold_sql,
            'is_correct': False,
            'predicted_sql': predicted_sql,
            'error': 'No valid SQL execution',
        }

    # Vote: pick most common result
    best_key = max(results_map.keys(), key=lambda k: results_map[k][1])
    best_sql, vote_count, best_results = results_map[best_key]

    # Check if voted result matches gold
    is_correct = check_result_match(best_results, gold_results)

    return {
        'question': question,
        'db_id': db_id,
        'gold_sql': gold_sql,
        'is_correct': is_correct,
        'predicted_sql': best_sql,
        'vote_count': vote_count,
        'total_candidates': len(candidates),
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    model_path: str,
    split: str = "dev",
    sample_size: int = 0,
    k_candidates: int = 1,
    self_consistency: bool = False,
    temperature: float = None,
) -> Dict:
    """Run evaluation with specified strategy."""

    strategy = "self_consistency" if self_consistency else "first_correct"

    # Determine temperature
    if temperature is not None:
        temp_display = temperature
    elif k_candidates > 1:
        temp_display = get_temperature("training")  # 0.7
    else:
        temp_display = get_temperature("inference")  # 0.1

    print("=" * 60)
    print("Spider SQL Evaluation")
    print("=" * 60)
    print(f"   Model: {model_path}")
    print(f"   Split: {split}")
    print(f"   k candidates: {k_candidates}")
    print(f"   Strategy: {strategy}")
    print(f"   Temperature: {temp_display}")
    print(f"   Sample size: {sample_size if sample_size > 0 else 'ALL'}")
    print()

    # Load data
    print("Loading data...")
    train_data, dev_data, schemas = load_spider_data()

    eval_data = dev_data if split == "dev" else train_data

    # Sample if needed
    if sample_size > 0 and sample_size < len(eval_data):
        import random
        eval_data = random.sample(eval_data, sample_size)

    print(f"Evaluating on {len(eval_data)} examples...")

    # Prepare prompts
    prompts = []
    for item in eval_data:
        schema = schemas.get(item['db_id'], "")
        prompt = format_prompt(item['question'], schema)
        prompts.append(prompt)

    # Generate candidates
    print(f"\nGenerating {k_candidates} candidates per question...")
    all_candidates = generate_with_vllm(
        prompts=prompts,
        model_path=model_path,
        k_candidates=k_candidates,
        temperature=temperature,
    )

    # Evaluate with chosen strategy
    print(f"\nEvaluating with strategy: {strategy}...")
    results = []
    correct = 0

    eval_fn = evaluate_self_consistency if self_consistency else evaluate_first_correct

    for item, candidates in tqdm(zip(eval_data, all_candidates), total=len(eval_data)):
        result = eval_fn(item, candidates, schemas)
        results.append(result)
        if result['is_correct']:
            correct += 1

    # Calculate metrics
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print(f"RESULTS ({split.upper()} SET)")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"Strategy: {strategy}")
    print(f"k candidates: {k_candidates}")

    # Show candidate distribution (first_correct only)
    if not self_consistency and k_candidates > 1:
        print("\nCorrect by candidate position:")
        candidate_dist = {}
        for r in results:
            if r['is_correct']:
                idx = r.get('candidate_idx', 1)
                candidate_dist[idx] = candidate_dist.get(idx, 0) + 1

        cumul = 0
        for idx in sorted(candidate_dist.keys()):
            count = candidate_dist[idx]
            cumul += count
            pct = (cumul / total * 100) if total > 0 else 0
            print(f"   k={idx}: {cumul} ({pct:.1f}%) +{count}")

    # Save results
    output_file = PROJECT_ROOT / "reports" / f"eval_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'model': model_path,
            'split': split,
            'k_candidates': k_candidates,
            'strategy': strategy,
            'temperature': temp_display,
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Collect errors
    errors = [r for r in results if not r['is_correct']]
    errors_file = PROJECT_ROOT / "data" / "errors" / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    errors_file.parent.mkdir(parents=True, exist_ok=True)
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"Errors saved to: {errors_file} ({len(errors)} errors)")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SQL model on Spider dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # BASELINE: k=1, greedy
    python src/evaluation/eval_dev.py --model_path mistralai/Ministral-8B-Instruct-2410 --k 1 --temperature 0.0 --no_self_consistency

    # STaR EVALUATION: k=16, self-consistency
    python src/evaluation/eval_dev.py --model_path ./models/merged_iter_3 --k 16 --self_consistency
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (base or fine-tuned)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "dev"],
        default="dev",
        help="Dataset split (default: dev)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Number of samples, 0=all (default: 0)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of candidates (default: 5)"
    )
    parser.add_argument(
        "--self_consistency",
        action="store_true",
        default=True,
        help="Use self-consistency (vote) instead of first-correct (default: True)"
    )
    parser.add_argument(
        "--no_self_consistency",
        action="store_true",
        help="Disable self-consistency, use first-correct instead"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature (0.0=greedy, 0.1=production, 0.7=diversity). Default: auto based on k"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Handle --no_self_consistency flag
    use_self_consistency = args.self_consistency and not args.no_self_consistency

    run_evaluation(
        model_path=args.model_path,
        split=args.split,
        sample_size=args.sample_size,
        k_candidates=args.k,
        self_consistency=use_self_consistency,
        temperature=args.temperature,
    )
