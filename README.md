# ReasonForge

> Iterative self-improvement for Text-to-SQL: rejection-sampling fine-tuning, verified by execution, with a STaR-inspired bootstrapping loop.

## Overview

This project implements an iterative self-improvement loop for Text-to-SQL generation. For each question, the model samples k SQL candidates, executes them against SQLite, and fine-tunes on the candidates that produce the correct result. For questions where all k candidates fail, the gold SQL is injected as a supervised example. The loop restarts from the base model each iteration.

```
Spider (7K train, 1K dev)
        |
Ministral-8B samples k=8 SQL candidates per question
        |
Execute each on SQLite, compare to the gold result
        |
   +-- correct      -> keep (self-generated signal)
   +-- 0/k correct  -> inject gold SQL (supervised signal)
        |
Fine-tune LoRA on (question, SQL), from the base model
        |
Repeat for 3 iterations
        |
Baseline 60.1%  ->  SFT 68.8%  ->  78.0% (self-consistency, k=16)
```

### Method note

Generation is direct SQL: there is no chain-of-thought or rationale step (`prompts.py`: "No Chain-of-Thought"). Without rationales, the core mechanism is rejection-sampling fine-tuning (RFT), wrapped in a STaR-style iterate-from-base loop, which places it close to ReST / iterative RFT. Correctness is judged by SQL execution against the gold result, not by a learned verifier. Note that the signal is not fully self-generated: the solved questions contribute self-generated SQL, while the unsolved subset is taught from injected gold labels.

## Results

### Dev set performance (1,034 examples)

| Model                     | Accuracy  | Method                |
| ------------------------- | --------- | --------------------- |
| Ministral-8B baseline     | 60.1%     | greedy                |
| + SFT on Spider           | 68.8%     | greedy                |
| + 3 iterations            | 78.0%     | self-consistency k=16 |

All three figures back to a logged eval JSON in `reports/`.

### Training progression

| Iteration | Train accuracy | Dev accuracy                  |
| --------- | -------------- | ----------------------------- |
| 1         | 82.2%          | 71.0%                         |
| 2         | 88.0%          | 71.7%                         |
| 3         | 88.0%          | 72.1% (greedy) / 78.0% (k=16) |

Train accuracy here is coverage: the share of sampled questions that yield at least one correct candidate (Pass@k), which is what produces harvestable training data. It is not a held-out performance number.

## Relation to prior work

| Reference                         | What it contributes                          | How this repo differs                         |
| --------------------------------- | -------------------------------------------- | --------------------------------------------- |
| STaR (Zelikman et al., 2022)      | iterate-from-base bootstrapping loop         | no rationales here                            |
| RFT (Yuan et al., 2023) / ReST (Gulcehre et al., 2023) | sample, filter correct, fine-tune, iterate | this is the closest mechanical match          |
| STaR-SQL (2025)                   | rationales + learned verifier + best-of-N    | none of these; self-consistency only          |

In short: STaR-inspired loop, RFT/ReST mechanism, execution as the correctness signal, gold injection on the questions the model cannot yet solve.

## Tech stack

| Component   | Technology                                          |
| ----------- | --------------------------------------------------- |
| Base model  | Ministral-8B-Instruct-2410 (bf16)                   |
| Fine-tuning | LoRA (rank 16) via LLaMA-Factory                    |
| Inference   | vLLM                                                |
| Dataset     | Spider (Yale), 7K train, 1K dev                     |
| Method      | Iterative RFT (STaR-inspired), k=8, execution-verified |
| Hardware    | NVIDIA H200                                          |

## Project structure

```
reasonforge/
├── src/
│   ├── star_train.py         # Main training loop (sample, execute, inject, SFT)
│   ├── star_retrain.py       # Mini retrain on errors
│   ├── sql_utils.py          # SQL execution and result matching
│   ├── prompts.py            # Prompt template and generation config (no CoT)
│   └── evaluation/
│       └── eval_dev.py       # Dev set evaluation
├── configs/
│   ├── config.yaml           # Pipeline configuration
│   └── sft_ministral8b.yaml  # SFT configuration
├── data/
│   ├── spider/               # Spider dataset (166 DBs)
│   └── errors/               # Error analysis
├── reports/                  # Logged evaluation results
└── scripts/
    └── download_spider.py    # Dataset download
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Spider dataset
python scripts/download_spider.py

# 3. Evaluate the baseline model
python src/evaluation/eval_dev.py \
    --model_path mistralai/Ministral-8B-Instruct-2410 \
    --k 1 \
    --temperature 0.0 \
    --no_self_consistency

# 4. Run the training loop
python src/star_train.py

# 5. Evaluate the fine-tuned model (k=16, self-consistency)
python src/evaluation/eval_dev.py \
    --model_path ./models/merged_iter_3 \
    --k 16 \
    --self_consistency
```

## Configuration

Main configuration in `configs/config.yaml`:

```yaml
star:
  k_candidates: 8              # Candidates sampled per question
  train_from_base: true        # Re-initialize from base each iteration
  difficulty_resampling: true  # Oversample hard questions

self_improvement:
  max_iterations: 5
  eval_sample_size: 7000
```

## License

MIT
