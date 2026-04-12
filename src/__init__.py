"""
ReasonForge: Core source modules.

This package contains the main implementation of the STaR (Self-Taught Reasoner)
pipeline for iterative fine-tuning of language models on Text-to-SQL tasks.

Modules:
    star_train: Main STaR training pipeline (5 × 2000 × k=8)
    star_retrain: Mini-STaR for production drift correction (2 × 500 × k=8)
    sql_utils: SQL execution, parsing, and result matching
    prompts: Centralized prompts and generation configuration
    evaluation: Model evaluation on Spider dev set
"""
