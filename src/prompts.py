#!/usr/bin/env python3
"""
Centralized prompts and generation parameters for STaR-SQL.

The prompt includes schema (with optional cell values loaded by sql_utils).
No Chain-of-Thought - direct SQL generation.

Usage:
    from prompts import format_prompt, SYSTEM_PROMPT, GENERATION_CONFIG
"""

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = "You are an expert SQL developer. Given a database schema and a question, write the correct SQLite query."

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================

PROMPT_TEMPLATE = """Given the following database schema:

{schema}

Question: {question}

Write the SQL query that answers the question.

SQL:"""


def format_prompt(question: str, schema: str) -> str:
    """Format prompt with schema and question."""
    return PROMPT_TEMPLATE.format(question=question, schema=schema)


# Backward compatibility alias
format_simple_prompt = format_prompt


# =============================================================================
# GENERATION CONFIG - Consistent across all files
# =============================================================================

GENERATION_CONFIG = {
    # Temperature for diverse candidate generation (STaR-SQL)
    "temperature_training": 0.7,
    "temperature_inference": 0.1,

    # Max tokens
    "max_tokens": 1024,

    # Stop tokens
    "stop_tokens": ["</s>"],

    # vLLM settings
    "gpu_memory_utilization": 0.85,
    "dtype": "bfloat16",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_stop_tokens() -> list:
    """Get stop tokens for generation."""
    return GENERATION_CONFIG["stop_tokens"].copy()


def get_temperature(mode: str = "inference") -> float:
    """
    Get temperature based on mode.

    Args:
        mode: "training" (0.7 for diversity) or "inference" (0.1 for consistency)
    """
    if mode == "training":
        return GENERATION_CONFIG["temperature_training"]
    return GENERATION_CONFIG["temperature_inference"]


def get_max_tokens() -> int:
    """Get max tokens for generation."""
    return GENERATION_CONFIG["max_tokens"]
