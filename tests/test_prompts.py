"""Tests for src/prompts.py — prompt formatting and generation config."""

from src.prompts import (
    GENERATION_CONFIG,
    SYSTEM_PROMPT,
    format_prompt,
    get_max_tokens,
    get_stop_tokens,
    get_temperature,
)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    def test_basic_interpolation(self):
        result = format_prompt("How many users?", "CREATE TABLE users (id, name)")
        assert "How many users?" in result
        assert "CREATE TABLE users (id, name)" in result
        assert result.endswith("SQL:")

    def test_special_characters_in_question(self):
        # Question with quotes, newlines, braces — should not break the template
        question = 'List users where name = "Alice" and {age} > 30\nGroup by status'
        result = format_prompt(question, "schema")
        assert question in result

    def test_empty_schema(self):
        result = format_prompt("question", "")
        assert "question" in result
        assert "SQL:" in result


# ---------------------------------------------------------------------------
# get_temperature
# ---------------------------------------------------------------------------


class TestGetTemperature:
    def test_training_returns_07(self):
        assert get_temperature("training") == 0.7

    def test_inference_returns_01(self):
        assert get_temperature("inference") == 0.1

    def test_default_is_inference(self):
        assert get_temperature() == 0.1

    def test_unknown_mode_falls_back_to_inference(self):
        # Anything other than "training" returns inference temp
        assert get_temperature("anything_else") == 0.1


# ---------------------------------------------------------------------------
# get_stop_tokens
# ---------------------------------------------------------------------------


class TestGetStopTokens:
    def test_returns_expected_tokens(self):
        assert get_stop_tokens() == ["</s>"]

    def test_returns_a_copy(self):
        """Critical: callers should not mutate the shared GENERATION_CONFIG."""
        first = get_stop_tokens()
        first.append("MUTATED")
        second = get_stop_tokens()
        assert "MUTATED" not in second
        assert second == ["</s>"]


# ---------------------------------------------------------------------------
# get_max_tokens
# ---------------------------------------------------------------------------


class TestGetMaxTokens:
    def test_returns_1024(self):
        assert get_max_tokens() == 1024


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_system_prompt_is_non_empty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_mentions_sqlite(self):
        # Regression: the prompt must mention SQLite to anchor the dialect
        assert "SQLite" in SYSTEM_PROMPT or "sqlite" in SYSTEM_PROMPT.lower()

    def test_generation_config_has_expected_keys(self):
        expected_keys = {
            "temperature_training",
            "temperature_inference",
            "max_tokens",
            "stop_tokens",
            "gpu_memory_utilization",
            "dtype",
        }
        assert expected_keys.issubset(GENERATION_CONFIG.keys())

    def test_generation_config_temperatures_are_floats(self):
        assert isinstance(GENERATION_CONFIG["temperature_training"], float)
        assert isinstance(GENERATION_CONFIG["temperature_inference"], float)
        assert 0.0 <= GENERATION_CONFIG["temperature_inference"] < GENERATION_CONFIG["temperature_training"]
