"""Structural gate: validates strategies.json schema and per-strategy file layout.

These tests must pass on an empty strategies.json with no strategy directories.
When strategies are added, each must satisfy all checks before being merged.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
STRATEGIES_JSON = ROOT / "strategies.json"
STRATEGIES_DIR = ROOT / "strategies"


def _load_registry() -> list:
    """Load and parse strategies.json, raising informative errors on failure."""
    with open(STRATEGIES_JSON) as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# strategies.json schema
# ---------------------------------------------------------------------------


class TestStrategiesJsonSchema:
    def test_strategies_json_exists(self):
        assert STRATEGIES_JSON.is_file(), "strategies.json is missing at repo root"

    def test_strategies_json_is_valid_json(self):
        try:
            _load_registry()
        except json.JSONDecodeError as e:
            pytest.fail(f"strategies.json is not valid JSON: {e}")

    def test_strategies_json_is_array(self):
        data = _load_registry()
        assert isinstance(data, list), "strategies.json must be a JSON array"

    def test_each_entry_has_name_field(self):
        for i, entry in enumerate(_load_registry()):
            assert "name" in entry, f"Entry {i} is missing required field 'name'"
            assert isinstance(entry["name"], str), f"Entry {i} 'name' must be a string"
            assert entry["name"].strip(), f"Entry {i} 'name' must not be empty"

    def test_names_are_unique(self):
        names = [e["name"] for e in _load_registry()]
        assert len(names) == len(set(names)), "Duplicate strategy names in strategies.json"


# ---------------------------------------------------------------------------
# Per-strategy file presence (skipped when registry is empty)
# ---------------------------------------------------------------------------


class TestStrategyDirectories:
    def test_each_entry_has_directory(self):
        for entry in _load_registry():
            strategy_dir = STRATEGIES_DIR / entry["name"]
            assert strategy_dir.is_dir(), (
                f"Directory missing for strategy '{entry['name']}': {strategy_dir}"
            )

    def test_each_entry_has_readme(self):
        for entry in _load_registry():
            readme = STRATEGIES_DIR / entry["name"] / "README.md"
            assert readme.is_file(), (
                f"README.md missing for strategy '{entry['name']}'"
            )

    def test_each_entry_has_metrics_json(self):
        for entry in _load_registry():
            mj = STRATEGIES_DIR / entry["name"] / "metrics.json"
            assert mj.is_file(), f"metrics.json missing for strategy '{entry['name']}'"

    def test_each_entry_has_strategy_ts(self):
        for entry in _load_registry():
            ts = STRATEGIES_DIR / entry["name"] / "strategy.ts"
            assert ts.is_file(), f"strategy.ts missing for strategy '{entry['name']}'"


# ---------------------------------------------------------------------------
# strategy.ts content checks (skipped when registry is empty)
# ---------------------------------------------------------------------------


class TestStrategyTypeScript:
    def test_strategy_ts_contains_add_order(self):
        for entry in _load_registry():
            ts = STRATEGIES_DIR / entry["name"] / "strategy.ts"
            content = ts.read_text(encoding="utf-8")
            assert "addOrder" in content, (
                f"strategy.ts for '{entry['name']}' has no addOrder call"
            )

    def test_strategy_ts_balanced_braces(self):
        for entry in _load_registry():
            ts = STRATEGIES_DIR / entry["name"] / "strategy.ts"
            content = ts.read_text(encoding="utf-8")
            open_count = content.count("{")
            close_count = content.count("}")
            assert open_count == close_count, (
                f"Unbalanced braces in strategy.ts for '{entry['name']}': "
                f"{open_count} '{{' vs {close_count} '}}'"
            )

    def test_metrics_json_is_valid_json(self):
        for entry in _load_registry():
            mj = STRATEGIES_DIR / entry["name"] / "metrics.json"
            try:
                with open(mj) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"metrics.json for '{entry['name']}' is invalid JSON: {e}")


# ---------------------------------------------------------------------------
# Empty registry passes trivially (regression guard)
# ---------------------------------------------------------------------------


class TestEmptyRegistryPasses:
    def test_empty_strategies_json_passes_all_schema_checks(self):
        data = _load_registry()
        # If registry is empty, there's nothing to validate — this is the
        # expected initial state and must not fail any of the above.
        if len(data) == 0:
            assert True  # explicit no-op for clarity
