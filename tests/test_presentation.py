"""Presentation layer sync tests.

Guards invariants between committed source files (strategies.json, metrics.json,
sensitivity.json, RANKING.md) and the gallery presentation layer (pieces.json,
viz/assets/, viz/data/). These tests have no network calls, no headless browser,
and no subprocess — pure offline file I/O.

Invariants protected:
1. Every strategy in strategies.json has a corresponding piece in pieces.json.
2. Every pieces.json entry has a viz/pieces/{id}/index.html on disk.
3. No stale pieces exist for deleted strategies (two-way check).
4. Every metric key appears in the glossary (if glossary.js exists).
5. piece.js and leaderboard.js contain no hardcoded metric float literals.
6. RANKING.md leaderboard table scores match committed sensitivity.json and
   leaderboard.json (param robustness and walk-forward consistency values).
7. The displayed strategy thesis traces to a committed source file; this becomes
   a hard guard once suggestion 02/03 extract thesis text from README verbatim.

Note on field name mapping: strategies.json uses a short numeric 'id' (e.g. "01")
and a full 'name' field (e.g. "01-dual-ema-momentum"). The pieces.json registry
uses the full name as its 'id'. All piece-matching logic therefore uses the
'name' field from strategies.json, not 'id'.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
VIZ = ROOT / "viz"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_strategies() -> list:
    """Load and parse strategies.json, returning the list of strategy entries."""
    return json.loads((ROOT / "strategies.json").read_text())


def _load_pieces() -> list:
    """Load and parse viz/pieces.json, returning the list of piece entries."""
    return json.loads((VIZ / "pieces.json").read_text())


def _strip_js_single_line_comments(source: str) -> str:
    """Remove // single-line JS comments to avoid false positives in float detection.

    Multi-line /* */ comments are not stripped here because piece.js and
    leaderboard.js use only single-line comments at the top of the file.
    """
    return re.sub(r"//[^\n]*", "", source)


def _parse_ranking_table(ranking_text: str) -> list:
    """Extract strategy rows from RANKING.md's leaderboard Markdown table.

    Each returned dict has:
        strategy_id (str)  – extracted from the README link in the Strategy column.
        param_robustness (float|None) – **X.XXX** on regime_switch, if present.
        oos_consistency (float|None)  – oos_consistency: X.XX, if present.

    The parser stops at the first non-table line after the header, so it is
    resilient to sections that follow the table.
    """
    rows = []
    header_seen = False
    for line in ranking_text.splitlines():
        stripped = line.strip()
        if re.match(r"\|\s*Rank\s*\|", stripped):
            header_seen = True
            continue
        if stripped.startswith("|---") or stripped.startswith("| ---"):
            continue
        if header_seen and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(cells) < 7:
                continue

            # Strategy column: [Title](strategies/ID/README.md)
            link_m = re.search(r"\(strategies/([^/]+)/README\.md\)", cells[1])
            if not link_m:
                continue
            sid = link_m.group(1)

            # Param Robustness column (index 4): **0.085** on regime_switch ...
            pr_m = re.search(r"\*\*([\d.]+)\*\*\s+on\s+regime_switch", cells[4])
            param_rob = float(pr_m.group(1)) if pr_m else None

            # Walk-fwd Sharpe column (index 6): oos_consistency: X.XX ...
            # Normalise Unicode minus so the same regex handles both signs.
            wf_cell = cells[6].replace("−", "-")
            oc_m = re.search(r"oos_consistency:\s*(\d+\.\d+)", wf_cell)
            oos_con = float(oc_m.group(1)) if oc_m else None

            rows.append(
                {
                    "strategy_id": sid,
                    "param_robustness": param_rob,
                    "oos_consistency": oos_con,
                }
            )
        elif header_seen and not stripped.startswith("|"):
            break

    return rows


# ---------------------------------------------------------------------------
# Required acceptance-criterion tests (1–5)
# ---------------------------------------------------------------------------


def test_every_strategy_has_a_piece():
    """Every strategies.json entry must have a matching piece in pieces.json.

    Uses the 'name' field from strategies.json (e.g. "01-dual-ema-momentum")
    rather than the short numeric 'id' field, because pieces.json uses the
    full name string as its piece id.
    """
    strategies = _load_strategies()
    pieces = {p["id"] for p in _load_pieces()}
    missing = [s["name"] for s in strategies if s["name"] not in pieces]
    assert not missing, f"strategies.json entries missing from pieces.json: {missing}"


def test_every_piece_directory_exists():
    """Every piece listed in pieces.json must have viz/pieces/{id}/index.html on disk.

    Fails on the first missing directory so the path is reported clearly.
    """
    for piece in _load_pieces():
        expected = VIZ / "pieces" / piece["id"] / "index.html"
        assert expected.exists(), (
            f"pieces.json references '{piece['id']}' but {expected} not found"
        )


def test_pieces_json_ids_match_strategies_json():
    """Two-way check: every piece id must appear in strategies.json or be 'leaderboard'.

    Catches stale entries left behind when a strategy is renamed or deleted —
    the drift direction opposite to test_every_strategy_has_a_piece.
    """
    strategy_names = {s["name"] for s in _load_strategies()} | {"leaderboard"}
    stale = [p["id"] for p in _load_pieces() if p["id"] not in strategy_names]
    assert not stale, f"pieces.json has entries not in strategies.json: {stale}"


def test_metrics_keys_have_glossary_entries():
    """Every top-level key in strategies/*/metrics.json must appear in glossary.js.

    Skips gracefully if viz/assets/glossary.js does not yet exist (forward-compatible
    with suggestion 01 which will add it). When glossary.js is present, the test
    parses the top-level GLOSSARY object keys and ensures full coverage of every
    metric key observed across all strategy metric files.
    """
    glossary_js = VIZ / "assets" / "glossary.js"
    if not glossary_js.exists():
        # Forward-compatible: glossary will be added by suggestion 01
        return

    text = glossary_js.read_text()
    # Matches two-space-indented keys: e.g. "  sharpe:" or "  max_drawdown:"
    glossary_keys = set(re.findall(r"^\s{2}(\w+)\s*:", text, re.M))

    metric_keys: set = set()
    for mpath in ROOT.glob("strategies/*/metrics.json"):
        data = json.loads(mpath.read_text())
        # metrics.json is keyed by dataset name; each value is a dict of metric keys.
        for dataset_metrics in data.values():
            if isinstance(dataset_metrics, dict):
                metric_keys.update(dataset_metrics.keys())

    missing = metric_keys - glossary_keys
    assert not missing, f"metrics.json keys without glossary entries: {sorted(missing)}"


def test_no_hardcoded_numbers_in_piece_js():
    """piece.js must not contain bare float literals with 3+ decimal places.

    Such literals would represent hardcoded metric values (sharpe, CAGR, etc.)
    that cannot be traced to loaded JSON. Acceptable rendering constants —
    thresholds like 0.6 for OOS-consistency colouring, or comparison values like
    1 for equity break-even — have at most 2 significant decimal places, so the
    regex `0.\\d{3,}` (three or more decimal digits) is a precise discriminator.

    Edge: the pattern is applied after stripping // comments so that commented-out
    numbers do not trigger false positives.
    """
    piece_js = (VIZ / "assets" / "piece.js").read_text()
    no_comments = _strip_js_single_line_comments(piece_js)
    suspicious = re.findall(r"(?<![.\w])0\.\d{3,}(?!\w)", no_comments)
    assert not suspicious, (
        f"piece.js appears to contain hardcoded metric values: {suspicious[:5]}. "
        "All numbers must be read from loaded JSON, not embedded in source."
    )


# ---------------------------------------------------------------------------
# Additional tests from orchestrator guidance
# ---------------------------------------------------------------------------


def test_no_hardcoded_numbers_in_leaderboard_js():
    """leaderboard.js must not contain bare float literals with 3+ decimal places.

    Extends the piece.js check to the leaderboard page, guarding against
    suggestion 04 workers who might embed RANKING.md per-dimension scores as
    static JS constants. All numeric data rendered by leaderboard.js must
    originate from leaderboard.json loaded at runtime.
    """
    leaderboard_js = (VIZ / "assets" / "leaderboard.js").read_text()
    no_comments = _strip_js_single_line_comments(leaderboard_js)
    suspicious = re.findall(r"(?<![.\w])0\.\d{3,}(?!\w)", no_comments)
    assert not suspicious, (
        f"leaderboard.js appears to contain hardcoded metric values: {suspicious[:5]}. "
        "All numbers must be derived from loaded JSON, not embedded in source."
    )


def test_leaderboard_ranking_traces_to_ranking_md():
    """RANKING.md leaderboard table scores must match committed data source files.

    Parses two columns from the RANKING.md leaderboard table:
    - Param Robustness: the sensitivity_score on regime_switch from sensitivity.json.
    - Walk-fwd Sharpe: the mean_oos_consistency from leaderboard.json.

    Fails loudly when RANKING.md is edited with stale numbers after data files
    are regenerated. Skips if RANKING.md has no parseable table or if
    viz/data/leaderboard.json has not been built yet.
    """
    ranking_text = (ROOT / "RANKING.md").read_text()
    rows = _parse_ranking_table(ranking_text)
    if not rows:
        pytest.skip("RANKING.md leaderboard table not yet parseable")

    lb_path = VIZ / "data" / "leaderboard.json"
    if not lb_path.exists():
        pytest.skip(
            "viz/data/leaderboard.json not built yet; run python3 viz/build.py first"
        )

    leaderboard = json.loads(lb_path.read_text())
    lb_by_id = {s["id"]: s for s in leaderboard["strategies"]}

    for row in rows:
        sid = row["strategy_id"]
        if sid not in lb_by_id:
            # Strategy in RANKING.md not yet in leaderboard — not a drift error
            continue

        # Check walk-forward consistency value against leaderboard.json
        if row["oos_consistency"] is not None:
            actual = lb_by_id[sid]["mean_oos_consistency"]
            stated = row["oos_consistency"]
            assert abs(actual - stated) < 0.01, (
                f"RANKING.md states oos_consistency {stated:.2f} for {sid} but "
                f"leaderboard.json has {actual:.4f}"
            )

        # Check param robustness (sensitivity_score) against sensitivity.json
        if row["param_robustness"] is not None:
            sens_path = ROOT / "strategies" / sid / "sensitivity.json"
            if not sens_path.exists():
                continue
            sens = json.loads(sens_path.read_text())
            actual_sens = sens.get("regime_switch", {}).get("sensitivity_score")
            if actual_sens is None:
                continue
            stated = row["param_robustness"]
            # Allow for rounding to 3 decimal places in the RANKING.md text
            assert abs(actual_sens - stated) < 0.001, (
                f"RANKING.md states param_robustness {stated:.3f} for {sid} on "
                f"regime_switch but sensitivity.json has {actual_sens:.6f}"
            )


def test_strategy_thesis_verbatim_in_readme():
    """The 'idea' field in viz/data/*.json must be a verbatim substring of its README.md.

    Guards against thesis text being paraphrased or fabricated rather than
    extracted from committed strategy README files (per suggestion 02/03).

    Current behaviour (pre-suggestion-02): the 'idea' field is authored inside
    viz/build.py as a hardcoded string, not extracted from README.md. This test
    skips when NO strategy has its idea verbatim in its README, which is the
    current state. Once suggestion 02/03 move idea text into READMEs and update
    build.py to extract it, the skip condition no longer triggers and the test
    becomes a hard drift guard: any strategy whose idea diverges from its README
    will cause a failure.

    Edge: if viz/data/*.json files have not been generated yet, the test skips.
    """
    data_dir = VIZ / "data"
    if not data_dir.exists():
        pytest.skip("viz/data/ not built yet; run python3 viz/build.py first")

    strategies = _load_strategies()
    pre_s02: list = []  # strategies whose idea is not yet verbatim in README
    confirmed: list = []  # strategies confirmed to have idea verbatim in README

    for strategy in strategies:
        sid = strategy["name"]
        data_file = data_dir / f"{sid}.json"
        if not data_file.exists():
            pre_s02.append(sid)
            continue

        # Extract 'idea' from the emitted JSON without loading the entire large file.
        raw = data_file.read_text()
        idea_m = re.search(r'"idea"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if not idea_m:
            pre_s02.append(sid)
            continue
        idea = (
            idea_m.group(1)
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\\\", "\\")
        )

        readme = ROOT / "strategies" / sid / "README.md"
        if not readme.exists():
            pre_s02.append(sid)
            continue

        if idea in readme.read_text():
            confirmed.append(sid)
        else:
            pre_s02.append(sid)

    if not confirmed:
        # Pre-suggestion-02: no strategy has its idea verbatim in README yet.
        pytest.skip(
            f"No strategy has its 'idea' field verbatim in its README.md: {pre_s02}. "
            "This test becomes a hard guard once suggestion 02/03 extract thesis "
            "text from committed README files rather than hardcoding in build.py."
        )

    # If some strategies are confirmed but others are not, that is real drift.
    assert not pre_s02, (
        f"Some strategies have 'idea' not verbatim in README.md: {pre_s02}. "
        f"Confirmed ok: {confirmed}. "
        "All strategies must source their thesis text from their README.md."
    )


# ---------------------------------------------------------------------------
# Edge-case and failure-mode tests
# ---------------------------------------------------------------------------


def test_pieces_json_exists():
    """viz/pieces.json must exist on disk (missing file is the simplest drift)."""
    assert (VIZ / "pieces.json").exists(), "viz/pieces.json is missing"


def test_ranking_md_exists():
    """RANKING.md must exist at the repo root."""
    assert (ROOT / "RANKING.md").exists(), "RANKING.md is missing at repo root"


def test_hardcoded_number_regex_does_not_false_positive_on_rendering_constants():
    """The 3-decimal-place float regex must not flag legitimate rendering constants.

    Verifies that short constants used for UI logic (0.6 OOS threshold, 0.02
    cell-colour boundary, comparison to 1 for break-even) do not trigger the
    heuristic used in test_no_hardcoded_numbers_in_piece_js and
    test_no_hardcoded_numbers_in_leaderboard_js.
    """
    pattern = re.compile(r"(?<![.\w])0\.\d{3,}(?!\w)")

    # Rendering constants that appear in the actual JS files — must not match.
    safe_snippets = [
        "oos_consistency >= 0.6 ? 'pos' : 'neg'",  # 1-decimal threshold
        "v > 0.02 ? 'cell-pos' : 'cell-neg'",        # 2-decimal UI threshold
        "a * 0.34).toFixed(3)",                        # 2-decimal alpha
        "fin >= 1 ? cssVar",                            # integer comparison
    ]
    for snippet in safe_snippets:
        assert not pattern.search(snippet), (
            f"Regex incorrectly flagged rendering constant in: {snippet!r}"
        )

    # A real metric value that should be caught — must match.
    assert pattern.search("const SHARPE = 0.689;"), (
        "Regex failed to detect hardcoded metric literal 0.689"
    )


def test_leaderboard_ranking_table_parse_returns_all_strategies():
    """_parse_ranking_table must return one row per strategy in RANKING.md.

    Guards against the helper silently returning an empty list when the table
    format changes — if it returns fewer rows than there are strategies, the
    synchronisation tests above would produce false negatives.
    """
    ranking_text = (ROOT / "RANKING.md").read_text()
    rows = _parse_ranking_table(ranking_text)
    n_strategies = len(_load_strategies())
    assert len(rows) == n_strategies, (
        f"_parse_ranking_table returned {len(rows)} rows but there are "
        f"{n_strategies} strategies — RANKING.md table format may have changed"
    )
