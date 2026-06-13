"""Read BFCL's OFFICIAL score output.

We do not score anything ourselves; we only parse what `bfcl evaluate` wrote, and
we REFUSE to report a partial multi-turn set.

NOTE ON SCHEMA: BFCL writes one score JSON per category under
`$BFCL_PROJECT_ROOT/score/<sanitized_model>/`. The exact field names below
(`accuracy`, `correct_count`, `total_count`) are read straight from BFCL's output;
if a future BFCL release renames a field, the loud asserts here report the actual
keys so the fix is a one-liner — there is no silent fallback to a guessed value.
"""
from __future__ import annotations

import json
from pathlib import Path


def _score_dir(project_root: str | Path, bfcl_model: str) -> Path:
    # BFCL sanitizes the model id (slashes -> underscores) for the score path.
    safe = bfcl_model.replace("/", "_")
    return Path(project_root) / "score" / safe


def _find_category_file(score_dir: Path, category: str) -> Path | None:
    matches = sorted(score_dir.glob(f"*{category}*score*.json"))
    return matches[0] if matches else None


def parse_multi_turn(project_root, bfcl_model, expected_categories) -> dict:
    """Return {'overall': float, 'per_category': {cat: {...}}} from BFCL's score
    files. Raises if the score dir is missing or any expected category is absent
    (no silent subsetting)."""
    score_dir = _score_dir(project_root, bfcl_model)
    if not score_dir.is_dir():
        raise RuntimeError(
            f"No score directory at {score_dir}. Did `bfcl evaluate` run, and is "
            f"BFCL_PROJECT_ROOT correct?"
        )

    per_category: dict[str, dict] = {}
    missing: list[str] = []
    for cat in expected_categories:
        f = _find_category_file(score_dir, cat)
        if f is None:
            missing.append(cat)
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        acc = data.get("accuracy")
        if acc is None:
            raise RuntimeError(
                f"No 'accuracy' field in {f}. Actual keys: {sorted(data)}. "
                f"(BFCL may have renamed the field — update scores.py.)"
            )
        per_category[cat] = {
            "accuracy": acc,
            "correct": data.get("correct_count"),
            "total": data.get("total_count"),
            "file": str(f),
        }

    if missing:
        raise RuntimeError(
            f"FULL-SET VIOLATION: missing score files for {missing} in {score_dir}. "
            f"Refusing to report a subset of multi-turn. Re-run generate + evaluate "
            f"for the full 'multi_turn' umbrella (and never with --partial-eval)."
        )

    accs = [per_category[c]["accuracy"] for c in expected_categories]
    # Leaderboard methodology: mean over the four equal-sized (200 each) subsets.
    overall = sum(accs) / len(accs)
    return {"overall": overall, "per_category": per_category}
