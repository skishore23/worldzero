"""The post-hoc comparison uses authenticated original traces, not edited scores."""

import copy
import json
from pathlib import Path

import pytest

from worldzero.protocol import read_trace


ROOT = Path(__file__).resolve().parents[1] / "evidence/verification-study"


def original_case():
    results = json.loads((ROOT / "results.json").read_text())
    row = next(r for r in results["rows"] if r["policy"] == "verify" and r["arm"] == "active"
               and r["seed"] == 1792432103 and r["family_id"] == "worldzero:catalysis")
    return row, read_trace(ROOT / row["trace"]["path"])


def test_rescore_changes_only_the_scoring_interpretation():
    from scripts.verification_rescore import rescore_cell
    row, trace = original_case()
    before = copy.deepcopy(row)
    result = rescore_cell(row, trace)
    assert row == before
    assert result["old_level"] == 2 and result["new_level"] == 4
    assert result["survived"] is True
    assert result["terminal_retention"] is False
    assert result["trace"] == row["trace"]


def test_rescore_refuses_changed_trace_or_inconsistent_old_score():
    from scripts.verification_rescore import rescore_cell
    row, trace = original_case()
    changed = copy.deepcopy(trace)
    changed["result"]["survived"] = False
    with pytest.raises(ValueError, match="trace digest"):
        rescore_cell(row, changed)
    row["level"] = 4
    with pytest.raises(ValueError, match="historical score"):
        rescore_cell(row, trace)
