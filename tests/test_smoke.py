
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Run from a fresh checkout without `pip install -e .`.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from matcher.scoring.base import LABELS, DummyScorer, ScoreResult


def test_dummy_scorer_obeys_contract():
    result = DummyScorer().score("python sql", "python aws")
    assert isinstance(result, ScoreResult)
    assert result.label in LABELS
    assert 0.0 <= result.score <= 1.0


def test_registry_always_offers_baseline_scorers():
    from scorers import build_registry

    registry = build_registry()
    assert "Keyword skills (real)" in registry
    assert "Dummy (placeholder)" in registry
    # Every registered factory must build a scorer that returns a valid result.
    for factory in registry.values():
        result = factory().score("python sql pandas", "python aws sql")
        assert result.label in LABELS
        assert 0.0 <= result.score <= 1.0


def test_keyword_scorer_reports_matched_and_missing():
    from scorers import KeywordScorer

    result = KeywordScorer().score(
        "python and sql experience", "need python and aws"
    )
    assert "python" in result.matched_skills
    assert "aws" in result.missing_skills
