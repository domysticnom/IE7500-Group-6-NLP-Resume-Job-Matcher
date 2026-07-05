"""Scorer registry and adapters — the integration seam the Streamlit app builds on.

The app never imports a specific model. It asks this module for whatever scorers
are available in the current checkout and renders whatever ``ScoreResult`` they
return. As teammates finish ``TfidfScorer`` / ``DistilBertScorer``, those light up
in the app automatically with no change to the UI code.

Design contract lives in ``matcher.scoring.base`` (do not change its signatures).
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

# Make ``matcher`` importable even before ``pip install -e .`` by adding src/ to
# the path. Once the package is installed this is a harmless no-op.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from matcher.scoring.base import Scorer, ScoreResult, DummyScorer, LABELS
from matcher.scoring.skill_extraction import extract_skills
from matcher.scoring.keyword_overlap import keyword_overlap_score

# Optional scorer modules teammates are still filling in. Empty/stub modules are
# skipped gracefully so the app never crashes on unfinished work.
_OPTIONAL_MODULES = (
    "matcher.scoring.tfidf",
    "matcher.scoring.distilbert",
)


def _label_for(score: float) -> str:
    """Map a continuous [0, 1] fit score onto the 3 canonical labels."""
    if score < 0.34:
        return LABELS[0]
    if score < 0.67:
        return LABELS[1]
    return LABELS[2]


class KeywordScorer(Scorer):
    """A real, working scorer available today — no model download required.

    Extracts skills from both texts (``skill_extraction``) and measures how many
    of the job's skills the résumé covers (``keyword_overlap``). Produces genuine
    matched / missing-skill explainability, which is the project's core value.
    """
    name = "keyword"

    def score(self, resume: str, job: str) -> ScoreResult:
        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job)
        overlap = keyword_overlap_score(resume_skills, job_skills)
        s = float(overlap["score"])
        return ScoreResult(
            label=_label_for(s),
            score=round(s, 3),
            matched_skills=overlap["matched_skills"],
            missing_skills=overlap["missing_skills"],
            approach=self.name,
        )


def _discover_optional():
    """Find concrete ``Scorer`` subclasses in the optional modules.

    Returns a list of (display_name, factory) where factory is a zero-arg callable
    that constructs the scorer. Construction is deferred to selection time so heavy
    models are only loaded when a user actually picks them.
    """
    found = []
    for module_name in _OPTIONAL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue  # module missing or import-time error — skip silently
        for _, obj in inspect.getmembers(module, inspect.isclass):
            is_own_scorer = (
                issubclass(obj, Scorer)
                and obj not in (Scorer, DummyScorer)
                and obj.__module__ == module_name
            )
            if is_own_scorer:
                found.append((obj.__name__, obj))
    return found


def build_registry() -> dict:
    """Return ``{display name: factory}`` for every scorer available right now.

    Always includes the keyword scorer (real) and the dummy placeholder. Optional
    real scorers are added only when a teammate has shipped an importable class.
    Each value is a zero-arg callable returning a fresh ``Scorer`` instance.
    """
    registry: dict = {"Keyword skills (real)": KeywordScorer}

    for name, factory in _discover_optional():
        registry[name] = factory

    registry["Dummy (placeholder)"] = DummyScorer
    return registry
