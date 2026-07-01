"""The Scorer contract, the single integration seam for the whole project.

Every approach (TF-IDF, fine-tuned DistilBERT) subclasses `Scorer` and implements
`score()`. The frontend imports this interface and builds against `DummyScorer`
until real scorers land, then swaps them in with zero rewrites.

DO NOT change the signatures of `ScoreResult` or `Scorer.score` without telling
the whole team, this is the contract everyone depends on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Canonical label order, reused by evaluation.metrics. Keep in sync.
LABELS = ["No Fit", "Potential Fit", "Good Fit"]


@dataclass
class ScoreResult:
    """What every scorer returns for one (resume, job) pair."""
    label: str                              # one of LABELS
    score: float                            # continuous fit score in [0.0, 1.0]
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    approach: str = ""                      # which scorer produced this


class Scorer(ABC):
    """Base class for every fit-scoring approach."""
    name: str = "base"

    @abstractmethod
    def score(self, resume: str, job: str) -> ScoreResult:
        """Return a ScoreResult for one résumé / job-description pair."""
        raise NotImplementedError


class DummyScorer(Scorer):
    """Deterministic placeholder so the frontend is never blocked on a real model."""
    name = "dummy"

    def score(self, resume: str, job: str) -> ScoreResult:
        resume_words = set(resume.lower().split())
        job_words = set(job.lower().split())
        overlap = len(resume_words & job_words)
        s = min(1.0, overlap / 40.0)
        label = LABELS[0] if s < 0.34 else LABELS[1] if s < 0.67 else LABELS[2]
        return ScoreResult(
            label=label,
            score=round(s, 3),
            matched_skills=sorted(resume_words & job_words)[:8],
            missing_skills=sorted(job_words - resume_words)[:8],
            approach=self.name,
        )
