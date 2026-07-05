from __future__ import annotations

import os
from pathlib import Path

import joblib

from .base import LABELS, Scorer, ScoreResult
from .keyword_overlap import keyword_overlap_score
from .skill_extraction import extract_skills

DEFAULT_MODEL_PATH = os.environ.get(
    "MATCHER_TFIDF_MODEL", "models/logistic_regression_baseline.pkl"
)

# The pkl (trained by scripts/train_tfidf_fit.py on the resume+job pair) uses
# integer labels 0/1/2 in the canonical order: No Fit, Potential Fit, Good Fit.
_ID_TO_LABEL = {0: LABELS[0], 1: LABELS[1], 2: LABELS[2]}


class TfidfScorer(Scorer):
    """Loads the joblib pipeline and classifies a résumé's fit bucket."""

    name = "tfidf"

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"TF-IDF model not found at '{model_path}'. Run "
                "`python scripts/run_ml_baseline.py` to create it, or set the "
                "MATCHER_TFIDF_MODEL environment variable."
            )
        self.model = joblib.load(model_path)

    def score(self, resume: str, job: str) -> ScoreResult:
        # Trained on the (resume, job) pair joined into one document.
        text = f"{resume} {job}"
        pred_id = int(self.model.predict([text])[0])
        label = _ID_TO_LABEL.get(pred_id, LABELS[pred_id])

        # Continuous fit score in [0, 1]: probability-weighted rank over labels.
        denom = len(LABELS) - 1
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([text])[0]
            classes = list(self.model.classes_)
            fit = sum(
                LABELS.index(_ID_TO_LABEL[int(c)]) * p for c, p in zip(classes, probs)
            ) / denom
        else:
            fit = pred_id / denom

        # Baseline emits no skills; reuse the keyword extractor for explainability.
        overlap = keyword_overlap_score(extract_skills(resume), extract_skills(job))
        return ScoreResult(
            label=label,
            score=round(float(fit), 3),
            matched_skills=overlap["matched_skills"],
            missing_skills=overlap["missing_skills"],
            approach=self.name,
        )
