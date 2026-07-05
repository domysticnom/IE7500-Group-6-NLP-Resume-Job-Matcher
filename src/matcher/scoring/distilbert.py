from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import LABELS, Scorer, ScoreResult
from .keyword_overlap import keyword_overlap_score
from .skill_extraction import extract_skills

# Where the trained checkpoint lives. Point MATCHER_DISTILBERT_DIR at the folder
# you downloaded from the team's Drive if it is somewhere else.
DEFAULT_MODEL_DIR = os.environ.get(
    "MATCHER_DISTILBERT_DIR", "models/trained_transformer_model"
)
MAX_LENGTH = 512


class DistilBertScorer(Scorer):
    """Loads the fine-tuned checkpoint and scores one (resume, job) pair."""

    name = "distilbert"

    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT checkpoint not found at '{model_dir}'. Download the "
                "trained model folder (the trainer.save_model output) and place it "
                "there, or set the MATCHER_DISTILBERT_DIR environment variable."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        # id -> label straight from the model config (authoritative).
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # Only trust the config labels if they are our canonical set; otherwise
        # fall back to positional order (No Fit, Potential Fit, Good Fit).
        self._labels_match = set(self.id2label.values()) == set(LABELS)

    def _rank(self, class_id: int) -> int:
        """Fit rank of a class id: 0 (No Fit) .. len(LABELS)-1 (Good Fit)."""
        if self._labels_match:
            return LABELS.index(self.id2label[class_id])
        return class_id

    @torch.no_grad()
    def score(self, resume: str, job: str) -> ScoreResult:
        inputs = self.tokenizer(
            resume, job, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).tolist()

        pred_id = max(range(len(probs)), key=lambda i: probs[i])
        label = self.id2label[pred_id] if self._labels_match else LABELS[pred_id]

        # Continuous fit score in [0, 1]: probability-weighted rank over the labels.
        denom = len(LABELS) - 1
        fit = sum(self._rank(i) * p for i, p in enumerate(probs)) / denom

        # The classifier does not emit skills; reuse the keyword extractor so the
        # app still shows matched/missing skills (the project's explainability goal).
        overlap = keyword_overlap_score(extract_skills(resume), extract_skills(job))
        return ScoreResult(
            label=label,
            score=round(float(fit), 3),
            matched_skills=overlap["matched_skills"],
            missing_skills=overlap["missing_skills"],
            approach=self.name,
        )
