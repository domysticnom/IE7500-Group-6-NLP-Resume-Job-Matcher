"""Résumé → ranked job recommendations (requirement MATCH-01).

Retrieve-then-rerank recommender:
  1. TF-IDF cosine instantly ranks the whole job corpus against the résumé.
  2. (optional) a Scorer (e.g. DistilBertScorer) reranks only the top-K
     candidates for accuracy — keeping the demo responsive even with a heavy
     model instead of scoring every posting.

The recommender is corpus-agnostic: hand it any list of Postings and it ranks
them, so swapping the sample corpus for a 100k-row dataset is just a loader
change, not an engine change.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .scoring.keyword_overlap import keyword_overlap_score
from .scoring.skill_extraction import extract_skills

FIT_DATASET = "cnamuangtoun/resume-job-description-fit"
CORPUS_CSV = Path("data/processed/job_postings.csv")


def _title(text: str, width: int = 80) -> str:
    """A short human-readable label for a posting (its first line / clause)."""
    first = text.strip().split("\n", 1)[0].strip()
    return first[:width] + ("…" if len(first) > width else "")


@dataclass
class Posting:
    id: str
    text: str
    title: str


@dataclass
class Recommendation:
    rank: int
    posting: Posting
    score: float                       # final fit score in [0, 1]
    retrieve_score: float              # TF-IDF cosine similarity
    label: str | None = None           # set when a reranker classifies the pair
    matched_skills: list = field(default_factory=list)
    missing_skills: list = field(default_factory=list)


def load_job_corpus(refresh: bool = False) -> list[Posting]:
    """Load unique job postings, caching to data/processed for offline reuse."""
    if CORPUS_CSV.exists() and not refresh:
        with open(CORPUS_CSV, encoding="utf-8") as f:
            return [Posting(**row) for row in csv.DictReader(f)]

    from datasets import load_dataset

    ds = load_dataset(FIT_DATASET)
    seen: dict[str, Posting] = {}
    for split in ds:
        for text in ds[split]["job_description_text"]:
            if text and text not in seen:
                seen[text] = Posting(id=f"job-{len(seen):05d}", text=text, title=_title(text))
    postings = list(seen.values())

    CORPUS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "title"])
        writer.writeheader()
        for p in postings:
            writer.writerow({"id": p.id, "text": p.text, "title": p.title})
    return postings


def load_sample_resumes(n: int = 8) -> list[str]:
    """A handful of distinct résumés for the demo's 'load a sample' control."""
    from datasets import load_dataset

    ds = load_dataset(FIT_DATASET)
    seen: list[str] = []
    for text in ds["test"]["resume_text"]:
        if text and text not in seen:
            seen.append(text)
        if len(seen) >= n:
            break
    return seen


class JobRecommender:
    """Ranks a job corpus against a résumé (TF-IDF retrieve, optional rerank)."""

    def __init__(self, postings: list[Posting], min_df: int = 2):
        if not postings:
            raise ValueError("empty job corpus")
        self.postings = postings
        self._vectorizer = TfidfVectorizer(
            stop_words="english", max_features=20000, ngram_range=(1, 2), min_df=min_df
        )
        self._matrix = self._vectorizer.fit_transform([p.text for p in postings])

    def recommend(self, resume: str, top_n: int = 10, reranker=None, pool_k: int = 20):
        """Return the top-N postings for a résumé, most-fitting first.

        Without a reranker, ordering is TF-IDF cosine. With one, the top
        ``max(pool_k, top_n)`` TF-IDF candidates are re-scored by the reranker
        (a Scorer) and re-sorted by that score.
        """
        query = self._vectorizer.transform([resume])
        sims = cosine_similarity(query, self._matrix)[0]
        order = sims.argsort()[::-1]

        pool_size = max(pool_k, top_n) if reranker is not None else top_n
        candidates = []
        for idx in order[:pool_size]:
            idx = int(idx)
            posting = self.postings[idx]
            retrieve = float(sims[idx])
            score, label = retrieve, None
            if reranker is not None:
                result = reranker.score(resume, posting.text)
                score, label = float(result.score), result.label
            candidates.append((posting, retrieve, score, label))

        candidates.sort(key=lambda c: c[2], reverse=True)

        recs = []
        for rank, (posting, retrieve, score, label) in enumerate(candidates[:top_n], 1):
            overlap = keyword_overlap_score(extract_skills(resume), extract_skills(posting.text))
            recs.append(
                Recommendation(
                    rank=rank,
                    posting=posting,
                    score=round(score, 3),
                    retrieve_score=round(retrieve, 3),
                    label=label,
                    matched_skills=overlap["matched_skills"],
                    missing_skills=overlap["missing_skills"],
                )
            )
        return recs
