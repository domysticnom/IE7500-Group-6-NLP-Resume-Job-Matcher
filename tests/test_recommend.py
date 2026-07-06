"""Smoke tests for the job recommender.

Uses a tiny in-memory corpus so the tests are fast and need no network or model.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from matcher.recommend import JobRecommender, Posting


def _corpus():
    return [
        Posting("j1", "python sql machine learning data analysis pandas python sql", "Data Scientist"),
        Posting("j2", "java spring backend microservices kubernetes java spring", "Backend Engineer"),
        Posting("j3", "sql excel tableau dashboards reporting sql excel", "BI Analyst"),
        Posting("j4", "python pytorch deep learning nlp python machine learning", "ML Engineer"),
    ]


def test_recommendations_are_sorted_and_limited():
    rec = JobRecommender(_corpus(), min_df=1)
    results = rec.recommend("python machine learning data pandas", top_n=3)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_relevant_job_ranks_first():
    rec = JobRecommender(_corpus(), min_df=1)
    results = rec.recommend("python machine learning pandas data scientist", top_n=4)
    # a python/ML posting should out-rank the java one
    assert results[0].posting.id in {"j1", "j4"}
    assert results[0].rank == 1


def test_recommendation_carries_skills():
    rec = JobRecommender(_corpus(), min_df=1)
    top = rec.recommend("python and sql experience", top_n=1)[0]
    assert isinstance(top.matched_skills, list)
    assert isinstance(top.missing_skills, list)
