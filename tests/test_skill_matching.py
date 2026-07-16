import os

import pytest

from matcher.scoring.skill_extraction import extract_skills
from matcher.scoring.keyword_overlap import keyword_overlap_score


def test_extract_skills_fallback(monkeypatch):
    # With NER disabled, extraction uses the keyword-dictionary fallback and is
    # fully deterministic (no model download).
    monkeypatch.setenv("MATCHER_DISABLE_NER", "1")
    text = "I have experience with Python, SQL, Excel, and Power BI."
    skills = extract_skills(text)

    assert "python" in skills
    assert "sql" in skills
    assert "excel" in skills
    assert "power bi" in skills


def test_extract_skills_explicit_list():
    # An explicit vocabulary always uses dictionary matching (NER bypassed, no
    # download), regardless of the NER setting.
    text = "We need Kafka and Kubernetes experience, plus some Python."
    skills = extract_skills(text, skill_list=["kafka", "kubernetes", "python", "go"])

    assert skills == ["kafka", "kubernetes", "python"]


def test_keyword_overlap_score():
    resume_skills = ["python", "sql", "excel"]
    job_skills = ["python", "sql", "tableau"]

    result = keyword_overlap_score(resume_skills, job_skills)

    assert result["score"] == 0.667
    assert result["matched_skills"] == ["python", "sql"]
    assert result["missing_skills"] == ["tableau"]


@pytest.mark.skipif(
    os.environ.get("MATCHER_RUN_NER_TESTS") != "1",
    reason="NER integration test downloads a model; set MATCHER_RUN_NER_TESTS=1 to run",
)
def test_extract_skills_ner_surfaces_unlisted_skills(monkeypatch):
    # Kubernetes / React are NOT in SKILL_KEYWORDS, so only the NER path can
    # surface them. This is the whole point of the swap.
    monkeypatch.delenv("MATCHER_DISABLE_NER", raising=False)
    skills = extract_skills(
        "Backend engineer experienced with Kubernetes and React."
    )
    joined = " ".join(skills)
    assert "kubernetes" in joined or "react" in joined
