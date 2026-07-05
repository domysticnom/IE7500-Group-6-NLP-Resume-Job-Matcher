"""Résumé ↔ Job fit demo.

Paste a résumé and a job posting, pick an approach, and get an explainable fit
score with the matched vs missing skills behind it. The app is built entirely
against the ``Scorer`` interface, so every approach the team ships appears in the
selector automatically — no changes here.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from scorers import build_registry

st.set_page_config(page_title="Résumé ↔ Job Fit", page_icon="🧭", layout="wide")

REGISTRY = build_registry()

# Small samples so the demo shows real matched/missing skills out of the box.
SAMPLE_RESUME = (
    "Data analyst with 3 years of experience. Strong in Python, SQL, pandas, and "
    "data visualization with Tableau. Built machine learning models with "
    "scikit-learn and communicated results to stakeholders."
)
SAMPLE_JOB = (
    "Seeking a data scientist skilled in Python, SQL, machine learning, and deep "
    "learning with PyTorch. Experience with AWS and dashboard reporting preferred. "
    "Strong communication and problem solving required."
)


def _skill_list(skills: list[str]) -> None:
    """Render a bullet list of skills, or a muted 'none'."""
    if skills:
        st.markdown("\n".join(f"- {s}" for s in skills))
    else:
        st.caption("none")


st.title("Résumé ↔ Job Fit")
st.caption(
    "Paste a résumé and a job posting to get an explainable fit score and the "
    "skills behind it."
)

with st.sidebar:
    st.header("Approach")
    choice = st.selectbox("Scoring approach", list(REGISTRY.keys()))
    st.markdown(
        "- **Keyword skills** — real skill overlap (available now)\n"
        "- **TF-IDF / DistilBERT** — appear here once the team finishes them\n"
        "- **Dummy** — deterministic placeholder"
    )
    if st.button("Load sample text"):
        st.session_state["resume"] = SAMPLE_RESUME
        st.session_state["job"] = SAMPLE_JOB
        st.rerun()

col_resume, col_job = st.columns(2)
with col_resume:
    st.subheader("Résumé")
    resume = st.text_area(
        "Résumé text", height=320, key="resume", placeholder="Paste a résumé…"
    )
with col_job:
    st.subheader("Job posting")
    job = st.text_area(
        "Job description", height=320, key="job", placeholder="Paste a job posting…"
    )

if st.button("Score fit", type="primary"):
    if not resume.strip() or not job.strip():
        st.warning("Paste both a résumé and a job posting first.")
    else:
        try:
            with st.spinner(f"Scoring with {choice}…"):
                scorer = REGISTRY[choice]()
                result = scorer.score(resume, job)
        except Exception as exc:  # missing weights, load error, etc.
            st.error(f"Could not run '{choice}': {exc}")
            st.stop()

        st.divider()
        col_score, col_skills = st.columns([1, 2])
        with col_score:
            st.metric("Fit score", f"{result.score * 100:.0f}%")
            st.metric("Verdict", result.label)
            st.caption(f"approach: {result.approach}")
        with col_skills:
            matched_col, missing_col = st.columns(2)
            with matched_col:
                st.markdown("**✅ Matched skills**")
                _skill_list(result.matched_skills)
            with missing_col:
                st.markdown("**❌ Missing skills**")
                _skill_list(result.missing_skills)
