"""Résumé ↔ Job fit demo.

Two views:
  • Recommend jobs — paste/pick a résumé, get a ranked list of matching job
    postings (TF-IDF retrieve, optional DistilBERT rerank) with fit scores and
    matched/missing skills.
  • Compare one pair — score a single résumé against a single job across every
    available approach.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from scorers import build_registry
from matcher.recommend import JobRecommender, load_job_corpus, load_sample_resumes

st.set_page_config(page_title="Résumé ↔ Job Fit", layout="wide")


# --- cached heavy resources (built once per session) ---------------------------
@st.cache_resource(show_spinner="Loading job postings…")
def get_recommender() -> JobRecommender:
    return JobRecommender(load_job_corpus())


@st.cache_resource(show_spinner=False)
def get_reranker():
    """The DistilBERT scorer, if its checkpoint is available; else None."""
    factory = build_registry().get("DistilBertScorer")
    if factory is None:
        return None
    try:
        return factory()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_sample_resumes():
    return load_sample_resumes()


VERDICT_COLOR = {"Good Fit": "green", "Potential Fit": "orange", "No Fit": "red"}


def _clean_title(text: str, width: int = 60) -> str:
    """A tidy one-line heading from a posting's text (collapsed, first clause)."""
    t = " ".join(text.split()).split(". ")[0]
    return t if len(t) <= width else t[:width].rsplit(" ", 1)[0] + "…"


def _preview(text: str, n: int = 240) -> str:
    """A short, single-paragraph preview of the posting."""
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def _badges(skills: list[str], color: str, limit: int = 12) -> str:
    """Render skills as colored inline badges, capping the number shown."""
    if not skills:
        return "_none_"
    shown = skills[:limit]
    md = " ".join(f":{color}-background[{s}]" for s in shown)
    extra = len(skills) - len(shown)
    return md + (f" _+{extra} more_" if extra else "")


# --- header --------------------------------------------------------------------
st.title("Résumé ↔ Job Fit")
recommend_tab, compare_tab = st.tabs(["Recommend jobs", "Compare one pair"])


# =============================== RECOMMEND JOBS ================================
with recommend_tab:
    st.caption("Paste or pick a résumé and get the best-matching job postings, ranked by fit.")

    samples = get_sample_resumes()
    options = ["— paste your own —"] + [f"Sample {i + 1}: {s[:50]}…" for i, s in enumerate(samples)]
    if "resume_text" not in st.session_state:
        st.session_state["resume_text"] = ""

    pick = st.selectbox("Load a sample résumé", options, key="sample_pick")
    if st.button("Load sample", disabled=(pick == options[0])):
        st.session_state["resume_text"] = samples[options.index(pick) - 1]
        st.rerun()

    resume = st.text_area(
        "Résumé text", key="resume_text", height=240, placeholder="Paste a résumé…"
    )

    settings_col, _ = st.columns([2, 2])
    with settings_col:
        top_n = st.slider("How many jobs to show", 3, 25, 10)
        use_rerank = st.checkbox(
            "Rerank with DistilBERT", value=False,
            help="TF-IDF finds candidates instantly; DistilBERT re-scores the top ones "
                 "for accuracy (slower, needs the checkpoint).",
        )

    if st.button("Recommend jobs", type="primary"):
        if not resume.strip():
            st.warning("Paste or load a résumé first.")
        else:
            reranker = None
            if use_rerank:
                reranker = get_reranker()
                if reranker is None:
                    st.warning("DistilBERT checkpoint not found — showing TF-IDF ranking only.")

            with st.spinner("Ranking postings…"):
                recs = get_recommender().recommend(
                    resume, top_n=top_n, reranker=reranker, pool_k=max(20, top_n)
                )

            mode = "TF-IDF retrieve + DistilBERT rerank" if reranker else "TF-IDF ranking"
            st.caption(f"Top {len(recs)} matches - {mode}")
            for rec in recs:
                with st.container(border=True):
                    head, meta = st.columns([7, 2])
                    with head:
                        st.markdown(f"##### {rec.rank}. {_clean_title(rec.posting.text)}")
                        st.caption(_preview(rec.posting.text))
                    with meta:
                        if rec.label:
                            st.markdown(
                                f":{VERDICT_COLOR.get(rec.label, 'gray')}-background[{rec.label}]"
                            )
                        st.progress(
                            min(1.0, max(0.0, rec.score)),
                            text=f"match {rec.score * 100:.0f}%",
                        )
                    st.markdown("**Matched skills** &nbsp; " + _badges(rec.matched_skills, "green"))
                    st.markdown("**Missing skills** &nbsp; " + _badges(rec.missing_skills, "red"))
                    with st.expander("View full posting"):
                        st.write(rec.posting.text)


# ============================== COMPARE ONE PAIR ==============================
with compare_tab:
    st.caption("Score one résumé against one job posting across every available approach.")
    registry = build_registry()

    choice = st.selectbox("Scoring approach", list(registry.keys()), key="compare_approach")
    col_r, col_j = st.columns(2)
    resume_one = col_r.text_area("Résumé", height=240, key="cmp_resume", placeholder="Paste a résumé…")
    job_one = col_j.text_area("Job posting", height=240, key="cmp_job", placeholder="Paste a job posting…")

    if st.button("Score fit", type="primary", key="cmp_score"):
        if not resume_one.strip() or not job_one.strip():
            st.warning("Paste both a résumé and a job posting first.")
        else:
            try:
                with st.spinner(f"Scoring with {choice}…"):
                    result = registry[choice]().score(resume_one, job_one)
            except Exception as exc:
                st.error(f"Could not run '{choice}': {exc}")
                st.stop()

            score_col, skills_col = st.columns([1, 2])
            with score_col:
                st.metric("Fit score", f"{result.score * 100:.0f}%")
                st.metric("Verdict", result.label)
                st.caption(f"approach: {result.approach}")
            with skills_col:
                matched_col, missing_col = st.columns(2)
                matched_col.markdown("**Matched skills**")
                matched_col.markdown(", ".join(result.matched_skills) or "—")
                missing_col.markdown("**Missing skills**")
                missing_col.markdown(", ".join(result.missing_skills) or "—")
