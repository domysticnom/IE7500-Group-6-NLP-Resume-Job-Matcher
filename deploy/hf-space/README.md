---
title: Resume Job Fit Matcher
emoji: 📄
colorFrom: gray
colorTo: indigo
sdk: streamlit
sdk_version: 1.57.0
app_file: app/streamlit_app.py
pinned: false
---

# Resume - Job Fit Matcher

Explainable resume-to-job fit scoring with ranked job recommendations.

- Recommend jobs: paste a resume, get the best-matching postings ranked by fit,
  with matched/missing skills. TF-IDF retrieves candidates; DistilBERT reranks.
- Compare one pair: score a single resume against a single job across approaches.

The fine-tuned DistilBERT model is loaded from a Hugging Face model repo set via
the MATCHER_DISTILBERT_DIR environment variable (a Space secret/variable).
