# Résumé & Job Matching

An explainable NLP system that scores how well a résumé fits a job posting,
surfaces the matched vs missing skills behind the score, and recommends the
best-matching job postings for a résumé.

**Live demo:** https://huggingface.co/spaces/domynom/resume-job-matcher

The Streamlit demo has two views:

- **Recommend jobs** — paste or pick a résumé and get a ranked list of matching
  job postings (TF-IDF retrieval, optional DistilBERT rerank) with fit scores and
  matched/missing skills.
- **Compare one pair** — score a single résumé against a single job across every
  available approach.

Every approach implements one shared `Scorer` interface
(`src/matcher/scoring/base.py`), so the app renders them interchangeably. Fit
labels are No Fit / Potential Fit / Good Fit.

## Scoring approaches

| Approach | What it is | Needs |
|----------|-----------|-------|
| **Keyword skills** | Skill overlap between résumé and job (extract + intersect) | pretrained NER models auto-download on first use; falls back to a keyword list offline |
| **TF-IDF + Logistic Regression** | scikit-learn pipeline over the résumé+job pair | `models/logistic_regression_baseline.pkl` (regenerate below) |
| **Fine-tuned DistilBERT** | 3-class sequence-pair classifier | the trained checkpoint (below) + `torch` |
| **Dummy** | deterministic placeholder | nothing (dev fallback) |

Whichever models are present light up automatically in the app's dropdown; the
Keyword and Dummy approaches always work, so the demo is never blocked on a model.

### Skill extraction (matched / missing skills)

The matched/missing-skill explainability is driven by two pretrained NER models
from the jjzha / SkillSpan line — `jjzha/jobbert_knowledge_extraction` (hard
skills/tools) and `jjzha/jobbert_skill_extraction` (applied/soft skills) — whose
spans are unioned. They download from Hugging Face on first use and are cached.
If they can't be loaded (offline, or `MATCHER_DISABLE_NER=1`), extraction falls
back to a fixed keyword list so the app still runs. Tunable via
`MATCHER_KNOWLEDGE_NER_MODEL`, `MATCHER_SKILL_NER_MODEL`, and
`MATCHER_SKILL_NER_MIN_SCORE`.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash  (macOS/Linux: source .venv/bin/activate)

# 1. PyTorch. The CPU wheel is enough to RUN the demo (single-pair inference is fast):
pip install torch --index-url https://download.pytorch.org/whl/cpu
#    For GPU training, install the CUDA wheel from https://pytorch.org instead
#    and verify: python -c "import torch; print(torch.cuda.is_available())"

# 2. Everything else:
pip install -r requirements.txt

# 3. The project package (so `import matcher` works anywhere):
pip install -e .
```

## Models

Model weights are git-ignored — each teammate regenerates or supplies them locally.

```bash
# TF-IDF + Logistic Regression: downloads the public fit dataset from Hugging Face
# and trains in seconds -> writes models/logistic_regression_baseline.pkl
python scripts/train_tfidf_fit.py
```

```
# Fine-tuned DistilBERT: place the trained checkpoint (config.json,
# model.safetensors, tokenizer files) at:
models/trained_transformer_model/
# or point MATCHER_DISTILBERT_DIR at wherever it lives.
```

## Run

```bash
streamlit run app/streamlit_app.py     # interactive demo at http://localhost:8501
python -m pytest tests/                # smoke tests
```

## Layout

```
src/matcher/
  scoring/      base.py = the Scorer contract; keyword_overlap, skill_extraction,
                tfidf, distilbert implement/support it
  data/         dataset loading + cleaning helpers
  evaluation/   shared metrics + frozen splits
app/            streamlit_app.py (demo) + scorers.py (Scorer registry)
scripts/        train_tfidf_fit.py, run_*_baseline.py, prepare_data.py
docs/           METHODS.md, baseline results, proposal/, figures/
tests/          smoke tests
```

## Team & branches

| Owner | Area | Branches |
|-------|------|----------|
| A | Data pipeline + TF-IDF | `feat/data-pipeline`, `feat/tfidf-scorer` |
| B | Fine-tuned DistilBERT   | `feat/distilbert-scorer` |
| C | Streamlit app + wiring  | `feat/app-shell`, `feat/app-integration` |

Workflow: branch off `main`, open a PR per branch, merge to `main` daily.
No direct commits to `main`.
