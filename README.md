# Résumé & Job Matching

An explainable NLP system that scores how well a résumé fits a job posting and
surfaces the matched vs missing skills behind the score. (Top-N job
recommendations are planned — see `.planning` / ROADMAP.)

The Streamlit demo lets you paste a résumé and a job posting, pick a scoring
approach, and see a fit score (No Fit / Potential Fit / Good Fit) plus the
matched and missing skills. Every approach implements one shared `Scorer`
interface (`src/matcher/scoring/base.py`), so the app renders them
interchangeably.

## Scoring approaches

| Approach | What it is | Needs |
|----------|-----------|-------|
| **Keyword skills** | Skill overlap between résumé and job (extract + intersect) | nothing — works out of the box |
| **TF-IDF + Logistic Regression** | scikit-learn pipeline over the résumé+job pair | `models/logistic_regression_baseline.pkl` (regenerate below) |
| **Fine-tuned DistilBERT** | 3-class sequence-pair classifier | the trained checkpoint (below) + `torch` |
| **Dummy** | deterministic placeholder | nothing (dev fallback) |

Whichever models are present light up automatically in the app's dropdown; the
Keyword and Dummy approaches always work, so the demo is never blocked on a model.

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
