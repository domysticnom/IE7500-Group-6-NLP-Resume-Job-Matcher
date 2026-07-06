# Résumé & Job Matching

An explainable NLP system that scores how well a résumé fits a job posting and
returns top-N recommendations.

**Live demo:** https://huggingface.co/spaces/domynom/resume-job-matcher

Two fit-scoring approaches share one interface and one evaluation harness:
1. **TF-IDF + cosine** (baseline)
2. **Fine-tuned DistilBERT** (3-class sequence-pair classifier)

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash  (macOS/Linux: source .venv/bin/activate)

# 1. PyTorch CUDA wheel first (see https://pytorch.org):
#    e.g. pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"   # must print True

# 2. Everything else:
pip install -r requirements.txt

# 3. The project package (so `import matcher` works anywhere):
pip install -e .
```

## Data

```bash
python scripts/download_data.py        # single reproducible entry point
```
Datasets are gitignored (regenerated on demand). Licenses in `docs/METHODS.md`.

## Run

```bash
streamlit run app/streamlit_app.py     # interactive demo
python -m pytest tests/                # smoke tests
```

## Layout

```
src/matcher/
  data/         load + clean the datasets
  scoring/      base.py = the Scorer contract; tfidf/distilbert implement it
  evaluation/   frozen splits + shared metrics
app/            Streamlit demo
scripts/        download_data.py, train_distilbert.py
docs/           METHODS.md, proposal/, figures/
```

## Team & branches

| Owner | Area | Branches |
|-------|------|----------|
| A | Data pipeline + TF-IDF | `feat/data-pipeline`, `feat/tfidf-scorer` |
| B | Fine-tuned DistilBERT   | `feat/distilbert-scorer` |
| C | Streamlit app + wiring  | `feat/app-shell`, `feat/app-integration` |

Workflow: branch off `main`, open a PR per branch, merge to `main` daily.
No direct commits to `main`.
