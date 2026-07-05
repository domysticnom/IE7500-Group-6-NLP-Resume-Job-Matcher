"""Train the TF-IDF + Logistic Regression fit baseline on the fit dataset.

This trains Approach 1 (TF-IDF baseline) on the SAME labeled data as the
fine-tuned DistilBERT model — `cnamuangtoun/resume-job-description-fit` — so the
two approaches are directly comparable on identical labels
(No Fit / Potential Fit / Good Fit).

Unlike the earlier résumé-only baseline (`scripts/run_ml_baseline.py`, which used
the Kaggle matched_score data), this model is trained on the (resume, job) PAIR,
so its prediction actually depends on the posting.

Output: models/logistic_regression_baseline.pkl  (loaded by matcher.scoring.tfidf)

Run:  python scripts/train_tfidf_fit.py
"""
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

DATASET = "cnamuangtoun/resume-job-description-fit"
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "logistic_regression_baseline.pkl"
RESULTS_FILE = Path("docs") / "tfidf_fit_baseline_results.md"

# Canonical label order — matches matcher.scoring.base.LABELS and the DistilBERT
# model's id2label, so classes 0/1/2 mean the same thing everywhere.
LABELS = ["No Fit", "Potential Fit", "Good Fit"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def _pair_text(example):
    """Join résumé and job into one document for the bag-of-ngrams model."""
    return f"{example['resume_text']} {example['job_description_text']}"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET} ...")
    ds = load_dataset(DATASET)

    X_train = [_pair_text(e) for e in ds["train"]]
    y_train = [LABEL2ID[e["label"]] for e in ds["train"]]
    X_test = [_pair_text(e) for e in ds["test"]]
    y_test = [LABEL2ID[e["label"]] for e in ds["test"]]
    print(f"Train: {len(X_train)}   Test: {len(X_test)}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression on the (resume, job) pair ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(
        y_test, y_pred, target_names=LABELS, digits=4, zero_division=0
    )
    print(f"\nAccuracy: {accuracy:.4f}   Macro-F1: {macro_f1:.4f}\n{report}")

    print(f"Saving model to {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)

    RESULTS_FILE.write_text(
        "# TF-IDF Fit Baseline Results\n\n"
        f"TF-IDF + Logistic Regression trained on the (resume, job) pair from "
        f"`{DATASET}`.\n\n"
        f"- Train rows: {len(X_train)}\n- Test rows: {len(X_test)}\n"
        f"- Accuracy: {accuracy:.4f}\n- Macro-F1: {macro_f1:.4f}\n\n"
        "```text\n" + report + "```\n",
        encoding="utf-8",
    )
    print(f"Saved results to {RESULTS_FILE}\nDone.")


if __name__ == "__main__":
    main()
