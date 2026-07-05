# TF-IDF Fit Baseline Results

TF-IDF + Logistic Regression trained on the (resume, job) pair from `cnamuangtoun/resume-job-description-fit`.

- Train rows: 6241
- Test rows: 1759
- Accuracy: 0.4821
- Macro-F1: 0.4557

```text
               precision    recall  f1-score   support

       No Fit     0.6388    0.5613    0.5975       857
Potential Fit     0.3296    0.4595    0.3838       444
     Good Fit     0.4212    0.3559    0.3858       458

     accuracy                         0.4821      1759
    macro avg     0.4632    0.4589    0.4557      1759
 weighted avg     0.5041    0.4821    0.4884      1759
```
