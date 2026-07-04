import joblib
import pandas as pd
from pathlib import Path


MODEL_FILE = "models/logistic_regression_baseline.pkl"
OUTPUT_FILE = Path("docs/model_feature_weights.csv")

model = joblib.load(MODEL_FILE)

tfidf = model.named_steps["tfidf"]
clf = model.named_steps["clf"]

feature_names = tfidf.get_feature_names_out()
class_names = ["low_match", "medium_match", "high_match"]

all_weights = []

for class_index, class_name in enumerate(class_names):
    weights = clf.coef_[class_index]

    for feature, weight in zip(feature_names, weights):
        all_weights.append(
            {
                "class": class_name,
                "feature": feature,
                "weight": weight,
            }
        )

weights_df = pd.DataFrame(all_weights)

weights_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved model feature weights to: {OUTPUT_FILE}")

for class_name in class_names:
    print()
    print("=" * 80)
    print(f"Top 20 positive features for {class_name}")
    print("=" * 80)

    print(
        weights_df[weights_df["class"] == class_name]
        .sort_values("weight", ascending=False)
        .head(20)
    )