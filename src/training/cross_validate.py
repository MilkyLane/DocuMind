from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

from src.features.tfidf import build_vectorizer
from src.training.train import load_data

def cross_validate(k=5):
    X, y = load_data()

    pipeline = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=skf,
        scoring="f1_weighted"
    )

    print(f"\n{k}-Fold Cross-Validation (weighted F1):")
    print(f"Scores: {scores}")
    print(f"Mean: {scores.mean():.3f}")
    print(f"Std: {scores.std():.3f}")

if __name__ == "__main__":
    cross_validate()
