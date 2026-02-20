"""
Model explainability using TF-IDF feature weights.

For a LogisticRegression inside a sklearn Pipeline, the contribution of each
token to a given class prediction is:

    contribution_i = tfidf_score_i * coef_class_i

We take the top-N by absolute contribution so the caller sees which words
most strongly pushed the classifier toward (positive weight) or away from
(negative weight) the predicted class.
"""

import numpy as np
from pathlib import Path
import joblib
from typing import Any

MODEL_PATH = "models/documind_classifier.joblib"

# Loaded once at import time — same as predict.py
_pipeline = joblib.load(MODEL_PATH)


def explain(text: str, top_n: int = 15) -> dict[str, Any]:
    """
    Return the top-N TF-IDF features that drove the classification.

    Returns
    -------
    {
        "predicted_class": str,
        "confidence": float,
        "top_features": [
            {"term": str, "weight": float, "direction": "for"|"against"},
            ...
        ],
        "all_classes": [str, ...]
    }
    """
    vectoriser = _pipeline.named_steps["tfidf"]
    classifier = _pipeline.named_steps["clf"]

    # Get TF-IDF representation
    tfidf_matrix = vectoriser.transform([text])           # (1, n_features)
    feature_names: list[str] = vectoriser.get_feature_names_out().tolist()

    # Predict
    probs = _pipeline.predict_proba([text])[0]
    raw_classes: list[str] = [str(c) for c in _pipeline.classes_]
    norm_classes: list[str] = [c.lower().replace(" ", "_") for c in raw_classes]
    pred_idx = int(np.argmax(probs))
    predicted_class = norm_classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Resolve the underlying LR estimator (may be wrapped in CalibratedClassifierCV)
    lr = _get_lr(classifier)

    if lr is not None and hasattr(lr, "coef_"):
        # coef_ shape: (n_classes, n_features) for multiclass
        # Map predicted class to its coefficient row using raw (un-normalised) label
        lr_raw_classes = [str(c) for c in lr.classes_] if hasattr(lr, "classes_") else raw_classes
        raw_predicted = raw_classes[pred_idx]
        if raw_predicted in lr_raw_classes:
            class_row = lr_raw_classes.index(raw_predicted)
        else:
            class_row = pred_idx

        coef = np.asarray(lr.coef_[class_row])               # (n_features,)
        tfidf_scores = np.asarray(tfidf_matrix.todense())[0]  # (n_features,)
        contributions = tfidf_scores * coef                    # element-wise

        # Top-N by absolute value, among terms that actually appeared in the doc
        present = np.where(tfidf_scores > 0)[0]
        if len(present) == 0:
            top_features = []
        else:
            present_contribs = contributions[present]
            sorted_idx = np.argsort(np.abs(present_contribs))[::-1][:top_n]
            top_features = [
                {
                    "term": feature_names[present[i]],
                    "weight": round(float(contributions[present[i]]), 4),
                    "direction": "for" if contributions[present[i]] >= 0 else "against",
                }
                for i in sorted_idx
            ]
    else:
        # Fallback: just return highest TF-IDF terms (no coef available)
        tfidf_scores = np.asarray(tfidf_matrix.todense())[0]
        sorted_idx = np.argsort(tfidf_scores)[::-1][:top_n]
        top_features = [
            {
                "term": feature_names[i],
                "weight": round(float(tfidf_scores[i]), 4),
                "direction": "for",
            }
            for i in sorted_idx
            if tfidf_scores[i] > 0
        ]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "top_features": top_features,
        "all_classes": norm_classes,
    }


def _get_lr(classifier):
    """
    Unwrap CalibratedClassifierCV → cross_val_estimators → base LR estimator.
    Works with sklearn >= 1.0.
    """
    # CalibratedClassifierCV stores a list of calibrated classifiers
    if hasattr(classifier, "calibrated_classifiers_"):
        for cc in classifier.calibrated_classifiers_:
            base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if base is not None and hasattr(base, "coef_"):
                return base
    # Direct LR
    if hasattr(classifier, "coef_"):
        return classifier
    return None
