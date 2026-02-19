from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

from src.features.tfidf import build_vectorizer

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    texts = []
    labels = []

    for class_dir in PROCESSED_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name

        for txt_file in class_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8").strip()
            if len(text) < 50:  # skips garbage OCR
                continue

            texts.append(text)
            labels.append(label)

    return texts, labels

def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", build_vectorizer()),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(
                max_iter=1_000,
                class_weight="balanced",
            ),
            method="isotonic",
            cv=5
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipeline, MODEL_DIR / "documind_classifier.joblib")
    print("\nModel saved to models/documind_classifier.joblib")

if __name__ == "__main__":
    train()
