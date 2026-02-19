import joblib
import numpy as np
from pathlib import Path

from src.ocr.ocr_utils import extract_text
from src.extraction.invoice import extract_invoice_fields
from src.extraction.resume import extract_resume_fields
from src.extraction.form import extract_form_fields
from src.extraction.bank_statement import extract_bank_statement_fields
from src.extraction.utility import extract_utility_fields

MODEL_PATH = "models/documind_classifier.joblib"
CONFIDENCE_THRESHOLD = 0.40

pipeline = joblib.load(MODEL_PATH)

EXTRACTORS = {
    "invoice": extract_invoice_fields,
    "resume": extract_resume_fields,
    "form": extract_form_fields,
    "bank_statement": extract_bank_statement_fields,
    "utility": extract_utility_fields,
}

def classify(text: str):
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    idx = np.argmax(probs)
    return classes[idx], float(probs[idx])

def predict(file_path: str):
    path = Path(file_path)

    text = extract_text(path)
    if len(text.strip()) < 100:
        return {
            "document_type": "unsupported",
            "reason": "OCR produced insufficient text"
        }

    doc_type, confidence = classify(text)

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "document_type": "unsupported",
            "confidence": confidence,
            "reason": "Low classification confidence"
        }

    extractor = EXTRACTORS[doc_type]
    fields = extractor(text)

    return {
        "document_type": doc_type,
        "confidence": confidence,
        "fields": fields
    }


if __name__ == "__main__":
    import sys
    result = predict(sys.argv[1])
    print(result)
