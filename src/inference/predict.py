import joblib
import numpy as np
from pathlib import Path
import time

from src.ocr.ocr_utils import extract_text
from src.extraction.invoice import extract_invoice_fields
from src.extraction.resume import extract_resume_fields
from src.extraction.form import extract_form_fields
from src.extraction.bank_statement import extract_bank_statement_fields
from src.extraction.utility import extract_utility_fields
from src.utils.logging import get_logger

logger = get_logger()


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
    start_total = time.perf_counter()

    path = Path(file_path)

    # --- OCR ---
    start_ocr = time.perf_counter()
    text = extract_text(path)
    ocr_time = time.perf_counter() - start_ocr

    if len(text.strip()) < 100:
        logger.info(
            f"OCR_FAILED file={path.name} ocr_time={ocr_time:.3f}s"
        )
        return {
            "status": "ocr_failed",
            "document_type": "unsupported",
            "reason": "OCR produced insufficient text",
            "latency": {"ocr_seconds": round(ocr_time, 3)},
        }

    # --- Classification ---
    start_cls = time.perf_counter()
    doc_type, confidence = classify(text)
    cls_time = time.perf_counter() - start_cls

    if confidence < CONFIDENCE_THRESHOLD:
        total_time = time.perf_counter() - start_total
        logger.info(
            f"REJECTED file={path.name} "
            f"conf={confidence:.2f} "
            f"ocr={ocr_time:.3f}s cls={cls_time:.3f}s total={total_time:.3f}s"
        )
        return {
            "status": "rejected",
            "document_type": "unsupported",
            "confidence": confidence,
            "reason": "Low classification confidence",
            "latency": {
                "ocr_seconds": round(ocr_time, 3),
                "classification_seconds": round(cls_time, 3),
                "total_seconds": round(total_time, 3),
            },
        }

    # --- Extraction ---
    start_ext = time.perf_counter()
    extractor = EXTRACTORS[doc_type]
    fields = extractor(text)
    ext_time = time.perf_counter() - start_ext

    total_time = time.perf_counter() - start_total

    logger.info(
        f"SUCCESS file={path.name} "
        f"type={doc_type} conf={confidence:.2f} "
        f"ocr={ocr_time:.3f}s cls={cls_time:.3f}s "
        f"ext={ext_time:.3f}s total={total_time:.3f}s"
    )

    return {
        "status": "success",
        "document_type": doc_type,
        "confidence": confidence,
        "latency": {
            "ocr_seconds": round(ocr_time, 3),
            "classification_seconds": round(cls_time, 3),
            "extraction_seconds": round(ext_time, 3),
            "total_seconds": round(total_time, 3),
        },
        "fields": fields
    }



if __name__ == "__main__":
    import sys
    result = predict(sys.argv[1])
    print(result)
