from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentLog, FeedbackLog


async def insert_document_log(
    db: AsyncSession,
    *,
    id: str | None = None,
    filename: str,
    file_extension: str,
    document_type: str,
    confidence: float | None,
    status: str,
    ocr_seconds: float | None = None,
    classification_seconds: float | None = None,
    extraction_seconds: float | None = None,
    total_seconds: float | None = None,
    file_size_bytes: int | None = None,
    error_message: str | None = None,
    client_ip: str | None = None,
) -> DocumentLog:
    kwargs: dict = {}
    if id is not None:
        kwargs["id"] = id
    log = DocumentLog(
        **kwargs,
        filename=filename,
        file_extension=file_extension,
        document_type=document_type,
        confidence=confidence,
        status=status,
        ocr_seconds=ocr_seconds,
        classification_seconds=classification_seconds,
        extraction_seconds=extraction_seconds,
        total_seconds=total_seconds,
        file_size_bytes=file_size_bytes,
        error_message=error_message,
        client_ip=client_ip,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


async def get_logs(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
    document_type: str | None = None,
    status: str | None = None,
) -> list[DocumentLog]:
    stmt = select(DocumentLog).order_by(DocumentLog.created_at.desc())
    if document_type:
        stmt = stmt.where(DocumentLog.document_type == document_type)
    if status:
        stmt = stmt.where(DocumentLog.status == status)
    stmt = stmt.limit(limit).offset(offset)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_stats(db: AsyncSession) -> dict[str, Any]:
    """Aggregate statistics across all processed documents."""

    total = await db.scalar(select(func.count()).select_from(DocumentLog))

    # Counts per status
    status_rows = await db.execute(
        select(DocumentLog.status, func.count().label("n"))
        .group_by(DocumentLog.status)
    )
    by_status = {row.status: row.n for row in status_rows}

    # Counts per document_type (success only)
    type_rows = await db.execute(
        select(DocumentLog.document_type, func.count().label("n"))
        .where(DocumentLog.status == "success")
        .group_by(DocumentLog.document_type)
    )
    by_type = {row.document_type: row.n for row in type_rows}

    # Average latencies for successful predictions
    avg_rows = await db.execute(
        select(
            func.avg(DocumentLog.ocr_seconds).label("avg_ocr"),
            func.avg(DocumentLog.classification_seconds).label("avg_cls"),
            func.avg(DocumentLog.extraction_seconds).label("avg_ext"),
            func.avg(DocumentLog.total_seconds).label("avg_total"),
        ).where(DocumentLog.status == "success")
    )
    avg = avg_rows.one()

    # Average confidence for successful predictions
    avg_conf = await db.scalar(
        select(func.avg(DocumentLog.confidence)).where(
            DocumentLog.status == "success"
        )
    )

    return {
        "total_documents_processed": total,
        "by_status": by_status,
        "by_document_type": by_type,
        "average_confidence": round(avg_conf, 4) if avg_conf else None,
        "average_latency_seconds": {
            "ocr": round(avg.avg_ocr, 3) if avg.avg_ocr else None,
            "classification": round(avg.avg_cls, 3) if avg.avg_cls else None,
            "extraction": round(avg.avg_ext, 3) if avg.avg_ext else None,
            "total": round(avg.avg_total, 3) if avg.avg_total else None,
        },
    }


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

async def insert_feedback(
    db: AsyncSession,
    *,
    doc_log_id: str,
    predicted_label: str,
    correct_label: str,
    is_correct: bool,
    user_note: str | None = None,
) -> FeedbackLog:
    row = FeedbackLog(
        doc_log_id=doc_log_id,
        predicted_label=predicted_label,
        correct_label=correct_label,
        is_correct=is_correct,
        user_note=user_note,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_model_metrics(db: AsyncSession) -> dict[str, Any]:
    """
    Compute live precision / recall / F1 per class from FeedbackLog rows.

    Each feedback row represents one human-verified prediction:
    - predicted_label = what the model said
    - correct_label   = what it actually was (= predicted_label when is_correct)

    We build a confusion matrix from these and derive per-class metrics.
    """
    rows = await db.execute(
        select(
            FeedbackLog.predicted_label,
            FeedbackLog.correct_label,
            FeedbackLog.is_correct,
        )
    )
    feedback = rows.all()

    total_feedback = len(feedback)
    if total_feedback == 0:
        return {
            "total_feedback": 0,
            "agreement_rate": None,
            "per_class": {},
            "confusion_matrix": {},
            "message": "No feedback submitted yet. Use the thumbs-up/down buttons after each prediction.",
        }

    # ── Aggregate into confusion matrix ──────────────────────────────────────
    # confusion[actual][predicted] = count
    confusion: dict[str, dict[str, int]] = {}
    classes: set[str] = set()

    for row in feedback:
        actual    = row.correct_label
        predicted = row.predicted_label
        classes.add(actual)
        classes.add(predicted)
        confusion.setdefault(actual, {})
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

    sorted_classes = sorted(classes)

    # ── Per-class precision / recall / F1 ────────────────────────────────────
    per_class: dict[str, Any] = {}
    total_correct = sum(1 for r in feedback if r.is_correct)

    for cls in sorted_classes:
        # TP: predicted cls AND actual cls
        tp = confusion.get(cls, {}).get(cls, 0)
        # FP: predicted cls but actual was something else
        fp = sum(
            confusion.get(actual, {}).get(cls, 0)
            for actual in sorted_classes if actual != cls
        )
        # FN: actual cls but predicted something else
        fn = sum(confusion.get(cls, {}).get(pred, 0)
                 for pred in sorted_classes if pred != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall    = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision is not None and recall is not None
                and (precision + recall) > 0)
            else None
        )
        support = tp + fn   # total actual instances of this class

        per_class[cls] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall":    round(recall, 4)    if recall is not None    else None,
            "f1":        round(f1, 4)        if f1 is not None        else None,
            "support":   support,
            "tp": tp, "fp": fp, "fn": fn,
        }

    agreement_rate = total_correct / total_feedback if total_feedback else None

    # Macro-average F1 (only classes with support > 0)
    f1_scores = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else None

    return {
        "total_feedback":  total_feedback,
        "agreement_rate":  round(agreement_rate, 4) if agreement_rate is not None else None,
        "macro_f1":        macro_f1,
        "per_class":       per_class,
        "confusion_matrix": {
            actual: {pred: confusion.get(actual, {}).get(pred, 0) for pred in sorted_classes}
            for actual in sorted_classes
        },
        "classes": sorted_classes,
    }
