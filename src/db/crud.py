from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DocumentLog


async def insert_document_log(
    db: AsyncSession,
    *,
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
    log = DocumentLog(
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
