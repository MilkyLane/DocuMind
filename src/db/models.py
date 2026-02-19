import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from src.db.database import Base


class DocumentLog(Base):
    """One row per document processed by the /predict endpoint."""

    __tablename__ = "document_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_extension: Mapped[str] = mapped_column(String(16), nullable=False)

    # Classification outcome
    document_type: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "success" | "rejected" | "ocr_failed" | "error"

    # Latency breakdown (seconds)
    ocr_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    classification_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # File size in bytes
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error details (populated only when status="error")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Client info
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
