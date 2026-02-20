import json
import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from arq import create_pool
from arq.connections import RedisSettings
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.crud import get_logs, get_stats, insert_document_log, insert_feedback, get_model_metrics
from src.db.database import Base, engine, get_db
from src.db.models import DocumentLog, FeedbackLog  # noqa: F401 — ensures models are registered
from src.middleware.rate_limit import get_client_ip, rate_limit
from src.utils.logging import get_logger
from api.worker import FILE_KEY_PREFIX, RESULT_KEY_PREFIX, RESULT_TTL_SECONDS, explain_document_task

load_dotenv()
logger = get_logger()

REDIS_URL: str = os.environ["REDIS_URL"]
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}

# File bytes uploaded for a job expire after this many seconds.
# Must be long enough for the worker to pick up the job.
FILE_TTL_SECONDS: int = int(os.getenv("JOB_FILE_TTL_SECONDS", "60"))

# How long POST /predict will wait inline for the result before giving up
# and returning just the job_id for manual polling.
INLINE_WAIT_SECONDS: float = float(os.getenv("INLINE_WAIT_SECONDS", "12.0"))
INLINE_POLL_INTERVAL: float = 0.3  # seconds between Redis checks


# ---------------------------------------------------------------------------
# Lifespan: create tables + connection pools on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified/created")

    # aioredis pool — used by the rate limiter
    app.state.redis = aioredis.from_url(
        REDIS_URL, encoding="utf-8", decode_responses=True
    )

    # ARQ pool — used to enqueue jobs and store file bytes
    app.state.arq = await create_pool(RedisSettings.from_dsn(REDIS_URL))
    logger.info("Redis pools created (rate-limit + ARQ)")

    yield

    await app.state.redis.aclose()
    await app.state.arq.aclose()
    await engine.dispose()
    logger.info("Shutdown: connections closed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DocuMind",
    description="Document classification and field extraction API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the frontend — must be mounted before the catch-all route
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "message": "DocuMind API is running"}


@app.post("/predict", dependencies=[Depends(rate_limit)])
async def predict_document(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Enqueue a prediction job and wait inline for the result (~0.5 s for a
    single-page doc).  Returns the full result directly — no manual polling
    needed.  If the worker takes longer than INLINE_WAIT_SECONDS the endpoint
    falls back to returning ``{"job_id": ..., "status": "queued"}`` so the
    client can poll GET /predict/{job_id} itself.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_bytes = await file.read()
    file_size = len(file_bytes)
    client_ip = get_client_ip(request)
    job_id = str(uuid.uuid4())

    arq = request.app.state.arq

    # Store raw bytes in Redis so the worker can retrieve them
    await arq.set(f"{FILE_KEY_PREFIX}{job_id}", file_bytes, ex=FILE_TTL_SECONDS)

    # Enqueue the task
    await arq.enqueue_job(
        "predict_document_task",
        job_id=job_id,
        filename=file.filename,
        file_extension=suffix,
        file_size_bytes=file_size,
        client_ip=client_ip,
        _job_id=job_id,
    )

    logger.info(f"ENQUEUED job_id={job_id} file={file.filename} size={file_size}B")

    # ── Inline wait ───────────────────────────────────────────────────────────
    elapsed = 0.0
    while elapsed < INLINE_WAIT_SECONDS:
        await asyncio.sleep(INLINE_POLL_INTERVAL)
        elapsed += INLINE_POLL_INTERVAL
        raw: bytes | None = await arq.get(f"{RESULT_KEY_PREFIX}{job_id}")
        if raw is not None:
            result: dict[str, Any] = json.loads(raw)
            return {"job_id": job_id, **result}

    # Worker hasn't finished within the timeout — return job_id for async polling
    logger.warning(f"INLINE_TIMEOUT job_id={job_id} waited={elapsed:.1f}s")
    return {"job_id": job_id, "status": "queued"}


@app.get("/predict/{job_id}", dependencies=[Depends(rate_limit)])
async def get_prediction_result(job_id: str, request: Request):
    """
    Manual poll endpoint — only needed if POST /predict timed out.

    Returns:
    - ``{"status": "pending"}`` while the worker hasn't finished yet.
    - The full prediction result dict once complete.
    - HTTP 404 if the job_id is unknown or the result has expired.
    """
    arq = request.app.state.arq
    raw: bytes | None = await arq.get(f"{RESULT_KEY_PREFIX}{job_id}")

    if raw is None:
        file_exists = await arq.exists(f"{FILE_KEY_PREFIX}{job_id}")
        if file_exists:
            return {"job_id": job_id, "status": "pending"}
        raise HTTPException(
            status_code=404,
            detail="Job not found. It may have expired or never existed.",
        )

    result: dict[str, Any] = json.loads(raw)
    return {"job_id": job_id, **result}


@app.post("/explain", dependencies=[Depends(rate_limit)])
async def explain_document(
    request: Request,
    file: UploadFile = File(...),
):
    """
    OCR the uploaded document and return the top TF-IDF features that drove
    the classification decision — i.e. *why* the model chose that class.

    Response shape:
    {
        "job_id": str,
        "status": "success",
        "predicted_class": str,
        "confidence": float,
        "top_features": [{"term": str, "weight": float, "direction": "for"|"against"}, ...],
        "all_classes": [str, ...]
    }
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    file_bytes = await file.read()
    job_id = str(uuid.uuid4())
    arq = request.app.state.arq

    await arq.set(f"{FILE_KEY_PREFIX}{job_id}", file_bytes, ex=FILE_TTL_SECONDS)
    await arq.enqueue_job(
        "explain_document_task",
        job_id=job_id,
        filename=file.filename,
        _job_id=job_id,
    )

    logger.info(f"EXPLAIN_ENQUEUED job_id={job_id} file={file.filename}")

    # Inline wait — explain is fast (no extraction step)
    elapsed = 0.0
    while elapsed < INLINE_WAIT_SECONDS:
        await asyncio.sleep(INLINE_POLL_INTERVAL)
        elapsed += INLINE_POLL_INTERVAL
        raw: bytes | None = await arq.get(f"{RESULT_KEY_PREFIX}{job_id}")
        if raw is not None:
            return {"job_id": job_id, **json.loads(raw)}

    return {"job_id": job_id, "status": "queued"}


@app.get("/logs", dependencies=[Depends(rate_limit)])
async def list_logs(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    document_type: str | None = Query(None),
    status: str | None = Query(None),
):
    """Return recent document processing logs with optional filters."""
    logs = await get_logs(
        db,
        limit=limit,
        offset=offset,
        document_type=document_type,
        status=status,
    )
    return [
        {
            "id": log.id,
            "filename": log.filename,
            "file_extension": log.file_extension,
            "document_type": log.document_type,
            "confidence": log.confidence,
            "status": log.status,
            "latency": {
                "ocr_seconds": log.ocr_seconds,
                "classification_seconds": log.classification_seconds,
                "extraction_seconds": log.extraction_seconds,
                "total_seconds": log.total_seconds,
            },
            "file_size_bytes": log.file_size_bytes,
            "client_ip": log.client_ip,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@app.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    """Aggregate statistics: totals, breakdown by type/status, avg latencies."""
    return await get_stats(db)


# Known document classes — used to validate feedback labels
KNOWN_CLASSES = {"invoice", "resume", "form", "bank_statement", "utility"}


@app.post("/feedback", dependencies=[Depends(rate_limit)])
async def submit_feedback(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Record human feedback on a classification result.

    Expected JSON body:
    {
        "job_id":        str,            -- the prediction's job_id
        "is_correct":    bool,           -- thumbs-up (true) or thumbs-down (false)
        "correct_label": str | null,     -- required when is_correct=false
        "user_note":     str | null      -- optional free-text comment
    }

    The job_id must match a row in document_logs.  On thumbs-up the
    correct_label defaults to the predicted label (no correction needed).
    """
    body = await request.json()
    job_id       = body.get("job_id")
    is_correct   = body.get("is_correct")
    correct_label = body.get("correct_label")
    user_note    = body.get("user_note")

    if not job_id or is_correct is None:
        raise HTTPException(status_code=422, detail="job_id and is_correct are required")

    # Look up the original prediction
    from sqlalchemy import select as sa_select
    result = await db.execute(
        sa_select(DocumentLog).where(DocumentLog.id == job_id)
    )
    doc = result.scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for job_id={job_id}. "
                   "Feedback can only be submitted for successful predictions.",
        )

    if is_correct:
        # Thumbs-up — model was right, correct label = predicted label
        correct_label = doc.document_type
    else:
        if not correct_label:
            raise HTTPException(
                status_code=422,
                detail="correct_label is required when is_correct=false",
            )
        correct_label = correct_label.lower().replace(" ", "_")
        if correct_label not in KNOWN_CLASSES:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown label '{correct_label}'. "
                       f"Valid options: {sorted(KNOWN_CLASSES)}",
            )

    fb = await insert_feedback(
        db,
        doc_log_id=job_id,
        predicted_label=doc.document_type,
        correct_label=correct_label,
        is_correct=bool(is_correct),
        user_note=user_note,
    )

    logger.info(
        f"FEEDBACK job_id={job_id} predicted={doc.document_type} "
        f"correct={correct_label} is_correct={is_correct}"
    )

    return {
        "feedback_id": fb.id,
        "job_id": job_id,
        "predicted_label": doc.document_type,
        "correct_label": correct_label,
        "is_correct": fb.is_correct,
        "created_at": fb.created_at.isoformat(),
    }


@app.get("/metrics")
async def model_metrics(db: AsyncSession = Depends(get_db)):
    """
    Live model performance metrics derived from user feedback.

    Returns per-class precision, recall, F1, support, a confusion matrix,
    overall agreement rate, and macro-averaged F1.  All computed from
    FeedbackLog rows — i.e. real-world human corrections, not held-out data.
    """
    return await get_model_metrics(db)
