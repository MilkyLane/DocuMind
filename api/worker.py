"""
ARQ worker — runs as a separate process alongside the FastAPI app.

Start with:
    arq api.worker.WorkerSettings

Each job:
1. Pulls the uploaded file bytes from Redis (stored by the API on enqueue).
2. Writes them to a temp file and calls predict().
3. Stores the JSON result back in Redis with a TTL so the API can poll it.
4. Logs the result to PostgreSQL.
5. Cleans up the raw-bytes key from Redis.
"""

import json
import os
import tempfile
from pathlib import Path

from arq import ArqRedis
from arq.connections import RedisSettings
from dotenv import load_dotenv

from src.db.crud import insert_document_log
from src.db.database import AsyncSessionLocal
from src.inference.predict import predict
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger()

REDIS_URL: str = os.environ["REDIS_URL"]

# How long (seconds) to keep the finished result in Redis before expiry.
# Long enough for any realistic polling loop; keeps Redis memory tidy.
RESULT_TTL_SECONDS: int = int(os.getenv("JOB_RESULT_TTL_SECONDS", "300"))

# Key prefixes
FILE_KEY_PREFIX = "job:file:"
RESULT_KEY_PREFIX = "job:result:"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
async def predict_document_task(
    ctx: dict,
    *,
    job_id: str,
    filename: str,
    file_extension: str,
    file_size_bytes: int,
    client_ip: str | None,
) -> None:
    """
    ARQ task.  The raw file bytes were stored in Redis by the API under
    ``job:file:<job_id>`` before this job was enqueued.
    """
    redis: ArqRedis = ctx["redis"]

    # --- Retrieve file bytes from Redis ---
    raw: bytes | None = await redis.get(f"{FILE_KEY_PREFIX}{job_id}")
    if raw is None:
        logger.warning(f"WORKER no file bytes found for job_id={job_id}")
        result = {"status": "error", "reason": "File bytes expired before processing"}
        await redis.set(
            f"{RESULT_KEY_PREFIX}{job_id}",
            json.dumps(result),
            ex=RESULT_TTL_SECONDS,
        )
        return

    # --- Write to temp file and run predict ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        tmp_path.write_bytes(raw)

        try:
            result = predict(str(tmp_path))
        except Exception as exc:
            logger.error(f"WORKER predict error job_id={job_id} error={exc}")
            result = {"status": "error", "reason": str(exc)}

    # --- Store result in Redis for the API to serve ---
    await redis.set(
        f"{RESULT_KEY_PREFIX}{job_id}",
        json.dumps(result),
        ex=RESULT_TTL_SECONDS,
    )
    logger.info(
        f"WORKER done job_id={job_id} status={result.get('status')} "
        f"type={result.get('document_type')} total={result.get('latency', {}).get('total_seconds')}s"
    )

    # --- Clean up the raw bytes key ---
    await redis.delete(f"{FILE_KEY_PREFIX}{job_id}")

    # --- Persist to PostgreSQL ---
    latency: dict = result.get("latency", {})
    async with AsyncSessionLocal() as db:
        await insert_document_log(
            db,
            filename=filename,
            file_extension=file_extension,
            document_type=result.get("document_type", "unknown"),
            confidence=result.get("confidence"),
            status=result.get("status", "unknown"),
            ocr_seconds=latency.get("ocr_seconds"),
            classification_seconds=latency.get("classification_seconds"),
            extraction_seconds=latency.get("extraction_seconds"),
            total_seconds=latency.get("total_seconds"),
            file_size_bytes=file_size_bytes,
            error_message=result.get("reason") if result.get("status") == "error" else None,
            client_ip=client_ip,
        )


# ---------------------------------------------------------------------------
# Worker settings — used by: arq api.worker.WorkerSettings
# ---------------------------------------------------------------------------
class WorkerSettings:
    functions = [predict_document_task]
    redis_settings = RedisSettings.from_dsn(REDIS_URL)

    @staticmethod
    async def on_startup(ctx: dict) -> None:
        logger.info("ARQ worker started")

    @staticmethod
    async def on_shutdown(ctx: dict) -> None:
        logger.info("ARQ worker shutting down")
