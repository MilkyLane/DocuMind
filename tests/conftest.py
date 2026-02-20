"""
Shared pytest fixtures.

Strategy
--------
- Use an **in-memory SQLite** database (via aiosqlite) so tests have zero
  external dependencies — no real PostgreSQL or Redis required.
- Patch `src.db.database.engine` and `get_db` so the FastAPI app uses the
  test database.
- Provide a mock ARQ pool so /predict and /explain never touch Redis.
"""

import json
import os
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Set env vars BEFORE any app module is imported so database.py / worker.py
# don't crash on missing environment variables.
# ---------------------------------------------------------------------------
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("INLINE_WAIT_SECONDS", "0.1")   # speed up poll loop in tests

# ---------------------------------------------------------------------------
# In-memory SQLite engine (shared across the whole test session)
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


# ---------------------------------------------------------------------------
# Create / drop all tables once per session
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_test_tables():
    """Create all ORM tables in the in-memory SQLite DB once per session."""
    # Import Base after env vars are set
    from src.db.database import Base  # noqa: F401
    import src.db.models  # noqa: F401 — register models with Base

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ---------------------------------------------------------------------------
# Per-test DB session (rolls back after each test)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Mock ARQ pool — used by the FastAPI app's request.app.state.arq
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_arq():
    """
    Returns a mock ARQ pool.

    By default:
    - get()          → None   (no result stored yet)
    - set()          → True
    - exists()       → 0
    - enqueue_job()  → MagicMock job handle
    """
    pool = AsyncMock()
    pool.get.return_value = None
    pool.set.return_value = True
    pool.exists.return_value = 0
    pool.enqueue_job.return_value = MagicMock(job_id=str(uuid.uuid4()))
    pool.aclose = AsyncMock()
    return pool


# ---------------------------------------------------------------------------
# Mock aioredis client — used by the rate limiter
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.incr.return_value = 1
    r.expire.return_value = True
    r.aclose = AsyncMock()
    return r


# ---------------------------------------------------------------------------
# FastAPI test client wired to test DB + mock Redis/ARQ
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client(db_session, mock_arq, mock_redis) -> AsyncGenerator[AsyncClient, None]:
    from api.main import app
    from src.db.database import get_db

    # Override the DB dependency
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Inject mock pools into app state (bypasses lifespan startup)
    app.state.arq = mock_arq
    app.state.redis = mock_redis

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper: seed a DocumentLog row directly (used by feedback/metrics tests)
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def seed_document(db_session):
    """
    Insert a successful DocumentLog row and return it.
    Accepts keyword overrides so each test can customise the row.
    """
    from src.db.crud import insert_document_log

    async def _seed(
        *,
        document_type: str = "invoice",
        confidence: float = 0.95,
        status: str = "success",
        job_id: str | None = None,
    ):
        return await insert_document_log(
            db_session,
            id=job_id or str(uuid.uuid4()),
            filename="test_doc.pdf",
            file_extension=".pdf",
            document_type=document_type,
            confidence=confidence,
            status=status,
            total_seconds=1.0,
            file_size_bytes=1024,
        )

    return _seed
