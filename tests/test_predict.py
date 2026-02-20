"""
Tests for POST /predict and GET /predict/{job_id}.

All Redis / ARQ calls are mocked — no real worker runs.
"""
import io
import json
import uuid

import pytest


# ---------------------------------------------------------------------------
# POST /predict — input validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_predict_rejects_unsupported_extension(client):
    """A .exe file must return 400."""
    data = {"file": ("malware.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    response = await client.post("/predict", files=data)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_predict_rejects_missing_file(client):
    """No file attached → 422 Unprocessable Entity from FastAPI."""
    response = await client.post("/predict")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_accepts_pdf_and_returns_job_id(client, mock_arq):
    """
    A valid PDF upload should be enqueued and return a job_id.
    Because the worker is mocked (ARQ get() returns None) the inline wait
    times out and the response contains status=queued.
    """
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    data = {"file": ("statement.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = await client.post("/predict", files=data)

    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    # Worker didn't respond (mock returns None), so inline wait times out
    assert body["status"] == "queued"
    # Confirm the job was actually enqueued
    mock_arq.enqueue_job.assert_called_once()


@pytest.mark.asyncio
async def test_predict_accepts_image_extensions(client):
    """PNG, JPG, JPEG, TIFF should all be accepted (not 400)."""
    for ext, mime in [
        ("doc.png", "image/png"),
        ("doc.jpg", "image/jpeg"),
        ("doc.jpeg", "image/jpeg"),
        ("doc.tiff", "image/tiff"),
    ]:
        data = {"file": (ext, io.BytesIO(b"fake image"), mime)}
        response = await client.post("/predict", files=data)
        # May be 200 (queued) — must NOT be 400
        assert response.status_code != 400, f"Rejected valid extension: {ext}"


@pytest.mark.asyncio
async def test_predict_inline_returns_result_when_worker_fast(client, mock_arq):
    """
    If the worker stores the result in Redis before the inline timeout,
    POST /predict should return the full result (not just queued).
    """
    fake_result = json.dumps({
        "status": "success",
        "document_type": "invoice",
        "confidence": 0.97,
        "fields": {},
    }).encode()

    # Make ARQ.get() immediately return the result
    mock_arq.get.return_value = fake_result

    pdf_bytes = b"%PDF-1.4 inline result test"
    data = {"file": ("invoice.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = await client.post("/predict", files=data)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["document_type"] == "invoice"
    assert body["confidence"] == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# GET /predict/{job_id}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_prediction_returns_pending_while_processing(client, mock_arq):
    """
    No result key but file key exists → status pending.
    """
    job_id = str(uuid.uuid4())
    mock_arq.get.return_value = None     # no result yet
    mock_arq.exists.return_value = 1     # file key present → job is in progress

    response = await client.get(f"/predict/{job_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "pending"


@pytest.mark.asyncio
async def test_get_prediction_returns_404_for_unknown_job(client, mock_arq):
    """
    No result key AND no file key → job never existed or expired → 404.
    """
    job_id = str(uuid.uuid4())
    mock_arq.get.return_value = None
    mock_arq.exists.return_value = 0

    response = await client.get(f"/predict/{job_id}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_prediction_returns_result_when_complete(client, mock_arq):
    """
    Result key present → return full prediction payload.
    """
    job_id = str(uuid.uuid4())
    fake_result = json.dumps({
        "status": "success",
        "document_type": "resume",
        "confidence": 0.91,
        "fields": {},
    }).encode()
    mock_arq.get.return_value = fake_result

    response = await client.get(f"/predict/{job_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["document_type"] == "resume"
    assert body["job_id"] == job_id
