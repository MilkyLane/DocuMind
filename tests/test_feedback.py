"""
Tests for POST /feedback.

Covers:
- 404 on unknown job_id
- 422 on missing required fields
- Thumbs-up stores correct_label = predicted_label
- Thumbs-down with valid label stores correction
- Thumbs-down with invalid label returns 422
- Thumbs-down without correct_label returns 422
"""
import uuid

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _feedback(client, payload: dict):
    return await client.post("/feedback", json=payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feedback_404_for_unknown_job(client):
    """Submitting feedback for a non-existent job_id must return 404."""
    resp = await _feedback(client, {
        "job_id": str(uuid.uuid4()),
        "is_correct": True,
    })
    assert resp.status_code == 404
    assert "No prediction found" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_feedback_422_missing_job_id(client):
    resp = await _feedback(client, {"is_correct": True})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_422_missing_is_correct(client):
    resp = await _feedback(client, {"job_id": str(uuid.uuid4())})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_thumbs_up(client, seed_document):
    """Thumbs-up: correct_label should be set to the predicted label."""
    doc = await seed_document(document_type="invoice")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": True,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_correct"] is True
    assert body["predicted_label"] == "invoice"
    assert body["correct_label"] == "invoice"   # matches prediction
    assert body["job_id"] == doc.id
    assert "feedback_id" in body


@pytest.mark.asyncio
async def test_feedback_thumbs_down_valid_label(client, seed_document):
    """Thumbs-down with a valid correction label must be stored."""
    doc = await seed_document(document_type="invoice")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": False,
        "correct_label": "resume",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_correct"] is False
    assert body["predicted_label"] == "invoice"
    assert body["correct_label"] == "resume"


@pytest.mark.asyncio
async def test_feedback_thumbs_down_normalises_label(client, seed_document):
    """Labels with spaces / mixed case must be normalised ('Bank Statement' → 'bank_statement')."""
    doc = await seed_document(document_type="invoice")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": False,
        "correct_label": "Bank Statement",
    })
    assert resp.status_code == 200
    assert resp.json()["correct_label"] == "bank_statement"


@pytest.mark.asyncio
async def test_feedback_thumbs_down_invalid_label(client, seed_document):
    """An unrecognised label must return 422."""
    doc = await seed_document(document_type="invoice")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": False,
        "correct_label": "selfie",
    })
    assert resp.status_code == 422
    assert "Unknown label" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_feedback_thumbs_down_missing_correct_label(client, seed_document):
    """Thumbs-down without correct_label must return 422."""
    doc = await seed_document(document_type="invoice")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": False,
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feedback_accepts_optional_user_note(client, seed_document):
    """A user_note should be accepted and stored without error."""
    doc = await seed_document(document_type="form")

    resp = await _feedback(client, {
        "job_id": doc.id,
        "is_correct": True,
        "user_note": "Looks right to me.",
    })
    assert resp.status_code == 200
