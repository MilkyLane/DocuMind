"""
Tests for GET /metrics endpoint.

Covers:
- Empty state (no feedback yet)
- After seeding thumbs-up feedback → agreement_rate = 1.0
- After seeding mixed feedback → per-class metrics present
"""
import pytest


@pytest.mark.asyncio
async def test_metrics_empty_state(client):
    """No feedback → metrics endpoint returns an informative empty response."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_feedback"] == 0
    assert body["agreement_rate"] is None
    assert body["per_class"] == {}
    assert "message" in body   # helpful hint for the user


@pytest.mark.asyncio
async def test_metrics_after_correct_prediction(client, seed_document):
    """
    One thumbs-up → agreement_rate 1.0, per_class has the predicted class,
    macro_f1 is populated.
    """
    doc = await seed_document(document_type="invoice")
    await client.post("/feedback", json={"job_id": doc.id, "is_correct": True})

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_feedback"] == 1
    assert body["agreement_rate"] == pytest.approx(1.0)
    assert "invoice" in body["per_class"]
    assert body["per_class"]["invoice"]["precision"] == pytest.approx(1.0)
    assert body["per_class"]["invoice"]["recall"] == pytest.approx(1.0)
    assert body["macro_f1"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_metrics_after_incorrect_prediction(client, seed_document):
    """
    One thumbs-down (model said invoice, correct was resume) →
    agreement_rate 0.0, invoice precision = 0.
    """
    doc = await seed_document(document_type="invoice")
    await client.post("/feedback", json={
        "job_id": doc.id,
        "is_correct": False,
        "correct_label": "resume",
    })

    resp = await client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_feedback"] == 1
    assert body["agreement_rate"] == pytest.approx(0.0)

    # Invoice was predicted but actual was resume → FP for invoice
    invoice_metrics = body["per_class"].get("invoice", {})
    assert invoice_metrics.get("precision") == pytest.approx(0.0)

    # Resume was the actual class but was never predicted → FN
    resume_metrics = body["per_class"].get("resume", {})
    assert resume_metrics.get("recall") == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_metrics_mixed_feedback(client, seed_document):
    """
    Two correct + one incorrect predictions.
    agreement_rate = 2/3 ≈ 0.6667.
    """
    doc1 = await seed_document(document_type="invoice")
    doc2 = await seed_document(document_type="invoice")
    doc3 = await seed_document(document_type="resume")

    await client.post("/feedback", json={"job_id": doc1.id, "is_correct": True})
    await client.post("/feedback", json={"job_id": doc2.id, "is_correct": True})
    await client.post("/feedback", json={
        "job_id": doc3.id,
        "is_correct": False,
        "correct_label": "form",
    })

    resp = await client.get("/metrics")
    body = resp.json()

    assert body["total_feedback"] == 3
    assert body["agreement_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert "classes" in body
    assert "confusion_matrix" in body


@pytest.mark.asyncio
async def test_metrics_confusion_matrix_structure(client, seed_document):
    """Confusion matrix keys must be all known classes seen in feedback."""
    doc = await seed_document(document_type="bank_statement")
    await client.post("/feedback", json={"job_id": doc.id, "is_correct": True})

    resp = await client.get("/metrics")
    body = resp.json()

    cm = body["confusion_matrix"]
    # All row keys must also be in the classes list
    for cls in body["classes"]:
        assert cls in cm
