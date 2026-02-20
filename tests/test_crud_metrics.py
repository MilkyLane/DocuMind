"""
Unit tests for get_model_metrics() CRUD function.

These test the metric calculation logic directly against an in-memory DB,
independent of the HTTP layer.
"""
import uuid

import pytest

from src.db.crud import get_model_metrics, insert_document_log, insert_feedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _doc(db, doc_type="invoice"):
    return await insert_document_log(
        db,
        id=str(uuid.uuid4()),
        filename="test.pdf",
        file_extension=".pdf",
        document_type=doc_type,
        confidence=0.9,
        status="success",
        total_seconds=1.0,
        file_size_bytes=512,
    )


async def _fb(db, doc, predicted, correct, is_correct):
    return await insert_feedback(
        db,
        doc_log_id=doc.id,
        predicted_label=predicted,
        correct_label=correct,
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_empty_db(db_session):
    result = await get_model_metrics(db_session)
    assert result["total_feedback"] == 0
    assert result["agreement_rate"] is None
    assert result["per_class"] == {}


@pytest.mark.asyncio
async def test_metrics_all_correct(db_session):
    """3 correct predictions of the same class → P=R=F1=1.0, agreement=1.0."""
    for _ in range(3):
        doc = await _doc(db_session, "invoice")
        await _fb(db_session, doc, "invoice", "invoice", True)

    result = await get_model_metrics(db_session)

    assert result["total_feedback"] == 3
    assert result["agreement_rate"] == pytest.approx(1.0)
    cls = result["per_class"]["invoice"]
    assert cls["precision"] == pytest.approx(1.0)
    assert cls["recall"] == pytest.approx(1.0)
    assert cls["f1"] == pytest.approx(1.0)
    assert cls["support"] == 3


@pytest.mark.asyncio
async def test_metrics_all_wrong(db_session):
    """
    Model always predicts 'invoice' but actual is always 'resume'.
    invoice → precision = 0 (all FP), recall = undefined (0 TP, 0 FN for invoice).
    resume  → recall = 0 (all FN), precision = undefined.
    """
    for _ in range(2):
        doc = await _doc(db_session, "invoice")
        await _fb(db_session, doc, "invoice", "resume", False)

    result = await get_model_metrics(db_session)

    assert result["total_feedback"] == 2
    assert result["agreement_rate"] == pytest.approx(0.0)

    invoice = result["per_class"]["invoice"]
    assert invoice["precision"] == pytest.approx(0.0)   # 0 TP, 2 FP
    assert invoice["tp"] == 0
    assert invoice["fp"] == 2

    resume = result["per_class"]["resume"]
    assert resume["recall"] == pytest.approx(0.0)        # 0 TP, 2 FN
    assert resume["fn"] == 2


@pytest.mark.asyncio
async def test_metrics_macro_f1_multiple_classes(db_session):
    """
    2 classes, each with perfect predictions → macro_f1 = 1.0.
    """
    for doc_type in ("form", "utility"):
        doc = await _doc(db_session, doc_type)
        await _fb(db_session, doc, doc_type, doc_type, True)

    result = await get_model_metrics(db_session)
    assert result["macro_f1"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_metrics_confusion_matrix_values(db_session):
    """
    Predicted invoice (correct) once + predicted invoice (wrong, actual=form) once.
    Confusion matrix:
      actual=invoice → predicted=invoice: 1
      actual=form    → predicted=invoice: 1
    """
    doc1 = await _doc(db_session, "invoice")
    await _fb(db_session, doc1, "invoice", "invoice", True)

    doc2 = await _doc(db_session, "invoice")   # model predicts invoice
    await _fb(db_session, doc2, "invoice", "form", False)   # actual=form

    result = await get_model_metrics(db_session)
    cm = result["confusion_matrix"]

    # actual=invoice, predicted=invoice → 1
    assert cm["invoice"]["invoice"] == 1
    # actual=form, predicted=invoice → 1
    assert cm["form"]["invoice"] == 1


@pytest.mark.asyncio
async def test_metrics_support_counts(db_session):
    """support = TP + FN = total actual instances of a class."""
    doc1 = await _doc(db_session, "resume")
    await _fb(db_session, doc1, "resume", "resume", True)   # TP

    doc2 = await _doc(db_session, "resume")
    await _fb(db_session, doc2, "resume", "form", False)    # FN (actual resume, predicted resume wrongly)
    # Note: doc2 predicted_label=resume, correct_label=form
    # → actual class is form, predicted is resume
    # So for resume: FP+=1; for form: FN+=1

    result = await get_model_metrics(db_session)
    # resume: 1 TP (correct pred) → support for resume = 1
    assert result["per_class"]["resume"]["support"] == 1
