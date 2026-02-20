"""
Unit tests for src.inference.predict.classify().

The real sklearn pipeline is replaced with a lightweight mock so these
tests run instantly with no model file required.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_pipeline(classes, probs):
    """Return a mock pipeline whose predict_proba returns `probs`."""
    mock_pipeline = MagicMock()
    mock_pipeline.classes_ = np.array(classes)
    mock_pipeline.predict_proba.return_value = np.array([probs])
    return mock_pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_classify_returns_highest_confidence_class():
    """classify() should pick the argmax class."""
    mock_pipeline = _make_pipeline(
        classes=["bank_statement", "form", "invoice", "resume", "utility"],
        probs=[0.05, 0.05, 0.80, 0.05, 0.05],
    )
    with patch("src.inference.predict.pipeline", mock_pipeline):
        from src.inference.predict import classify
        label, conf = classify("some invoice text")

    assert label == "invoice"
    assert conf == pytest.approx(0.80)


def test_classify_normalises_label_with_spaces():
    """Labels like 'Bank Statement' must be normalised to 'bank_statement'."""
    mock_pipeline = _make_pipeline(
        classes=["Bank Statement", "form", "invoice", "resume", "utility"],
        probs=[0.90, 0.02, 0.03, 0.02, 0.03],
    )
    with patch("src.inference.predict.pipeline", mock_pipeline):
        from src.inference.predict import classify
        label, conf = classify("IBAN balance credit debit")

    assert label == "bank_statement"
    assert conf == pytest.approx(0.90)


def test_classify_normalises_mixed_case():
    """Labels like 'Resume' must become 'resume'."""
    mock_pipeline = _make_pipeline(
        classes=["bank_statement", "form", "invoice", "Resume", "utility"],
        probs=[0.01, 0.01, 0.01, 0.95, 0.02],
    )
    with patch("src.inference.predict.pipeline", mock_pipeline):
        from src.inference.predict import classify
        label, conf = classify("skills experience education")

    assert label == "resume"
    assert conf == pytest.approx(0.95)


def test_classify_returns_float_confidence():
    """Confidence must be a Python float (not np.float32/64)."""
    mock_pipeline = _make_pipeline(
        classes=["invoice"],
        probs=[1.0],
    )
    with patch("src.inference.predict.pipeline", mock_pipeline):
        from src.inference.predict import classify
        label, conf = classify("any text")

    assert isinstance(conf, float)


def test_classify_confidence_threshold_not_enforced_by_classify():
    """
    classify() itself does NOT filter low-confidence results —
    that's predict()'s job.  Even a 0.20 confidence should be returned.
    """
    mock_pipeline = _make_pipeline(
        classes=["invoice", "form"],
        probs=[0.20, 0.80],
    )
    with patch("src.inference.predict.pipeline", mock_pipeline):
        from src.inference.predict import classify
        label, conf = classify("ambiguous document text")

    assert label == "form"
    assert conf == pytest.approx(0.80)
