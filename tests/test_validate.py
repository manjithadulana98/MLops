"""
Unit tests for src/ingestion/validate.py
IEC 62304 ref: SU-INGEST-01 Test Cases TC-001 – TC-008
"""

import numpy as np
import pytest

from src.ingestion.validate import ClinicalDataError, validate_ecg_signal, validate_ecg_record

SR = 360          # sampling rate Hz
DUR = 10.0        # duration seconds
N = SR * int(DUR) # expected sample count = 3600


# ── Helpers ─────────────────────────────────────────────────────────────────

def _good_signal(n: int = N) -> np.ndarray:
    """Return a valid synthetic ECG-like signal."""
    t = np.linspace(0, DUR, n, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm sinusoidal proxy


# ── TC-001: Valid signal passes without exception ────────────────────────────

def test_valid_signal_passes():
    validate_ecg_signal(_good_signal())


# ── TC-002: NaN values are rejected ─────────────────────────────────────────

def test_nan_raises_clinical_error():
    sig = _good_signal()
    sig[100] = float("nan")
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(sig)
    assert exc_info.value.check_name == "NON_FINITE_VALUES"


# ── TC-003: Infinite values are rejected ────────────────────────────────────

def test_inf_raises_clinical_error():
    sig = _good_signal()
    sig[200] = float("inf")
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(sig)
    assert exc_info.value.check_name == "NON_FINITE_VALUES"


# ── TC-004: Wrong length (too short) ────────────────────────────────────────

def test_too_short_raises_clinical_error():
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(_good_signal(N - 1))
    assert exc_info.value.check_name == "SIGNAL_LENGTH"


# ── TC-005: Wrong length (too long) ─────────────────────────────────────────

def test_too_long_raises_clinical_error():
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(_good_signal(N + 100))
    assert exc_info.value.check_name == "SIGNAL_LENGTH"


# ── TC-006: Voltage below lower bound ───────────────────────────────────────

def test_voltage_below_min_raises():
    sig = _good_signal()
    sig[50] = -6.0  # below -5 mV
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(sig)
    assert exc_info.value.check_name == "VOLTAGE_RANGE"


# ── TC-007: Voltage above upper bound ───────────────────────────────────────

def test_voltage_above_max_raises():
    sig = _good_signal()
    sig[50] = 5.1  # above +5 mV
    with pytest.raises(ClinicalDataError) as exc_info:
        validate_ecg_signal(sig)
    assert exc_info.value.check_name == "VOLTAGE_RANGE"


# ── TC-008: Non-ndarray input raises TypeError ───────────────────────────────

def test_non_ndarray_raises_type_error():
    with pytest.raises(TypeError):
        validate_ecg_signal([0.0] * N)  # plain list


# ── TC-009: Multi-lead record validation ─────────────────────────────────────

def test_multi_lead_record_valid():
    record = {"MLII": _good_signal(), "V5": _good_signal()}
    validate_ecg_record(record)


def test_multi_lead_record_bad_lead_raises():
    bad_sig = _good_signal()
    bad_sig[0] = float("nan")
    record = {"MLII": _good_signal(), "V5": bad_sig}
    with pytest.raises(ClinicalDataError):
        validate_ecg_record(record)
