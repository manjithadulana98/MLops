"""
ECG Signal Validation Module
-----------------------------
Validates raw ECG signals against clinical acceptance criteria prior to
ingestion into the VT-detection training pipeline.

IEC 62304 Traceability
    Software Unit  : SU-INGEST-01
    Requirement Ref: SRS-DATA-001, SRS-DATA-002, SRS-DATA-003
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ClinicalDataError(Exception):
    """Raised when an ECG signal fails one or more clinical acceptance checks.

    Attributes
    ----------
    check_name : str
        Identifier of the failed validation rule (e.g. 'SIGNAL_LENGTH').
    detail : str
        Human-readable description of the failure.
    """

    def __init__(self, check_name: str, detail: str) -> None:
        self.check_name = check_name
        self.detail = detail
        super().__init__(f"[{check_name}] {detail}")


# ---------------------------------------------------------------------------
# Validation constants  (override via keyword arguments where appropriate)
# ---------------------------------------------------------------------------

DEFAULT_SAMPLING_RATE_HZ: int = 360          # MIT-BIH standard
DEFAULT_DURATION_SECONDS: float = 10.0       # expected recording window
DEFAULT_VOLTAGE_MIN_MV: float = -5.0         # lower physiological bound (mV)
DEFAULT_VOLTAGE_MAX_MV: float = 5.0          # upper physiological bound (mV)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_ecg_signal(
    signal: np.ndarray,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    voltage_min_mv: float = DEFAULT_VOLTAGE_MIN_MV,
    voltage_max_mv: float = DEFAULT_VOLTAGE_MAX_MV,
    lead_label: Optional[str] = None,
) -> None:
    """Validate a single ECG lead signal against clinical acceptance criteria.

    Performs the following checks in order:

    1. **Array type** – signal must be a NumPy array.
    2. **NaN / Infinity** – no non-finite values permitted.
    3. **Signal length** – sample count must match
       ``sampling_rate_hz * duration_seconds`` exactly.
    4. **Voltage range** – all samples must lie within
       [``voltage_min_mv``, ``voltage_max_mv``].

    Parameters
    ----------
    signal : np.ndarray
        1-D array of voltage samples (mV).
    sampling_rate_hz : int
        Expected sampling frequency in Hz.  Default: 360 Hz (MIT-BIH).
    duration_seconds : float
        Expected recording duration in seconds.  Default: 10 s.
    voltage_min_mv : float
        Lower bound of acceptable voltage (mV).  Default: -5.0 mV.
    voltage_max_mv : float
        Upper bound of acceptable voltage (mV).  Default: +5.0 mV.
    lead_label : str, optional
        Human-readable lead identifier (e.g. ``"MLII"``) used in error
        messages to aid traceability.

    Raises
    ------
    TypeError
        If *signal* is not a ``np.ndarray``.
    ClinicalDataError
        If any clinical acceptance criterion is violated.

    Examples
    --------
    >>> import numpy as np
    >>> from src.ingestion.validate import validate_ecg_signal
    >>> good_signal = np.zeros(3600)          # 10 s @ 360 Hz, all 0 mV
    >>> validate_ecg_signal(good_signal)      # no exception → passes
    """
    context = f" (lead: {lead_label})" if lead_label else ""

    # ------------------------------------------------------------------
    # Check 1 – Input type guard
    # ------------------------------------------------------------------
    if not isinstance(signal, np.ndarray):
        raise TypeError(
            f"ECG signal must be a numpy.ndarray, got {type(signal).__name__}{context}."
        )

    # ------------------------------------------------------------------
    # Check 2 – Non-finite values (NaN / ±Infinity)
    # ------------------------------------------------------------------
    if not np.all(np.isfinite(signal)):
        nan_count = int(np.sum(np.isnan(signal)))
        inf_count = int(np.sum(np.isinf(signal)))
        raise ClinicalDataError(
            check_name="NON_FINITE_VALUES",
            detail=(
                f"Signal contains {nan_count} NaN(s) and {inf_count} "
                f"infinite value(s){context}. "
                "Non-finite samples indicate sensor dropout or ADC overflow."
            ),
        )

    # ------------------------------------------------------------------
    # Check 3 – Signal length consistency
    # ------------------------------------------------------------------
    expected_samples = int(sampling_rate_hz * duration_seconds)
    actual_samples = signal.shape[0]

    if actual_samples != expected_samples:
        raise ClinicalDataError(
            check_name="SIGNAL_LENGTH",
            detail=(
                f"Expected {expected_samples} samples "
                f"({duration_seconds} s × {sampling_rate_hz} Hz) "
                f"but received {actual_samples} samples{context}. "
                "Truncated or padded recordings are not accepted."
            ),
        )

    # ------------------------------------------------------------------
    # Check 4 – Physiological voltage range
    # ------------------------------------------------------------------
    signal_min = float(signal.min())
    signal_max = float(signal.max())

    if signal_min < voltage_min_mv or signal_max > voltage_max_mv:
        raise ClinicalDataError(
            check_name="VOLTAGE_RANGE",
            detail=(
                f"Signal voltage [{signal_min:.4f} mV, {signal_max:.4f} mV] "
                f"exceeds the acceptable range "
                f"[{voltage_min_mv} mV, {voltage_max_mv} mV]{context}. "
                "Values outside this range suggest electrode artifact or "
                "incorrect ADC gain calibration."
            ),
        )


def validate_ecg_record(
    record: dict[str, np.ndarray],
    **kwargs,
) -> None:
    """Validate all leads in a multi-lead ECG record.

    Parameters
    ----------
    record : dict[str, np.ndarray]
        Mapping of lead label → signal array
        (e.g. ``{"MLII": array(...), "V5": array(...)})``).
    **kwargs
        Forwarded to :func:`validate_ecg_signal` for each lead.

    Raises
    ------
    ClinicalDataError
        On the first lead that fails validation.

    Examples
    --------
    >>> record = {"MLII": np.zeros(3600), "V5": np.zeros(3600)}
    >>> validate_ecg_record(record)
    """
    for lead_label, signal in record.items():
        validate_ecg_signal(signal, lead_label=lead_label, **kwargs)
