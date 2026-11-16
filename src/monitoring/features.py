"""
Per-window ECG summary features for monitoring
----------------------------------------------
A raw window (3600 x 2 samples) is too high-dimensional to compare between
the training data and live traffic directly, so each window is reduced to a
few interpretable statistics. The same function runs inside the served model
(to log live traffic) and on the held-out test split (to build the reference
profile), so both sides are computed identically.

IEC 62304 Traceability
    Software Unit  : SU-MON-01
    Requirement Ref: SRS-MON-001 (Post-deployment input monitoring)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

SAMPLING_RATE_HZ = 360

# Per lead: level, spread, amplitude and high-frequency content, plus heart rate.
#   *_mean_mv  : baseline offset (shifts with electrode/DC problems)
#   *_std_mv   : overall signal energy
#   *_ptp_mv   : peak-to-peak amplitude (shifts with amplifier gain)
#   *_noise_mv : std of the sample-to-sample difference; rises with muscle
#                noise or mains hum, which add fast changes the ECG doesn't have
FEATURE_COLUMNS = [
    f"lead{lead}_{stat}_mv"
    for lead in (1, 2)
    for stat in ("mean", "std", "ptp", "noise")
] + ["heart_rate_bpm"]


def estimate_heart_rate(lead: np.ndarray, fs: int = SAMPLING_RATE_HZ) -> float:
    """Rough heart rate (beats/min) of one lead, from the median R-R interval.

    QRS complexes are the steepest part of the ECG, so the smoothed absolute
    slope peaks once per beat whatever the lead's polarity or baseline. Good
    enough to notice a population shift; not a clinical measurement.
    Returns 0.0 if fewer than two beats are found.
    """
    slope = np.abs(np.diff(lead))
    width = int(0.1 * fs)  # ~QRS duration
    envelope = np.convolve(slope, np.ones(width) / width, mode="same")
    # Most of the window is between beats, so the median is the noise floor.
    floor, top = np.median(envelope), envelope.max()
    if top <= floor:
        return 0.0
    # Beats at least 250 ms apart (<= 240 bpm), rising at least 40% of the way
    # from the noise floor to the steepest beat.
    peaks, _ = find_peaks(
        envelope, height=floor + 0.4 * (top - floor), distance=int(0.25 * fs)
    )
    if len(peaks) < 2:
        return 0.0
    return float(60.0 * fs / np.median(np.diff(peaks)))


def compute_features(signals: np.ndarray) -> dict[str, np.ndarray]:
    """Summarise ECG windows, shape (N, samples, 2) in mV.

    Returns
    -------
    features : dict[str, np.ndarray]
        One array of N values per name in ``FEATURE_COLUMNS``, in that order.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 3 or signals.shape[2] != 2:
        raise ValueError(f"Expected windows of shape (N, samples, 2), got {signals.shape}")

    features: dict[str, np.ndarray] = {}
    for lead in range(2):
        x = signals[:, :, lead]
        prefix = f"lead{lead + 1}"
        features[f"{prefix}_mean_mv"] = x.mean(axis=1)
        features[f"{prefix}_std_mv"] = x.std(axis=1)
        features[f"{prefix}_ptp_mv"] = np.ptp(x, axis=1)
        features[f"{prefix}_noise_mv"] = np.diff(x, axis=1).std(axis=1)
    features["heart_rate_bpm"] = np.array(
        [estimate_heart_rate(window[:, 0]) for window in signals], dtype=np.float64
    )
    return features
