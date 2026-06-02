"""
MIT-BIH Arrhythmia Dataset Loader
----------------------------------
Loads, preprocesses, and caches the MIT-BIH Arrhythmia Database for model training.

IEC 62304 Traceability
    Software Unit  : SU-INGEST-02
    Requirement Ref: SRS-DATA-004 (MIT-BIH Gold Standard loading)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import wfdb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# MIT-BIH configuration
MIT_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# VT-specific annotation codes (based on AAMI standard)
VT_ANNOTATIONS = {"V", "[", "!", "F"}  # V=PVC, [=start PVC, !=fusion, F=fusion/normal

SAMPLING_RATE_HZ = 360
SIGNAL_DURATION_SECONDS = 10.0
EXPECTED_SAMPLES = int(SAMPLING_RATE_HZ * SIGNAL_DURATION_SECONDS)


class MITDatasetLoader:
    """Loader for MIT-BIH Arrhythmia Database with preprocessing & caching.

    Attributes
    ----------
    dataset_path : Path
        Root directory containing MIT record files (*.hea, *.dat).
    cache_dir : Path
        Where to store preprocessed .npy arrays.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        cache_dir: str | Path = "data/processed/mit_cache",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"MIT dataset path not found: {self.dataset_path}")
        logger.info(f"Initialized MITDatasetLoader: {self.dataset_path}")

    def load_record(self, record_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single MIT record and return (two_lead_signals, annotations).

        Parameters
        ----------
        record_id : str
            Record identifier (e.g., "100", "101").

        Returns
        -------
        signals : np.ndarray
            Shape (2, 108000) — two leads, 360 Hz × 5 min.
        sample_indices : np.ndarray
            Beat annotation indices.
        beat_types : np.ndarray
            Beat category codes.

        Raises
        ------
        FileNotFoundError
            If record file not found.
        """
        record_path = self.dataset_path / record_id
        record = wfdb.rdrecord(str(record_path), physical=True)
        annotation = wfdb.rdann(str(record_path), "atr", physical=True)

        signals = record.p_signal  # shape (N, 2) for MIT-BIH
        return signals, annotation.sample, annotation.symbol

    def segment_signals(
        self,
        signals: np.ndarray,
        sample_indices: np.ndarray,
        beat_types: np.ndarray,
    ) -> list[Tuple[np.ndarray, int]]:
        """Split long signals into 10-second windows around beat annotations.

        Parameters
        ----------
        signals : np.ndarray
            Shape (N, 2) ECG signal.
        sample_indices : np.ndarray
            Annotation sample indices.
        beat_types : np.ndarray
            Annotation symbols.

        Returns
        -------
        segments : list[tuple[np.ndarray, int]]
            List of tuples (segment_array, label).
            segment_array: shape (10*360, 2) — 10 seconds of two leads
            label: 1 if segment contains VT/PVC, else 0.
        """
        segments = []
        total_samples = signals.shape[0]
        half_window = EXPECTED_SAMPLES // 2

        for idx, beat_sample, beat_type in zip(
            range(len(sample_indices)), sample_indices, beat_types
        ):
            # Compute window bounds centered on beat
            start = max(0, beat_sample - half_window)
            end = min(total_samples, start + EXPECTED_SAMPLES)

            # Skip if window is incomplete
            if end - start < EXPECTED_SAMPLES:
                continue

            # Extract segment
            segment = signals[start:end, :]  # shape (3600, 2)
            label = 1 if beat_type in VT_ANNOTATIONS else 0

            segments.append((segment, label))

        logger.debug(f"Extracted {len(segments)} segments from record")
        return segments

    def load_and_preprocess_all_records(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all MIT records, preprocess, and return train/val/test splits.

        Parameters
        ----------
        test_size : float
            Fraction of records to reserve for testing. Default: 0.2.
        val_size : float
            Fraction of remaining records to reserve for validation. Default: 0.1.
        force_recompute : bool
            If True, recompute from raw files (ignore cache). Default: False.

        Returns
        -------
        X_train, y_train : np.ndarray
            Training signals (N, 3600, 2) and labels (N,).
        X_val, y_val : np.ndarray
            Validation signals and labels.
        X_test, y_test : np.ndarray
            Test signals and labels.
        """
        cache_file = self.cache_dir / "mit_preprocessed.npz"

        if cache_file.exists() and not force_recompute:
            logger.info(f"Loading from cache: {cache_file}")
            data = np.load(cache_file)
            return (
                data["X_train"],
                data["y_train"],
                data["X_val"],
                data["y_val"],
                data["X_test"],
                data["y_test"],
            )

        logger.info(f"Preprocessing MIT dataset from {len(MIT_RECORDS)} records...")
        all_segments = []
        all_labels = []

        for record_id in MIT_RECORDS:
            try:
                signals, indices, types = self.load_record(record_id)
                segments = self.segment_signals(signals, indices, types)

                for segment, label in segments:
                    all_segments.append(segment)
                    all_labels.append(label)
                logger.info(f"✓ Record {record_id}: {len(segments)} segments")

            except Exception as e:
                logger.warning(f"✗ Record {record_id} failed: {e}")
                continue

        X = np.array(all_segments)  # shape (N, 3600, 2)
        y = np.array(all_labels)    # shape (N,)

        logger.info(
            f"Total segments: {X.shape[0]}, "
            f"VT/PVC: {np.sum(y)}, Normal: {np.sum(1 - y)}"
        )

        # Train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Val/train split on remaining
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=y_temp,
        )

        logger.info(
            f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}"
        )

        # Cache
        np.savez(
            cache_file,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        logger.info(f"Cached preprocessed data to {cache_file}")

        return X_train, y_train, X_val, y_val, X_test, y_test
