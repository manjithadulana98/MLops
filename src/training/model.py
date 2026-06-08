"""
VGG16 Transfer Learning Model for ECG Arrhythmia Detection
-----------------------------------------------------------
Adapts pre-trained VGG16 (ImageNet weights) for 1D ECG signal classification.

IEC 62304 Traceability
    Software Unit  : SU-TRAIN-01
    Requirement Ref: SRS-MODEL-001 (Model architecture & transfer learning)

Design Notes
    - Input: 2-lead ECG, 10 seconds @ 360 Hz → (3600, 2)
    - Reshaping strategy: duplicate leads 3x to form (3600, 6) → treat as
      grayscale image with channel dim for Conv1D compatibility
    - VGG16 frozen base layers: leverage ImageNet feature extraction
    - Top layers: dense classification head with dropout regularization
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logger = logging.getLogger(__name__)


class VGG16ECGModel:
    """VGG16-based transfer learning model for ECG classification.

    The model converts 2-lead ECG signals into a pseudo-image representation
    suitable for convolutional processing, leveraging pre-trained VGG16.

    Attributes
    ----------
    input_shape : tuple
        Expected input shape (3600, 2) for time-series ECG.
    model : tf.keras.Model
        Compiled Keras model instance.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (3600, 2),
        dropout_rate: float = 0.5,
        learning_rate: float = 1e-3,
    ) -> None:
        """Initialize VGG16 ECG model with transfer learning.

        Parameters
        ----------
        input_shape : tuple
            Single sample shape (3600, 2) for 10s of 2-lead ECG.
        dropout_rate : float
            Dropout probability for regularization. Default: 0.5.
        learning_rate : float
            Adam optimizer learning rate. Default: 1e-3.
        """
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

        logger.info(
            f"Initialized VGG16ECGModel: input_shape={input_shape}, "
            f"dropout={dropout_rate}, lr={learning_rate}"
        )

    def _prepare_for_vgg16(self, signals: np.ndarray) -> np.ndarray:
        """Convert (N, 3600, 2) → (N, 224, 224, 3) for VGG16 input.

        Strategy:
        1. Resize ECG to VGG's native input (224×224).
        2. Stack the 2 leads 3 times to form a "pseudo-RGB" image.
           (Real images have 3 channels; we replicate to match.)
        3. Normalize to [0, 1] range.

        Parameters
        ----------
        signals : np.ndarray
            Shape (N, 3600, 2) — batch of 2-lead signals.

        Returns
        -------
        images : np.ndarray
            Shape (N, 224, 224, 3) ready for VGG16 input.
        """
        N = signals.shape[0]
        images = np.zeros((N, 224, 224, 3), dtype=np.float32)

        for i in range(N):
            # Normalize each lead independently
            sig = signals[i]  # shape (3600, 2)
            sig_norm = (sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)

            # Resize to 224×224 using bilinear interpolation
            # Treat (3600, 2) as a 2-channel image
            sig_resized = tf.image.resize(
                tf.expand_dims(sig_norm, -1),  # (3600, 2, 1)
                [224, 224],
                method="bilinear",
            )
            sig_resized = tf.squeeze(sig_resized, axis=-1)  # (224, 224, 2)

            # Stack 2 leads 1.5x to fill 3 channels: [L1, L2, L1]
            images[i, :, :, 0] = sig_resized[:, :, 0]  # Lead 1
            images[i, :, :, 1] = sig_resized[:, :, 1]  # Lead 2
            images[i, :, :, 2] = sig_resized[:, :, 0]  # Lead 1 (repeated)

        return images

    def _signal_to_vgg16_image_tf(self, signal: tf.Tensor) -> tf.Tensor:
        """Convert one ECG sample (3600, 2) to one VGG16 image (224, 224, 3)."""
        signal = tf.cast(signal, tf.float32)

        # Normalize each lead independently.
        lead1 = signal[:, 0:1]
        lead2 = signal[:, 1:2]

        lead1 = (lead1 - tf.reduce_mean(lead1)) / (tf.math.reduce_std(lead1) + 1e-8)
        lead2 = (lead2 - tf.reduce_mean(lead2)) / (tf.math.reduce_std(lead2) + 1e-8)

        # Resize each lead to image space, then compose pseudo-RGB: [L1, L2, L1].
        lead1_img = tf.image.resize(tf.expand_dims(lead1, axis=-1), [224, 224], method="bilinear")
        lead2_img = tf.image.resize(tf.expand_dims(lead2, axis=-1), [224, 224], method="bilinear")

        return tf.concat([lead1_img, lead2_img, lead1_img], axis=-1)

    def _make_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> tf.data.Dataset:
        """Build a memory-safe tf.data pipeline with on-the-fly image conversion."""
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(min(int(X.shape[0]), 10000), seed=42, reshuffle_each_iteration=True)

        ds = ds.map(
            lambda sig, lbl: (self._signal_to_vgg16_image_tf(sig), tf.cast(lbl, tf.float32)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def build(self) -> Model:
        """Build the VGG16 transfer learning model.

        Architecture:
        1. Input: (224, 224, 3)
        2. VGG16 backbone (ImageNet pre-trained) with frozen weights
        3. Global Average Pooling → (512,)
        4. Dense(256) + Dropout + Dense(64) + Dropout
        5. Output Dense(1) + Sigmoid for binary classification

        Returns
        -------
        model : tf.keras.Model
            Compiled model ready for training.
        """
        # Load pre-trained VGG16 (ImageNet weights)
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )

        # Freeze base model weights
        base_model.trainable = False
        logger.info("VGG16 base loaded (ImageNet pre-trained, frozen)")

        # Build custom head
        model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation="relu"),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation="sigmoid"),  # binary classification
        ])

        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        self.model = model
        logger.info("Model compiled successfully")
        return model

    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model is None:
            raise ValueError("Model not built yet. Call .build() first.")
        self.model.summary()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        epochs: int = 20,
        checkpoint_path: str = "models/artifacts/vgg16_best.keras",
        class_weight: dict | None = None,
        save_best_checkpoint: bool = True,
        early_stopping_patience: int = 5,
    ) -> dict:
        """Train the model with validation and early stopping.

        Parameters
        ----------
        X_train : np.ndarray
            Training signals (N, 3600, 2).
        y_train : np.ndarray
            Training labels (N,).
        X_val : np.ndarray
            Validation signals (M, 3600, 2).
        y_val : np.ndarray
            Validation labels (M,).
        batch_size : int
            Batch size for training. Default: 32.
        epochs : int
            Maximum epochs for training. Default: 20.
        checkpoint_path : str
            Path to save best model checkpoint.
        class_weight : dict | None
            Optional class weighting for imbalanced datasets.
        save_best_checkpoint : bool
            Whether to save a best-model checkpoint at each epoch.
        early_stopping_patience : int
            Number of epochs without improvement before early stopping.

        Returns
        -------
        history : dict
            Training history containing loss, accuracy, etc.
        """
        if self.model is None:
            raise ValueError("Model not built. Call .build() first.")

        logger.info(
            f"Preparing data pipeline: {X_train.shape[0]} train, {X_val.shape[0]} val"
        )

        train_ds = self._make_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_ds = self._make_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        if save_best_checkpoint:
            callbacks.insert(
                0,
                ModelCheckpoint(
                    checkpoint_path,
                    monitor="val_auc",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                ),
            )

        # Train
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )

        logger.info(f"Training complete. Best checkpoint saved: {checkpoint_path}")
        return history.history

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
    ) -> dict:
        """Evaluate model on test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test signals (N, 3600, 2).
        y_test : np.ndarray
            Test labels (N,).

        Returns
        -------
        metrics : dict
            Evaluation metrics (loss, accuracy, auc, etc.).
        """
        if self.model is None:
            raise ValueError("Model not built. Call .build() first.")

        test_ds = self._make_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)
        logger.info(f"Evaluating on {X_test.shape[0]} test samples")

        results = self.model.evaluate(test_ds, verbose=0)
        metrics = {
            name: float(val)
            for name, val in zip(self.model.metrics_names, results)
        }

        logger.info(f"Test metrics: {metrics}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions on input signals.

        Parameters
        ----------
        X : np.ndarray
            Signals with shape (N, 3600, 2).

        Returns
        -------
        predictions : np.ndarray
            Probabilities (N,) with values in [0, 1].
        """
        if self.model is None:
            raise ValueError("Model not built. Call .build() first.")

        # Dummy labels are required to reuse the same tf.data conversion path.
        dummy_y = np.zeros((X.shape[0],), dtype=np.uint8)
        pred_ds = self._make_dataset(X, dummy_y, batch_size=32, shuffle=False).map(
            lambda img, _: img,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return self.model.predict(pred_ds, verbose=0).flatten()

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File path (e.g., 'models/vgg16_ecg.keras').
        """
        if self.model is None:
            raise ValueError("Model not built. Call .build() first.")
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load pre-trained model from disk.

        Parameters
        ----------
        path : str
            File path (e.g., 'models/vgg16_ecg.keras').
        """
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")
