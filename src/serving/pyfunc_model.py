"""
MLflow pyfunc wrapper: raw 2-lead ECG in, VT/PVC probability out
-----------------------------------------------------------------
Packages the trained Keras network together with its preprocessing
(ECG -> 224x224x3 pseudo-image, see VGG16ECGModel) and the clinical input
validation, so every registered model version is self-contained:

    model = mlflow.pyfunc.load_model("models:/vt-detector@champion")
    probabilities = model.predict(signals)   # signals: (N, 3600, 2) float32, mV

and can be served as a REST API without any repo code:

    mlflow models serve -m "models:/vt-detector@champion" -p 5001 --env-manager local

Set VT_PREDICTION_LOG_DIR in the server's environment to log every request
for drift monitoring (see src/monitoring/prediction_log.py).

IEC 62304 Traceability
    Software Unit  : SU-SERVE-01
    Requirement Ref: SRS-SERVE-001 (Model packaging & inference input validation)
"""

from __future__ import annotations

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

# Name of the logged artifact holding the .keras file.
KERAS_ARTIFACT = "keras_model"
# 10 s of 2-lead ECG at 360 Hz.
INPUT_SHAPE = (3600, 2)


class VTDetectorModel(mlflow.pyfunc.PythonModel):
    """Validates raw ECG windows, converts them to images and runs the network."""

    def load_context(self, context) -> None:
        # Imported here so they resolve from the model's bundled code
        # (logged with code_paths) when loaded outside the repo.
        from src.monitoring.prediction_log import PredictionLogger
        from src.training.model import VGG16ECGModel

        self._model = VGG16ECGModel()
        self._model.load(context.artifacts[KERAS_ARTIFACT])
        # The training run that produced this version, so logged predictions
        # can be traced back to the model that made them.
        model_config = getattr(context, "model_config", None) or {}
        self._prediction_log = PredictionLogger.from_env(model_run_id=model_config.get("run_id"))

    def predict(self, context, model_input, params=None):
        """Return P(VT/PVC) for each 10-second window, shape (N,).

        Invalid input raises MlflowException(INVALID_PARAMETER_VALUE), which the
        MLflow scoring server returns as HTTP 400 with this message (any other
        exception type becomes a generic error with a server stack trace).
        """
        from src.ingestion.validate import ClinicalDataError, validate_ecg_signal

        prediction_log = getattr(self, "_prediction_log", None)
        signals = np.asarray(model_input, dtype=np.float32)
        # Normally unreachable through mlflow.pyfunc / the scoring server, whose
        # signature check rejects other shapes first; kept for direct callers.
        if signals.ndim != 3 or signals.shape[1:] != INPUT_SHAPE:
            if prediction_log:
                prediction_log.log_rejected(signals.shape[0] if signals.ndim else 1, "SHAPE")
            raise MlflowException(
                f"Expected ECG windows of shape (N, {INPUT_SHAPE[0]}, {INPUT_SHAPE[1]}) "
                f"(10 s of 2-lead ECG at 360 Hz, in mV), got {signals.shape}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        # Reject physiologically implausible input instead of scoring it.
        try:
            for i, window in enumerate(signals):
                for lead in range(INPUT_SHAPE[1]):
                    validate_ecg_signal(
                        window[:, lead], lead_label=f"window {i}, lead {lead + 1}"
                    )
        except ClinicalDataError as err:
            if prediction_log:
                prediction_log.log_rejected(len(signals), err.check_name)
            raise MlflowException(
                f"Input rejected by clinical validation: {err}",
                error_code=INVALID_PARAMETER_VALUE,
            ) from err

        probabilities = self._model.predict(signals)
        if prediction_log:
            prediction_log.log_scored(signals, probabilities)
        return probabilities
