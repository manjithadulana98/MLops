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
        from src.training.model import VGG16ECGModel

        self._model = VGG16ECGModel()
        self._model.load(context.artifacts[KERAS_ARTIFACT])

    def predict(self, context, model_input, params=None):
        """Return P(VT/PVC) for each 10-second window, shape (N,).

        Invalid input raises MlflowException(INVALID_PARAMETER_VALUE), which the
        MLflow scoring server returns as HTTP 400 with this message (any other
        exception type becomes a generic error with a server stack trace).
        """
        from src.ingestion.validate import ClinicalDataError, validate_ecg_signal

        signals = np.asarray(model_input, dtype=np.float32)
        if signals.ndim != 3 or signals.shape[1:] != INPUT_SHAPE:
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
            raise MlflowException(
                f"Input rejected by clinical validation: {err}",
                error_code=INVALID_PARAMETER_VALUE,
            ) from err

        return self._model.predict(signals)
