"""Shared MLflow Model Registry conventions.

Every newly trained model is registered with the ``challenger`` alias.
The promotion gate moves ``champion`` to it only if it meets the
acceptance thresholds, so training alone never changes what downstream
consumers load (``models:/<name>@champion``).

Where runs and the registry live is decided by ``resolve_tracking_uri``: the
``MLFLOW_TRACKING_URI`` environment variable if set (e.g. the shared server in
deploy/mlflow), otherwise ``mlflow.tracking_uri`` from the config.
"""

from __future__ import annotations

import os
import sys

CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow_tracking.db"


def resolve_tracking_uri(config: dict) -> str:
    """MLFLOW_TRACKING_URI if set, else the config's mlflow.tracking_uri, else local SQLite.

    The environment wins so one config works both locally (SQLite) and against a
    shared tracking server, without editing the file.
    """
    return (
        os.environ.get("MLFLOW_TRACKING_URI")
        or config.get("mlflow", {}).get("tracking_uri")
        or DEFAULT_TRACKING_URI
    )


def tolerate_unprintable_output() -> None:
    """Don't crash when printing characters the console encoding can't show.

    With a tracking server, MLflow prints "🏃 View run ... at: <url>" when a run
    ends. On Windows, when output is piped (e.g. under ``dvc repro``), stdout uses
    the legacy code page, which has no emoji, and the print raises
    UnicodeEncodeError. Replace such characters instead.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def is_remote_tracking_uri(uri: str) -> bool:
    """True for a tracking server (http/https), as opposed to a local store."""
    return uri.startswith(("http://", "https://"))
