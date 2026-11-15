"""Shared MLflow Model Registry conventions.

Every newly trained model is registered with the ``challenger`` alias.
The promotion gate moves ``champion`` to it only if it meets the
acceptance thresholds, so training alone never changes what downstream
consumers load (``models:/<name>@champion``).
"""

CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
