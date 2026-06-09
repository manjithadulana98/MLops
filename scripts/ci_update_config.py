"""Update training_config.yaml with values from environment variables."""
import os

import yaml

with open("configs/training_config.yaml") as f:
    cfg = yaml.safe_load(f)

cfg["dataset"]["sample_fraction"] = float(os.environ["SAMPLE_FRACTION"])
cfg["training"]["epochs"] = int(os.environ["EPOCHS"])

with open("configs/training_config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(
    f"Config updated: sample_fraction={cfg['dataset']['sample_fraction']}, "
    f"epochs={cfg['training']['epochs']}"
)
