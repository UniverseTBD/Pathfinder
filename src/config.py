# src/config.py
import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yml")

def load_config(path: str = CONFIG_PATH) -> dict:
    """Loads YAML config from path."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

config = load_config()
print(config)

