import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"
FIGURES = ROOT / "reports/figures"
TABLES = ROOT / "reports/tables"

def load_config():
    with open(CONFIGS / "default.json") as f:
        return json.load(f)

def require(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return Path(path)