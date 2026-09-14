from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"
VALIDATION_JSON = DATA_DIR / "validation.json"
CASES_YAML = ROOT / "cases.yaml"
MODELS_YAML = ROOT / "models.yaml"
RUNS_DIR = ROOT / "runs"


def db_path(db_id: str) -> Path:
    return DATABASE_DIR / db_id / f"{db_id}.sqlite"
