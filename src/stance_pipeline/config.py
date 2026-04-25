from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "catcher_stance_mlp.pt"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "standard_scaler.pkl"
StatusCallback = Callable[[str, int | None, int | None], None]
