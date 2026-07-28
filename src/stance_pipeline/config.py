from typing import Callable

from project_paths import CLASSIFIER_PATH, LABEL_ENCODER_PATH, RUNS_DIR, SCALER_PATH

StatusCallback = Callable[[str, int | None, int | None], None]
