from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESEARCH_DIR = ROOT / "research"

WEB_DIR = SRC_DIR / "web"

AUTH_DIR = DATA_DIR / "auth"
RAW_DIR = DATA_DIR / "raw"
RAW_DOWNLOADS_DIR = RAW_DIR / "downloads"
RAW_LABELED_VIDEOS_DIR = RAW_DIR / "labeled_videos"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_DIR = PROCESSED_DIR / "dataset"
RUNS_DIR = DATA_DIR / "runs"
EXAMPLES_DIR = DATA_DIR / "examples"
SCHEDULES_DIR = DATA_DIR / "schedules"
SCHEDULE_PATH = SCHEDULES_DIR / "duke_baseball_2026.json"

CLASSIFIER_DIR = MODELS_DIR / "classifier"
CLASSIFIER_PATH = CLASSIFIER_DIR / "catcher_stance_mlp.pt"
LABEL_ENCODER_PATH = CLASSIFIER_DIR / "label_encoder.pkl"
SCALER_PATH = CLASSIFIER_DIR / "standard_scaler.pkl"
POSE_MODEL_PATH = MODELS_DIR / "pose" / "yolo26n-pose.pt"
PLAYWRIGHT_STATE_PATH = AUTH_DIR / "playwright_state.json"
