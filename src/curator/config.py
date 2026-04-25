from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELED_VIDEO_PATH = PROJECT_ROOT / "data/labeled_videos/labeled-videos-2026-04-22-00-05-e3d836a2.csv"
DATASET_OUTPUT_PATH = PROJECT_ROOT / "data/dataset/keypoints-labeled.csv"
MODEL_PATH = PROJECT_ROOT / "notebooks/yolo26n-pose.pt"

DESIRED_COLUMNS = ["id", "choice", "video"]
NUM_FRAMES = 7
CSV_FIELDNAMES = ["choice", "id", "features", "status"]
DEFAULT_NUM_WORKERS = 2
