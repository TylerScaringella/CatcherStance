import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curator.config import DEFAULT_NUM_WORKERS, MODEL_PATH
from curator.dataset import generate_dataset
from curator.features import cfg, init_worker, load_yolo_once, normalize_keypoints, pad_last, process_video

__all__ = [
    "MODEL_PATH",
    "cfg",
    "generate_dataset",
    "init_worker",
    "load_yolo_once",
    "normalize_keypoints",
    "pad_last",
    "process_video",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a resumable catcher keypoint dataset from labeled videos."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker processes to use. Keep this small on a laptop.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(num_workers=args.workers)
