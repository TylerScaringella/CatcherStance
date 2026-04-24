from __future__ import annotations

import csv
import json
import time
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import joblib
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curator.video_to_keypoint import MODEL_PATH, process_video
from downloader.main import run_download_pipeline
from notebook.catcher_detection import detect_catcher_from_res_item

RUNS_DIR = PROJECT_ROOT / "data" / "runs"
CLASSIFIER_PATH = PROJECT_ROOT / "model" / "catcher_stance_mlp.pt"
LABEL_ENCODER_PATH = PROJECT_ROOT / "model" / "label_encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "model" / "standard_scaler.pkl"
StatusCallback = Callable[[str, int | None, int | None], None]
COCO_SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class CatcherMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class PitchDetection:
    pitch_index: int
    clip_id: str
    video_path: str
    stance: str
    confidence: float
    status: str
    error: str = ""


@dataclass
class PitchFeature:
    choice: str
    id: str
    features: list[float] | str
    status: str


class StanceClassifier:
    def __init__(self):
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.model = CatcherMLP(
            input_dim=int(getattr(self.scaler, "n_features_in_", 238)),
            num_classes=len(self.label_encoder.classes_),
        )
        state = torch.load(CLASSIFIER_PATH, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        scaled = self.scaler.transform(np.asarray([features], dtype=np.float32))
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return str(self.label_encoder.inverse_transform([idx])[0]), float(probs[idx])


def _load_yolo_once():
    from curator import video_to_keypoint

    if video_to_keypoint._worker_model is None:
        video_to_keypoint._worker_model = YOLO(str(MODEL_PATH))
    return video_to_keypoint._worker_model


def draw_catcher_overlay(frame: np.ndarray, detection: dict | None, label: str = ""):
    if detection is None:
        cv2.putText(
            frame,
            "No catcher detected",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    box = detection.get("box")
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 180, 90), 2)

    keypoints = detection.get("keypoints")
    if keypoints is not None:
        points = np.asarray(keypoints, dtype=float)
        for a, b in COCO_SKELETON:
            if a >= len(points) or b >= len(points):
                continue
            pa = points[a]
            pb = points[b]
            if np.any(~np.isfinite(pa)) or np.any(~np.isfinite(pb)):
                continue
            if (pa == 0).all() or (pb == 0).all():
                continue
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                (40, 170, 255),
                2,
            )

        for point in points:
            if np.any(~np.isfinite(point)) or (point == 0).all():
                continue
            center = (int(point[0]), int(point[1]))
            cv2.circle(frame, center, 4, (255, 255, 255), -1)
            cv2.circle(frame, center, 4, (30, 110, 255), 1)

    text = label or f"catcher score {detection.get('score', 0):.2f}"
    cv2.putText(
        frame,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 90, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def overlay_mjpeg_frames(video_path: Path, pitch_label: str = ""):
    from curator.video_to_keypoint import cfg

    model = _load_yolo_once()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = min(1 / fps, 0.05)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(
                frame,
                show=False,
                stream=False,
                save=False,
                verbose=False,
                imgsz=512,
            )[0]
            detection = detect_catcher_from_res_item(result, cfg=cfg)
            draw_catcher_overlay(frame, detection, label=pitch_label)

            encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            time.sleep(frame_delay)
    finally:
        cap.release()


def read_manifest_rows(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def downloaded_video_rows(manifest_path: Path) -> list[dict]:
    rows = []
    for row in read_manifest_rows(manifest_path):
        path = row.get("saved_path", "")
        if row.get("status") == "downloaded" and path and Path(path).exists():
            rows.append(row)
    return rows


def detect_stances_for_manifest(
    manifest_path: Path,
    status_callback: StatusCallback | None = None,
) -> tuple[list[PitchDetection], list[PitchFeature]]:
    _load_yolo_once()
    classifier = StanceClassifier()
    detections: list[PitchDetection] = []
    feature_rows: list[PitchFeature] = []
    video_rows = downloaded_video_rows(manifest_path)
    total = len(video_rows)

    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", 0, total)

    progress = tqdm(video_rows, desc="Catcher detection", unit="pitch", total=total)
    for idx, row in enumerate(progress, start=1):
        clip_id = row.get("clip_id") or Path(row.get("saved_path", "")).stem
        video_path = row.get("saved_path", "")
        try:
            features = process_video(video_path)
            if features is None:
                feature_rows.append(PitchFeature("", clip_id, "", "no_valid_catcher"))
                detections.append(
                    PitchDetection(idx, clip_id, video_path, "", 0.0, "no_valid_catcher")
                )
                continue

            feature_rows.append(PitchFeature("", clip_id, features.tolist(), "ok"))
            stance, confidence = classifier.predict(features)
            detections.append(
                PitchDetection(idx, clip_id, video_path, stance, confidence, "ok")
            )
        except Exception as exc:
            feature_rows.append(PitchFeature("", clip_id, "", f"error:{type(exc).__name__}"))
            detections.append(
                PitchDetection(
                    idx,
                    clip_id,
                    video_path,
                    "",
                    0.0,
                    f"error:{type(exc).__name__}",
                    str(exc),
                )
            )

        if status_callback is not None:
            status_callback(f"Processed {idx} of {total} pitches", idx, total)

    return detections, feature_rows


def write_detection_outputs(
    run_dir: Path,
    detections: Iterable[PitchDetection],
    feature_rows: Iterable[PitchFeature],
) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in detections]
    features = [asdict(item) for item in feature_rows]

    json_path = run_dir / "detections.json"
    csv_path = run_dir / "detections.csv"
    feature_csv_path = run_dir / "pitch_features.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(PitchDetection.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(feature_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(PitchFeature.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)

    return rows


def run_game_detection(
    run_id: str,
    start_url: str,
    headless: bool = False,
    download_workers: int = 8,
    status_callback: StatusCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    download_dir = run_dir / "downloads"
    manifest_path = run_dir / "video_manifest.csv"
    storage_state_path = PROJECT_ROOT / "downloader" / "playwright_state.json"

    run_download_pipeline(
        start_url=start_url,
        download_dir=str(download_dir),
        manifest_path=str(manifest_path),
        storage_state_path=str(storage_state_path),
        headless=headless,
        download_workers=download_workers,
    )
    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", None, None)

    detections, feature_rows = detect_stances_for_manifest(
        manifest_path,
        status_callback=status_callback,
    )
    return write_detection_outputs(run_dir, detections, feature_rows)


def run_detection_for_existing_run(
    run_id: str,
    status_callback: StatusCallback | None = None,
) -> list[dict]:
    run_dir = RUNS_DIR / run_id
    manifest_path = run_dir / "video_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for run: {run_id}")

    if status_callback is not None:
        status_callback("Running catcher detection and stance classifier", None, None)

    detections, feature_rows = detect_stances_for_manifest(
        manifest_path,
        status_callback=status_callback,
    )
    return write_detection_outputs(run_dir, detections, feature_rows)
