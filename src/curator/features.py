from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from catcher_detection import CatcherDetectionConfig, detect_catcher_from_res_item
from .config import MODEL_PATH, NUM_FRAMES

_worker_model = None

cfg = CatcherDetectionConfig(
    search_roi_norm=(0.30, 0.42, 0.70, 0.98),
    frame_gate_roi_norm=(0.24, 0.34, 0.76, 1.00),
    anchor_norm=(0.50, 0.78),
    invalid_zones_norm=[
        (0.00, 0.00, 0.18, 1.00),
        (0.82, 0.00, 1.00, 1.00),
        (0.00, 0.00, 1.00, 0.18),
    ],
    max_anchor_dist_norm=0.30,
    max_final_anchor_dist_norm=0.30,
    min_score=0.48,
    min_margin=0,
    min_aspect_ratio=0.7,
    max_compactness=2.30,
)

LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def pad_last(items, target_len):
    if len(items) == 0:
        return None
    while len(items) < target_len:
        items.append(items[-1].copy() if hasattr(items[-1], "copy") else items[-1])
    return items


def normalize_keypoints(kpts):
    kpts = kpts.astype(np.float32).copy()
    hip_center = (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2.0
    shoulder_center = (kpts[LEFT_SHOULDER] + kpts[RIGHT_SHOULDER]) / 2.0
    scale = np.linalg.norm(shoulder_center - hip_center)

    if scale < 1e-6:
        y_min, x_min = np.min(kpts[:, 1]), np.min(kpts[:, 0])
        y_max, x_max = np.max(kpts[:, 1]), np.max(kpts[:, 0])
        scale = max(y_max - y_min, x_max - x_min, 1.0)

    return (kpts - hip_center) / scale


def init_worker(model_path):
    global _worker_model
    _worker_model = YOLO(str(model_path))


def load_yolo_once():
    global _worker_model
    if _worker_model is None:
        _worker_model = YOLO(str(MODEL_PATH))
    return _worker_model


def process_video(video):
    if _worker_model is None:
        raise RuntimeError("Worker YOLO model has not been initialized.")

    catcher_detections = []
    results = _worker_model(
        video,
        show=False,
        stream=True,
        save=False,
        verbose=False,
        imgsz=512,
        vid_stride=2,
    )

    for result in results:
        detected_catcher = detect_catcher_from_res_item(result, cfg=cfg)
        if detected_catcher is None:
            continue

        keypoints = normalize_keypoints(detected_catcher["keypoints"])
        catcher_detections.append(keypoints)

    if len(catcher_detections) == 0:
        return None

    selected = pad_last(catcher_detections[-NUM_FRAMES:], NUM_FRAMES)
    return np.concatenate([frame.reshape(-1) for frame in selected], axis=0)
