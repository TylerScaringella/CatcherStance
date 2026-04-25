from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from catcher_detection import detect_catcher_from_res_item
from curator.features import cfg
from .yolo import load_yolo_once

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
    model = load_yolo_once()
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
