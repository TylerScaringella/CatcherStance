from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from catcher_detection import detect_catcher_from_res_item
from curator.features import cfg as catcher_config
from curator.features import normalize_keypoints
from project_paths import POSE_MODEL_PATH

from .assets import resolve_model_asset
from .interfaces import (
    CatcherProposer,
    EventAnchorDetector,
    PoseExtractor,
    SceneDetector,
    TemporalStanceClassifier,
)
from .model import StanceClassifier
from .schemas import EventAnchor, FrameBox, PitchStanceResult, PoseObservation
from .temporal import StableWindow, choose_stable_window, contiguous_groups

_MODEL_CACHE: dict[str, YOLO] = {}


def _load_model(path: Path) -> YOLO:
    key = str(path.resolve())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]


def _video_info(video_path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return fps, frame_count, width, height


def _sampled_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int,
) -> Iterable[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_index = max(0, start_frame)
    while frame_index <= end_frame:
        if not capture.grab():
            break
        if (frame_index - start_frame) % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                break
            yield frame_index, frame
        frame_index += 1
    capture.release()


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _predict_kwargs(device: str | None, image_size: int, confidence: float) -> dict:
    kwargs = {
        "verbose": False,
        "imgsz": image_size,
        "conf": confidence,
    }
    if device:
        kwargs["device"] = device
    return kwargs


@dataclass
class PitchStanceConfig:
    pose_model_path: Path = POSE_MODEL_PATH
    phc_model_path: Path | None = field(default_factory=lambda: resolve_model_asset("baseballcv_phc"))
    event_model_path: Path | None = field(default_factory=lambda: resolve_model_asset("baseballcv_glove"))
    device: str | None = field(
        default_factory=lambda: os.environ.get("CATCHER_STANCE_DEVICE")
    )
    scene_stride: int = 12
    scene_cut_threshold: float = 0.25
    phc_stride_seconds: float = 0.5
    event_stride: int = 5
    pose_stride: int = 2
    pre_impact_start_seconds: float = 1.75
    pre_impact_end_seconds: float = 0.45
    stable_window_seconds: float = 0.8
    min_stable_window_seconds: float = 0.6
    min_valid_pose_frames: int = 8
    min_valid_coverage: float = 0.60
    min_vote_share: float = 0.65
    pose_confidence: float = 0.20
    event_confidence: float = 0.15
    phc_confidence: float = 0.20
    batch_size: int = 8
    retry_batch_size: int = 4
    pose_on_crops: bool = False
    scene_detector: SceneDetector | None = None
    catcher_proposer: CatcherProposer | None = None
    event_anchor_detector: EventAnchorDetector | None = None
    pose_extractor: PoseExtractor | None = None
    stance_classifier: TemporalStanceClassifier | None = None


class HistogramSceneDetector:
    def __init__(self, stride: int = 12, threshold: float = 0.25):
        self.stride = stride
        self.threshold = threshold

    def detect_cuts(self, video_path: Path) -> list[int]:
        _, frame_count, _, _ = _video_info(video_path)
        previous_histogram = None
        cuts: list[int] = []
        for frame_index, frame in _sampled_frames(video_path, 0, frame_count - 1, self.stride):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(histogram, histogram)
            if previous_histogram is not None:
                distance = cv2.compareHist(
                    previous_histogram,
                    histogram,
                    cv2.HISTCMP_BHATTACHARYYA,
                )
                if distance >= self.threshold:
                    cuts.append(frame_index)
            previous_histogram = histogram
        return cuts


class YOLOCatcherProposer:
    def __init__(self, model_path: Path, config: PitchStanceConfig):
        self.model = _load_model(model_path)
        self.config = config
        self.class_id = next(
            (class_id for class_id, name in self.model.names.items() if str(name).lower() == "catcher"),
            2,
        )

    def propose(self, video_path: Path, cuts: Sequence[int]) -> list[FrameBox]:
        fps, frame_count, _, _ = _video_info(video_path)
        stride = max(1, int(round(fps * self.config.phc_stride_seconds)))
        samples = list(_sampled_frames(video_path, 0, frame_count - 1, stride))
        proposals: list[FrameBox] = []
        for batch in _batched(samples, self.config.batch_size):
            results = self.model.predict(
                [frame for _, frame in batch],
                **_predict_kwargs(self.config.device, 640, self.config.phc_confidence),
            )
            for (frame_index, _), result in zip(batch, results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                matches = np.flatnonzero(classes == self.class_id)
                if not len(matches):
                    continue
                best_index = int(matches[np.argmax(confidences[matches])])
                proposals.append(
                    FrameBox(
                        frame_index=frame_index,
                        box=tuple(float(value) for value in boxes[best_index]),
                        confidence=float(confidences[best_index]),
                        source="baseballcv_phc",
                    )
                )

        groups = contiguous_groups(
            proposals,
            frame_of=lambda item: item.frame_index,
            max_gap_frames=max(stride * 3, int(round(fps))),
            cut_frames=cuts,
        )
        if not groups:
            return []
        return max(
            groups,
            key=lambda group: len(group) * float(np.mean([item.confidence for item in group])),
        )


class YOLOBallEventAnchor:
    def __init__(self, model_path: Path, config: PitchStanceConfig):
        self.model = _load_model(model_path)
        self.config = config
        names = {int(key): str(value).lower() for key, value in self.model.names.items()}
        self.ball_class = next((key for key, value in names.items() if value in {"baseball", "ball"}), 2)
        self.plate_class = next((key for key, value in names.items() if value == "homeplate"), 1)

    def detect(
        self,
        video_path: Path,
        search_start: int,
        search_end: int,
    ) -> EventAnchor | None:
        samples = list(
            _sampled_frames(
                video_path,
                search_start,
                search_end,
                self.config.event_stride,
            )
        )
        ball_rows: list[tuple[int, float, np.ndarray]] = []
        plate_centers: list[np.ndarray] = []
        for batch in _batched(samples, self.config.batch_size):
            results = self.model.predict(
                [frame for _, frame in batch],
                **_predict_kwargs(self.config.device, 640, self.config.event_confidence),
            )
            for (frame_index, _), result in zip(batch, results):
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxyn.cpu().numpy()
                ball_matches = np.flatnonzero(classes == self.ball_class)
                if len(ball_matches):
                    index = int(ball_matches[np.argmax(confidences[ball_matches])])
                    box = boxes[index]
                    center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    ball_rows.append((frame_index, float(confidences[index]), center))
                plate_matches = np.flatnonzero(classes == self.plate_class)
                if len(plate_matches):
                    index = int(plate_matches[np.argmax(confidences[plate_matches])])
                    box = boxes[index]
                    plate_centers.append(
                        np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    )

        groups = contiguous_groups(
            ball_rows,
            frame_of=lambda item: item[0],
            max_gap_frames=self.config.event_stride * 2,
        )
        trajectories = [
            group
            for group in groups
            if len(group) >= 3
            and group[-1][0] - group[0][0] >= self.config.event_stride * 2
        ]
        if not trajectories:
            return None

        plate_center = np.median(plate_centers, axis=0) if plate_centers else None

        def trajectory_score(group: Sequence[tuple[int, float, np.ndarray]]) -> float:
            centers = [row[2] for row in group]
            path_length = sum(
                float(np.linalg.norm(second - first))
                for first, second in zip(centers, centers[1:])
            )
            endpoint_penalty = (
                float(np.linalg.norm(centers[-1] - plate_center))
                if plate_center is not None
                else 0.0
            )
            return len(group) + min(path_length, 2.0) - endpoint_penalty

        trajectory = max(trajectories, key=trajectory_score)
        confidence = float(np.mean([row[1] for row in trajectory]))
        return EventAnchor(
            frame_index=trajectory[-1][0],
            confidence=confidence,
            source="baseballcv_ball_trajectory",
            trajectory_start_frame=trajectory[0][0],
            trajectory_length=len(trajectory),
        )


class YOLOCatcherPoseExtractor:
    def __init__(self, model_path: Path, config: PitchStanceConfig):
        self.model = _load_model(model_path)
        self.config = config

    @staticmethod
    def _nearest_proposal(
        frame_index: int,
        proposals: Sequence[FrameBox],
        max_distance: int,
    ) -> FrameBox | None:
        if not proposals:
            return None
        nearest = min(proposals, key=lambda item: abs(item.frame_index - frame_index))
        return nearest if abs(nearest.frame_index - frame_index) <= max_distance else None

    @staticmethod
    def _expanded_crop(
        frame: np.ndarray,
        box: Sequence[float],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        left = max(0, int(x1 - 0.30 * box_width))
        right = min(width, int(x2 + 0.30 * box_width))
        top = max(0, int(y1 - 0.22 * box_height))
        bottom = min(height, int(y2 + 0.15 * box_height))
        return frame[top:bottom, left:right], (left, top)

    def _crop_observation(
        self,
        frame_index: int,
        result,
        offset: tuple[int, int],
    ) -> PoseObservation | None:
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        box_confidences = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is None:
            keypoint_confidences = np.ones(keypoints.shape[:2], dtype=float)
        else:
            keypoint_confidences = result.keypoints.conf.cpu().numpy()
        crop_height, crop_width = result.orig_shape
        center = np.asarray([crop_width / 2.0, crop_height / 2.0])
        scores = []
        for index, box in enumerate(boxes):
            box_center = np.asarray([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
            center_distance = np.linalg.norm(box_center - center) / max(crop_width, crop_height, 1)
            lower_quality = float(np.mean(keypoint_confidences[index, 11:17]))
            scores.append(float(box_confidences[index]) + 0.5 * lower_quality - center_distance)
        selected = int(np.argmax(scores))
        points = keypoints[selected].astype(float)
        box = boxes[selected].astype(float)
        points += np.asarray(offset)
        box[[0, 2]] += offset[0]
        box[[1, 3]] += offset[1]
        confidences = keypoint_confidences[selected].astype(float)
        return self._make_observation(
            frame_index,
            box,
            points,
            confidences,
            float(box_confidences[selected]),
            "phc_crop_pose",
        )

    def _full_frame_observation(self, frame_index: int, result) -> PoseObservation | None:
        detection = detect_catcher_from_res_item(result, cfg=catcher_config)
        if detection is None:
            return None
        selected = int(detection["index"])
        if result.keypoints.conf is None:
            confidences = np.ones(17, dtype=float)
        else:
            confidences = result.keypoints.conf[selected].cpu().numpy().astype(float)
        return self._make_observation(
            frame_index,
            np.asarray(detection["box"], dtype=float),
            np.asarray(detection["keypoints"], dtype=float),
            confidences,
            float(detection["confidence"]),
            "full_frame_pose_gate",
        )

    def _proposal_observation(
        self,
        frame_index: int,
        result,
        proposal: FrameBox,
    ) -> PoseObservation | None:
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        box_confidences = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is None:
            keypoint_confidences = np.ones(keypoints.shape[:2], dtype=float)
        else:
            keypoint_confidences = result.keypoints.conf.cpu().numpy()
        target = np.asarray(proposal.box, dtype=float)

        def overlap(box: np.ndarray) -> float:
            x1 = max(box[0], target[0])
            y1 = max(box[1], target[1])
            x2 = min(box[2], target[2])
            y2 = min(box[3], target[3])
            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            union = (
                max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                + max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
                - intersection
            )
            return intersection / max(union, 1.0)

        scores = [
            1.5 * overlap(box)
            + 0.35 * float(box_confidences[index])
            + 0.35 * float(np.mean(keypoint_confidences[index, 11:17]))
            for index, box in enumerate(boxes)
        ]
        selected = int(np.argmax(scores))
        if overlap(boxes[selected]) < 0.10:
            return None
        return self._make_observation(
            frame_index,
            boxes[selected].astype(float),
            keypoints[selected].astype(float),
            keypoint_confidences[selected].astype(float),
            float(box_confidences[selected]),
            "phc_guided_full_frame_pose",
        )

    def _make_observation(
        self,
        frame_index: int,
        box: np.ndarray,
        points: np.ndarray,
        confidences: np.ndarray,
        box_confidence: float,
        source: str,
    ) -> PoseObservation | None:
        lower = confidences[11:17]
        if int(np.sum(lower >= self.config.pose_confidence)) < 5:
            return None
        if int(np.sum(confidences >= self.config.pose_confidence)) < 9:
            return None
        limb_lengths = []
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            limb_lengths.extend(
                [
                    float(np.linalg.norm(points[hip] - points[knee])),
                    float(np.linalg.norm(points[knee] - points[ankle])),
                ]
            )
        positive_lengths = [value for value in limb_lengths if value > 1.0]
        if len(positive_lengths) < 3:
            return None
        if max(positive_lengths) / max(min(positive_lengths), 1.0) > 5.0:
            return None
        quality = float(
            0.55 * np.mean(np.clip(lower, 0.0, 1.0))
            + 0.25 * np.mean(np.clip(confidences, 0.0, 1.0))
            + 0.20 * np.clip(box_confidence, 0.0, 1.0)
        )
        return PoseObservation(
            frame_index=frame_index,
            box=box,
            keypoints=points,
            keypoint_confidences=confidences,
            quality=quality,
            source=source,
        )

    def extract(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        proposals: Sequence[FrameBox],
    ) -> list[PoseObservation]:
        fps, _, _, _ = _video_info(video_path)
        samples = list(
            _sampled_frames(video_path, start_frame, end_frame, self.config.pose_stride)
        )
        prepared = []
        max_proposal_distance = max(1, int(round(fps * 0.8)))
        for frame_index, frame in samples:
            proposal = self._nearest_proposal(frame_index, proposals, max_proposal_distance)
            if proposal is not None and self.config.pose_on_crops:
                crop, offset = self._expanded_crop(frame, proposal.box)
                if crop.size:
                    prepared.append((frame_index, crop, offset, True, proposal))
            else:
                prepared.append((frame_index, frame, (0, 0), False, proposal))

        observations: list[PoseObservation] = []
        pose_image_size = 640 if self.config.pose_on_crops else 512
        for batch in _batched(prepared, self.config.batch_size):
            results = self.model.predict(
                [frame for _, frame, _, _, _ in batch],
                **_predict_kwargs(
                    self.config.device,
                    pose_image_size,
                    self.config.pose_confidence,
                ),
            )
            for (frame_index, _, offset, cropped, proposal), result in zip(batch, results):
                if cropped:
                    observation = self._crop_observation(frame_index, result, offset)
                elif proposal is not None:
                    observation = self._proposal_observation(frame_index, result, proposal)
                else:
                    observation = self._full_frame_observation(frame_index, result)
                if observation is None:
                    continue
                if observations:
                    previous = observations[-1]
                    previous_center = (previous.box[:2] + previous.box[2:]) / 2.0
                    center = (observation.box[:2] + observation.box[2:]) / 2.0
                    scale = max(float(previous.box[3] - previous.box[1]), 1.0)
                    if np.linalg.norm(center - previous_center) / scale > 1.25:
                        continue
                observations.append(observation)
        return observations


class RollingMLPClassifier:
    label_mapping = {
        "Knee-down Left": "LKD",
        "Knee-down Right": "RKD",
        "Squat": "Squat",
    }

    def __init__(self):
        self.classifier = StanceClassifier()

    def aggregate(
        self,
        observations: Sequence[PoseObservation],
        max_gap_frames: int,
    ) -> tuple[str | None, float, dict[str, float], np.ndarray | None, int]:
        predictions = []
        for end_index in range(6, len(observations)):
            sequence = observations[end_index - 6 : end_index + 1]
            gaps = [
                second.frame_index - first.frame_index
                for first, second in zip(sequence, sequence[1:])
            ]
            if max(gaps, default=0) > max_gap_frames:
                continue
            if self._has_joint_side_swap(sequence):
                continue
            features = np.concatenate(
                [normalize_keypoints(item.keypoints).reshape(-1) for item in sequence]
            )
            raw_label, model_confidence = self.classifier.predict(features)
            label = self.label_mapping.get(raw_label, raw_label)
            pose_quality = float(np.mean([item.quality for item in sequence]))
            weight = max(1e-6, model_confidence * pose_quality)
            predictions.append(
                {
                    "label": label,
                    "model_confidence": model_confidence,
                    "weight": weight,
                    "features": features,
                    "center_frame": sequence[3].frame_index,
                }
            )
        if not predictions:
            return None, 0.0, {}, None, 0

        weights = Counter()
        for prediction in predictions:
            weights[prediction["label"]] += prediction["weight"]
        total_weight = float(sum(weights.values()))
        distribution = {
            label: float(weight / total_weight)
            for label, weight in sorted(weights.items())
        }
        label = max(distribution, key=distribution.get)
        matching = [item for item in predictions if item["label"] == label]
        confidence = float(
            distribution[label]
            * np.average(
                [item["model_confidence"] for item in matching],
                weights=[item["weight"] for item in matching],
            )
        )
        middle_frame = int(np.median([item.frame_index for item in observations]))
        representative = min(
            matching,
            key=lambda item: abs(item["center_frame"] - middle_frame),
        )
        return label, confidence, distribution, representative["features"], len(predictions)

    @staticmethod
    def _has_joint_side_swap(observations: Sequence[PoseObservation]) -> bool:
        side_swap = np.asarray([12, 11, 14, 13, 16, 15])
        lower_body = np.asarray([11, 12, 13, 14, 15, 16])
        for previous, current in zip(observations, observations[1:]):
            x1, y1, x2, y2 = previous.box
            scale = np.asarray([max(x2 - x1, 1.0), max(y2 - y1, 1.0)])
            previous_points = previous.keypoints[lower_body] / scale
            current_points = current.keypoints[lower_body] / scale
            swapped_points = current.keypoints[side_swap] / scale
            direct_cost = float(np.mean(np.linalg.norm(current_points - previous_points, axis=1)))
            swapped_cost = float(np.mean(np.linalg.norm(swapped_points - previous_points, axis=1)))
            if direct_cost > 0.08 and swapped_cost < direct_cost * 0.55:
                return True
        return False


def _proposal_search_bounds(
    proposals: Sequence[FrameBox],
    fps: float,
    frame_count: int,
) -> tuple[int, int]:
    if not proposals:
        return 0, max(0, frame_count - 1)
    padding = int(round(0.6 * fps))
    return (
        max(0, proposals[0].frame_index - padding),
        min(frame_count - 1, proposals[-1].frame_index + padding),
    )


def _rejected_result(
    reason: str,
    provenance: Sequence[str],
    flags: Sequence[str],
    anchor: EventAnchor | None = None,
    diagnostics: dict | None = None,
) -> PitchStanceResult:
    return PitchStanceResult(
        label=None,
        confidence=0.0,
        impact_frame=anchor.frame_index if anchor else None,
        window_start_frame=None,
        window_end_frame=None,
        vote_distribution={},
        valid_frame_count=0,
        camera_quality=0.0,
        detector_provenance=list(provenance),
        quality_flags=list(flags),
        rejection_reason=reason,
        diagnostics=diagnostics or {},
    )


def analyze_pitch_clip(
    video_path: str | Path,
    config: PitchStanceConfig | None = None,
) -> PitchStanceResult:
    config = config or PitchStanceConfig()
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)
    fps, frame_count, _, _ = _video_info(path)
    if frame_count <= 0:
        return _rejected_result("empty_video", [], ["empty_video"])

    provenance: list[str] = []
    flags: list[str] = []
    scene_detector = config.scene_detector or HistogramSceneDetector(
        stride=config.scene_stride,
        threshold=config.scene_cut_threshold,
    )
    cuts = scene_detector.detect_cuts(path)
    provenance.append("histogram_scene_cuts")

    proposals: list[FrameBox] = []
    catcher_proposer = config.catcher_proposer
    if catcher_proposer is None and config.phc_model_path is not None:
        catcher_proposer = YOLOCatcherProposer(config.phc_model_path, config)
    if catcher_proposer is not None:
        proposals = catcher_proposer.propose(path, cuts)
        if proposals:
            provenance.append("baseballcv_phc")
        else:
            flags.append("phc_no_catcher_track")
    else:
        flags.append("phc_model_unavailable")

    search_start, search_end = _proposal_search_bounds(proposals, fps, frame_count)
    anchor = None
    anchor_detector = config.event_anchor_detector
    if anchor_detector is None and config.event_model_path is not None:
        anchor_detector = YOLOBallEventAnchor(config.event_model_path, config)
    if anchor_detector is not None:
        anchor = anchor_detector.detect(
            path,
            search_start,
            search_end,
        )
        # MPS can occasionally exceed Ultralytics' batched NMS time budget on a
        # cold model. Retry with smaller batches only when no trajectory survives.
        if anchor is None and config.batch_size > config.retry_batch_size:
            if isinstance(anchor_detector, YOLOBallEventAnchor) and config.event_model_path is not None:
                retry_detector = YOLOBallEventAnchor(
                    config.event_model_path,
                    replace(config, batch_size=config.retry_batch_size),
                )
                anchor = retry_detector.detect(path, search_start, search_end)
            else:
                anchor = anchor_detector.detect(path, search_start, search_end)
        if anchor:
            provenance.append(anchor.source)
        else:
            flags.append("ball_trajectory_unavailable")
    else:
        flags.append("event_model_unavailable")

    pose_extractor = config.pose_extractor or YOLOCatcherPoseExtractor(
        config.pose_model_path,
        config,
    )
    if anchor is not None:
        pose_start = max(
            search_start,
            int(round(anchor.frame_index - config.pre_impact_start_seconds * fps)),
        )
        pose_end = min(
            search_end,
            int(round(anchor.frame_index - config.pre_impact_end_seconds * fps)),
        )
    else:
        pose_start, pose_end = search_start, search_end
    if pose_end <= pose_start:
        return _rejected_result(
            "invalid_temporal_search_range",
            provenance,
            flags,
            anchor=anchor,
        )

    observations = pose_extractor.extract(
        path,
        pose_start,
        pose_end,
        proposals,
    )
    provenance.append("yolo26_pose")

    def select_window(items: Sequence[PoseObservation]) -> StableWindow | None:
        if len(items) < config.min_valid_pose_frames:
            return None
        return choose_stable_window(
            items,
            fps=fps,
            sample_stride=config.pose_stride,
            min_seconds=config.min_stable_window_seconds,
            target_seconds=config.stable_window_seconds,
            max_gap_multiplier=4,
            cut_frames=cuts,
            require_following_motion=anchor is None,
        )

    window = select_window(observations)
    if window is None and config.batch_size > config.retry_batch_size:
        if isinstance(pose_extractor, YOLOCatcherPoseExtractor):
            retry_extractor = YOLOCatcherPoseExtractor(
                config.pose_model_path,
                replace(config, batch_size=config.retry_batch_size),
            )
            observations = retry_extractor.extract(
                path,
                pose_start,
                pose_end,
                proposals,
            )
        else:
            observations = pose_extractor.extract(
                path,
                pose_start,
                pose_end,
                proposals,
            )
        window = select_window(observations)
        flags.append("reduced_batch_pose_retry")

    if window is None and anchor is not None:
        window = choose_stable_window(
            observations,
            fps=fps,
            sample_stride=config.pose_stride,
            min_seconds=0.45,
            target_seconds=0.6,
            max_gap_multiplier=4,
            cut_frames=cuts,
        )
        if window is not None:
            flags.append("short_set_window")

    if len(observations) < config.min_valid_pose_frames:
        return _rejected_result(
            "insufficient_valid_pose_frames",
            provenance,
            [*flags, "low_pose_coverage"],
            anchor=anchor,
            diagnostics={
                "fps": fps,
                "pose_search_start": pose_start,
                "pose_search_end": pose_end,
                "pose_observations": len(observations),
                "camera_cuts": cuts,
            },
        )
    if window is None:
        return _rejected_result(
            "no_stable_set_stance_window",
            provenance,
            [*flags, "unstable_pose_sequence"],
            anchor=anchor,
            diagnostics={
                "fps": fps,
                "pose_observations": len(observations),
                "camera_cuts": cuts,
            },
        )
    if anchor is not None and window.end_frame >= anchor.frame_index:
        return _rejected_result(
            "window_crosses_impact",
            provenance,
            [*flags, "invalid_window"],
            anchor=anchor,
        )

    expected_frames = (
        int(round((window.end_frame - window.start_frame) / config.pose_stride)) + 1
    )
    coverage = min(1.0, len(window.observations) / max(expected_frames, 1))
    if coverage < config.min_valid_coverage:
        return _rejected_result(
            "insufficient_window_coverage",
            provenance,
            [*flags, "low_pose_coverage"],
            anchor=anchor,
            diagnostics={"coverage": coverage},
        )

    classifier = config.stance_classifier or RollingMLPClassifier()
    label, confidence, distribution, features, sequence_count = classifier.aggregate(
        window.observations,
        max_gap_frames=config.pose_stride * 2,
    )
    if label is None:
        return _rejected_result(
            "insufficient_contiguous_classifier_frames",
            provenance,
            [*flags, "no_classifier_sequence"],
            anchor=anchor,
        )
    vote_share = distribution[label]
    if vote_share < config.min_vote_share:
        return _rejected_result(
            "ambiguous_stance_vote",
            provenance,
            [*flags, "low_vote_share"],
            anchor=anchor,
            diagnostics={
                "vote_distribution": distribution,
                "window_start_frame": window.start_frame,
                "window_end_frame": window.end_frame,
            },
        )

    camera_quality = float(
        coverage * np.mean([item.quality for item in window.observations])
    )
    if anchor is None:
        flags.append("motion_fallback_anchor")
        provenance.append("motion_followed_stable_window")
    return PitchStanceResult(
        label=label,
        confidence=confidence,
        impact_frame=anchor.frame_index if anchor else None,
        window_start_frame=window.start_frame,
        window_end_frame=window.end_frame,
        vote_distribution=distribution,
        valid_frame_count=len(window.observations),
        camera_quality=camera_quality,
        detector_provenance=provenance,
        quality_flags=flags,
        feature_vector=features,
        diagnostics={
            "fps": fps,
            "frame_count": frame_count,
            "camera_cuts": cuts,
            "pose_search_start": pose_start,
            "pose_search_end": pose_end,
            "pose_observations": len(observations),
            "classifier_sequences": sequence_count,
            "window_motion_score": window.motion_score,
            "window_coverage": coverage,
            "event_confidence": anchor.confidence if anchor else None,
            "event_trajectory_length": anchor.trajectory_length if anchor else 0,
        },
    )


class PitchStanceAnalyzer:
    """Reusable single-worker analyzer with model-backed components initialized once."""

    def __init__(self, config: PitchStanceConfig | None = None):
        base = config or PitchStanceConfig()
        scene_detector = base.scene_detector or HistogramSceneDetector(
            stride=base.scene_stride,
            threshold=base.scene_cut_threshold,
        )
        catcher_proposer = base.catcher_proposer
        if catcher_proposer is None and base.phc_model_path is not None:
            catcher_proposer = YOLOCatcherProposer(base.phc_model_path, base)
        event_anchor_detector = base.event_anchor_detector
        if event_anchor_detector is None and base.event_model_path is not None:
            event_anchor_detector = YOLOBallEventAnchor(base.event_model_path, base)
        pose_extractor = base.pose_extractor or YOLOCatcherPoseExtractor(
            base.pose_model_path,
            base,
        )
        stance_classifier = base.stance_classifier or RollingMLPClassifier()
        self.config = replace(
            base,
            scene_detector=scene_detector,
            catcher_proposer=catcher_proposer,
            event_anchor_detector=event_anchor_detector,
            pose_extractor=pose_extractor,
            stance_classifier=stance_classifier,
        )

    def analyze(self, video_path: str | Path) -> PitchStanceResult:
        return analyze_pitch_clip(video_path, config=self.config)
