from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PitchDetection:
    pitch_index: int
    clip_id: str
    video_path: str
    stance: str
    confidence: float
    status: str
    error: str = ""
    impact_frame: int | str = ""
    window_start_frame: int | str = ""
    window_end_frame: int | str = ""
    valid_frame_count: int = 0
    vote_distribution: str = ""
    detector_provenance: str = ""
    quality_flags: str = ""
    accepted: bool = False
    rejection_reason: str = ""
    camera_quality: float = 0.0
    fps: float | str = ""
    impact_seconds: float | str = ""
    window_start_seconds: float | str = ""
    window_end_seconds: float | str = ""


@dataclass
class PitchFeature:
    choice: str
    id: str
    features: list[float] | str
    status: str


@dataclass(frozen=True)
class FrameBox:
    frame_index: int
    box: tuple[float, float, float, float]
    confidence: float
    source: str


@dataclass
class PoseObservation:
    frame_index: int
    box: np.ndarray
    keypoints: np.ndarray
    keypoint_confidences: np.ndarray
    quality: float
    source: str


@dataclass(frozen=True)
class EventAnchor:
    frame_index: int
    confidence: float
    source: str
    trajectory_start_frame: int | None = None
    trajectory_length: int = 0


@dataclass
class PitchStanceResult:
    label: str | None
    confidence: float
    impact_frame: int | None
    window_start_frame: int | None
    window_end_frame: int | None
    vote_distribution: dict[str, float]
    valid_frame_count: int
    camera_quality: float
    detector_provenance: list[str]
    quality_flags: list[str]
    rejection_reason: str | None = None
    feature_vector: np.ndarray | None = None
    diagnostics: dict[str, Any] | None = None

    @property
    def accepted(self) -> bool:
        return self.label is not None and self.rejection_reason is None
