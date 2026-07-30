from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from .schemas import EventAnchor, FrameBox, PoseObservation


class SceneDetector(Protocol):
    def detect_cuts(self, video_path: Path) -> list[int]:
        """Return source-frame indices where a new camera shot begins."""


class CatcherProposer(Protocol):
    def propose(self, video_path: Path, cuts: Sequence[int]) -> list[FrameBox]:
        """Return source-frame-indexed catcher boxes."""


class EventAnchorDetector(Protocol):
    def detect(
        self,
        video_path: Path,
        search_start: int,
        search_end: int,
    ) -> EventAnchor | None:
        """Locate pitch impact or return None when evidence is insufficient."""


class PoseExtractor(Protocol):
    def extract(
        self,
        video_path: Path,
        start_frame: int,
        end_frame: int,
        proposals: Sequence[FrameBox],
    ) -> list[PoseObservation]:
        """Extract quality-scored catcher poses using source frame indices."""


class TemporalStanceClassifier(Protocol):
    def aggregate(
        self,
        observations: Sequence[PoseObservation],
        max_gap_frames: int,
    ) -> tuple[str | None, float, dict[str, float], object | None, int]:
        """Aggregate contiguous pose observations into one pitch-level result."""
