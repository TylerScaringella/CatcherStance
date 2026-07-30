from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .schemas import PoseObservation


@dataclass(frozen=True)
class StableWindow:
    start_frame: int
    end_frame: int
    observations: tuple[PoseObservation, ...]
    motion_score: float
    coverage: float


def contiguous_groups(
    items: Sequence,
    frame_of: Callable[[object], int],
    max_gap_frames: int,
    cut_frames: Sequence[int] = (),
) -> list[list]:
    if not items:
        return []
    ordered = sorted(items, key=frame_of)
    cuts = sorted(cut_frames)
    groups: list[list] = [[ordered[0]]]
    for item in ordered[1:]:
        previous_frame = frame_of(groups[-1][-1])
        frame = frame_of(item)
        crosses_cut = any(previous_frame < cut <= frame for cut in cuts)
        if frame - previous_frame > max_gap_frames or crosses_cut:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def observation_motion(observations: Sequence[PoseObservation]) -> float:
    if len(observations) < 2:
        return float("inf")
    normalized = []
    centers = []
    areas = []
    qualities = []
    lower_body = np.asarray([11, 12, 13, 14, 15, 16])
    for observation in observations:
        x1, y1, x2, y2 = observation.box
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        points = observation.keypoints[lower_body].astype(float).copy()
        points[:, 0] = (points[:, 0] - x1) / width
        points[:, 1] = (points[:, 1] - y1) / height
        normalized.append(points)
        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        areas.append(width * height)
        qualities.append(observation.quality)

    pose_velocity = np.median(
        [
            np.linalg.norm(b - a, axis=1).mean()
            for a, b in zip(normalized, normalized[1:])
        ]
    )
    centers_array = np.asarray(centers)
    box_scale = np.sqrt(max(float(np.median(areas)), 1.0))
    center_velocity = np.median(
        np.linalg.norm(np.diff(centers_array, axis=0), axis=1) / box_scale
    )
    area_variation = float(np.std(areas) / max(float(np.mean(areas)), 1.0))
    quality_penalty = 1.0 - float(np.mean(qualities))
    return float(pose_velocity + 0.35 * center_velocity + 0.15 * area_variation + 0.1 * quality_penalty)


def choose_stable_window(
    observations: Sequence[PoseObservation],
    fps: float,
    sample_stride: int,
    min_seconds: float = 0.6,
    target_seconds: float = 0.8,
    max_gap_multiplier: int = 2,
    cut_frames: Sequence[int] = (),
    require_following_motion: bool = False,
) -> StableWindow | None:
    min_span = max(1, int(round(min_seconds * fps)))
    target_span = max(min_span, int(round(target_seconds * fps)))
    groups = contiguous_groups(
        observations,
        frame_of=lambda item: item.frame_index,
        max_gap_frames=max(1, sample_stride * max_gap_multiplier),
        cut_frames=cut_frames,
    )
    candidates: list[StableWindow] = []
    for group in groups:
        for start_index, first in enumerate(group):
            end_target = first.frame_index + target_span
            chunk = tuple(item for item in group[start_index:] if item.frame_index <= end_target)
            if not chunk or chunk[-1].frame_index - chunk[0].frame_index < min_span:
                continue
            expected = max(1, int(round((chunk[-1].frame_index - chunk[0].frame_index) / sample_stride)) + 1)
            coverage = min(1.0, len(chunk) / expected)
            if coverage < 0.6:
                continue
            motion = observation_motion(chunk)
            if require_following_motion:
                following = [
                    item
                    for item in group
                    if chunk[-1].frame_index < item.frame_index <= chunk[-1].frame_index + int(0.6 * fps)
                ]
                if len(following) >= 3:
                    follow_motion = observation_motion((chunk[-1], *following))
                    if follow_motion <= motion * 1.2:
                        continue
                    motion -= min(0.05, 0.1 * follow_motion)
            candidates.append(
                StableWindow(
                    start_frame=chunk[0].frame_index,
                    end_frame=chunk[-1].frame_index,
                    observations=chunk,
                    motion_score=motion,
                    coverage=coverage,
                )
            )
    return min(candidates, key=lambda item: item.motion_score, default=None)
