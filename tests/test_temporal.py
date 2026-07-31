from __future__ import annotations

import unittest

import numpy as np

from stance_pipeline.schemas import PoseObservation
from stance_pipeline.temporal import choose_stable_window, contiguous_groups


def observation(frame_index: int, x_offset: float = 0.0) -> PoseObservation:
    keypoints = np.zeros((17, 2), dtype=float)
    keypoints[:, 0] = np.arange(17) + x_offset
    keypoints[:, 1] = np.arange(17)
    return PoseObservation(
        frame_index=frame_index,
        box=np.asarray([0.0 + x_offset, 0.0, 100.0 + x_offset, 100.0]),
        keypoints=keypoints,
        keypoint_confidences=np.ones(17),
        quality=0.9,
        source="test",
    )


class TemporalWindowTests(unittest.TestCase):
    def test_contiguous_groups_split_on_missing_frames(self):
        items = [observation(frame) for frame in (10, 12, 14, 30, 32)]
        groups = contiguous_groups(
            items,
            frame_of=lambda item: item.frame_index,
            max_gap_frames=4,
        )
        self.assertEqual([[10, 12, 14], [30, 32]], [[x.frame_index for x in g] for g in groups])

    def test_contiguous_groups_split_on_camera_cut(self):
        items = [observation(frame) for frame in range(0, 22, 2)]
        groups = contiguous_groups(
            items,
            frame_of=lambda item: item.frame_index,
            max_gap_frames=4,
            cut_frames=[10],
        )
        self.assertEqual([0, 8], [groups[0][0].frame_index, groups[0][-1].frame_index])
        self.assertEqual([10, 20], [groups[1][0].frame_index, groups[1][-1].frame_index])

    def test_stable_window_never_spans_gap(self):
        items = [observation(frame) for frame in range(0, 48, 2)]
        items.extend(observation(frame, x_offset=(frame - 80) * 0.5) for frame in range(80, 128, 2))
        window = choose_stable_window(
            items,
            fps=30.0,
            sample_stride=2,
            min_seconds=0.6,
            target_seconds=0.8,
        )
        self.assertIsNotNone(window)
        self.assertLess(window.end_frame, 80)


if __name__ == "__main__":
    unittest.main()
