from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from project_paths import CLASSIFIER_METADATA_PATH
from stance_pipeline.analyzer import PitchStanceConfig, RollingMLPClassifier, YOLOCatcherPoseExtractor
from stance_pipeline.assets import MODEL_ASSETS, resolve_model_asset
from stance_pipeline.model import StanceClassifier
from stance_pipeline.schemas import PitchStanceResult


class PipelineContractTests(unittest.TestCase):
    def test_classifier_metadata_matches_artifacts(self):
        metadata = json.loads(CLASSIFIER_METADATA_PATH.read_text(encoding="utf-8"))
        classifier = StanceClassifier()
        self.assertEqual(metadata["feature_count"], classifier.scaler.n_features_in_)
        self.assertEqual(metadata["class_order"], list(classifier.label_encoder.classes_))

    def test_external_assets_have_checksums_and_environment_overrides(self):
        for asset in MODEL_ASSETS.values():
            self.assertEqual(64, len(asset.sha256))
            self.assertTrue(asset.url.startswith("https://"))
            self.assertTrue(asset.environment_variable.startswith("CATCHER_STANCE_"))

    def test_result_acceptance_requires_label_and_no_rejection(self):
        result = PitchStanceResult(
            label="LKD",
            confidence=0.9,
            impact_frame=100,
            window_start_frame=20,
            window_end_frame=60,
            vote_distribution={"LKD": 1.0},
            valid_frame_count=12,
            camera_quality=0.8,
            detector_provenance=["test"],
            quality_flags=[],
        )
        self.assertTrue(result.accepted)

    def test_manifest_adapter_preserves_quality_metadata(self):
        from stance_pipeline.detect import detect_stances_for_manifest

        result = PitchStanceResult(
            label="LKD",
            confidence=0.9,
            impact_frame=100,
            window_start_frame=20,
            window_end_frame=60,
            vote_distribution={"LKD": 1.0},
            valid_frame_count=12,
            camera_quality=0.8,
            detector_provenance=["phc", "ball", "pose"],
            quality_flags=["test_flag"],
            feature_vector=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            video_path = directory_path / "pitch.mp4"
            video_path.touch()
            manifest_path = directory_path / "manifest.csv"
            manifest_path.write_text(
                "clip_id,saved_path,status\n"
                f"pitch-1,{video_path},downloaded\n",
                encoding="utf-8",
            )
            with patch("stance_pipeline.detect.analyze_pitch_clip", return_value=result):
                detections, features = detect_stances_for_manifest(manifest_path)
        self.assertEqual("LKD", detections[0].stance)
        self.assertEqual(100, detections[0].impact_frame)
        self.assertEqual('{"LKD": 1.0}', detections[0].vote_distribution)
        self.assertEqual("phc,ball,pose", detections[0].detector_provenance)
        self.assertEqual("test_flag", detections[0].quality_flags)
        self.assertTrue(detections[0].accepted)
        self.assertEqual(0.8, detections[0].camera_quality)
        self.assertEqual("", detections[0].rejection_reason)
        self.assertEqual("ok", features[0].status)

    def test_pose_quality_rejects_missing_lower_body(self):
        extractor = YOLOCatcherPoseExtractor.__new__(YOLOCatcherPoseExtractor)
        extractor.config = PitchStanceConfig(
            phc_model_path=None,
            event_model_path=None,
        )
        points = np.arange(34, dtype=float).reshape(17, 2)
        confidences = np.ones(17, dtype=float)
        confidences[11:17] = 0.0
        result = extractor._make_observation(
            frame_index=10,
            box=np.asarray([0.0, 0.0, 100.0, 100.0]),
            points=points,
            confidences=confidences,
            box_confidence=0.9,
            source="test",
        )
        self.assertIsNone(result)

    def test_stance_labels_use_anatomical_coco_sides(self):
        self.assertEqual("LKD", RollingMLPClassifier.label_mapping["Knee-down Left"])
        self.assertEqual("RKD", RollingMLPClassifier.label_mapping["Knee-down Right"])

    def test_temporal_classifier_rejects_joint_side_swap(self):
        def pose(frame_index, swapped=False):
            points = np.zeros((17, 2), dtype=float)
            points[11:17] = np.asarray(
                [[30, 40], [70, 40], [30, 65], [70, 65], [30, 90], [70, 90]],
                dtype=float,
            )
            if swapped:
                points[11:17] = points[[12, 11, 14, 13, 16, 15]]
            from stance_pipeline.schemas import PoseObservation

            return PoseObservation(
                frame_index=frame_index,
                box=np.asarray([0.0, 0.0, 100.0, 100.0]),
                keypoints=points,
                keypoint_confidences=np.ones(17),
                quality=1.0,
                source="test",
            )

        observations = [pose(index * 2) for index in range(6)]
        observations.append(pose(12, swapped=True))
        self.assertTrue(RollingMLPClassifier._has_joint_side_swap(observations))


@unittest.skipUnless(
    os.environ.get("RUN_SAMPLE_MODEL_TESTS") == "1",
    "Set RUN_SAMPLE_MODEL_TESTS=1 to run the model-backed sample regression.",
)
class SampleModelRegressionTests(unittest.TestCase):
    def test_five_sample_clips(self):
        from research.scripts.benchmark_staged_pipeline import (
            SAMPLE_EXPECTATIONS,
            benchmark_clip,
        )
        from stance_pipeline import PitchStanceConfig

        phc_model = resolve_model_asset("baseballcv_phc")
        event_model = resolve_model_asset("baseballcv_glove")
        if phc_model is None or event_model is None:
            self.fail(
                "BaseballCV model assets are missing. Run "
                "`python src/download_models.py all` before enabling "
                "RUN_SAMPLE_MODEL_TESTS."
            )
        config = PitchStanceConfig(
            phc_model_path=phc_model,
            event_model_path=event_model,
            device=os.environ.get("CATCHER_STANCE_DEVICE"),
        )
        video_dir = Path("data/examples/duke-2026-04-21-liberty-sample/downloads")
        for video_name, expected in SAMPLE_EXPECTATIONS.items():
            row = benchmark_clip(video_dir / video_name, config)
            self.assertEqual(expected["label"], row["predicted_label"], video_name)
            self.assertIsNone(row["rejection_reason"], video_name)
            self.assertLessEqual(row["impact_error_seconds"], 0.35, video_name)


if __name__ == "__main__":
    unittest.main()
