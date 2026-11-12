from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryDocumentationTests(unittest.TestCase):
    def test_layered_agent_guides_and_roadmap_exist(self):
        for relative in (
            "AGENTS.md",
            "src/backend/AGENTS.md",
            "src/stance_pipeline/AGENTS.md",
            "src/web/AGENTS.md",
            "research/AGENTS.md",
            "docs/FUTURE_DIRECTIONS.md",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_docs_use_current_research_paths_and_commands(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        setup = (ROOT / "SETUP.md").read_text(encoding="utf-8")
        combined = readme + setup
        for required in (
            "research/scripts/audit_model.py",
            "research/scripts/audit_abstentions.py",
            "research/outputs/generated/",
            "KneeDown - Left",
        ):
            self.assertIn(required, combined)
        for stale in (
            "research/notebooks",
            "docs/screenshots",
            "pitch_stance_research.py",
            "sample_pitch_stance_summary.json",
            "staged_pipeline_benchmark_summary.json",
            "abstention_audit_summary.json`",
        ):
            self.assertNotIn(stale, combined)

    def test_generated_outputs_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("research/outputs/generated/", ignore)
        self.assertIn("research/outputs/*.json", ignore)
        self.assertIn("research/annotations/generated/", ignore)

    def test_model_metadata_matches_documented_contract(self):
        metadata = json.loads((ROOT / "models/classifier/metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["feature_schema"], "coco17_xy_torso_normalized_7_frames")
        self.assertEqual(metadata["class_order"], ["Knee-down Left", "Knee-down Right", "Squat"])
        self.assertEqual(metadata["dataset"]["canonical_path"], "data/processed/dataset/keypoints-labeled.csv")

    def test_deleted_historical_artifacts_are_not_present(self):
        for relative in (
            "research/scripts/pitch_stance_research.py",
            "research/notebooks",
            "docs/screenshots",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)
