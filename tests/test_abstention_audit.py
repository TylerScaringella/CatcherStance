from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from research.scripts.audit_abstentions import load_manifest, source_group
from research.scripts.replay_abstention_recovery import replay


class AbstentionAuditTests(unittest.TestCase):
    def test_source_group_is_stable_for_pitch_pix_clip(self):
        self.assertEqual(
            "abc123",
            source_group("pitch-pix-abc123-100-112.mp4"),
        )

    def test_unmatched_local_clips_are_retained_for_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "pitch-pix-abc123-100-112.mp4"
            video.touch()
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "choice", "video"])
                writer.writeheader()
                writer.writerow({"id": "1", "choice": "Knee-down Right", "video": "missing.mp4"})
            rows = load_manifest(labels, [root])
            self.assertEqual(1, len(rows))
            self.assertEqual("", rows[0]["choice"])
            self.assertEqual(str(video), rows[0]["path"])

    def test_replay_does_not_recover_without_timing_and_identity_evidence(self):
        result = replay(
            [{
                "id": "ambiguous",
                "accepted": False,
                "abstention_reason": "ambiguous_stance_vote",
                "vote_share": 0.72,
                "event_available": True,
                "window_coverage": 1.0,
                "identity_jumps": 1,
            }],
            min_vote_share=0.65,
        )
        self.assertEqual(0, result["recovered_count"])
        self.assertEqual("insufficient_recovery_evidence", result["blocked"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
