from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app
from project_paths import WEB_DIR


class WebContractTests(unittest.TestCase):
    def test_app_shell_uses_module_frontend_and_accessible_status(self):
        index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('type="module"', index)
        self.assertIn('href="#/games"', index)
        self.assertIn('href="#/runs"', index)
        self.assertIn('aria-live="polite"', index)
        self.assertIn('id="trumediaButton"', index)
        self.assertIn("Skip to content", index)

    def test_frontend_monitors_visibility_and_avoids_inner_html(self):
        script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("visibilitychange", script)
        self.assertIn("schedulePoll", script)
        self.assertIn("getRunSummaries", script)
        self.assertIn("AbortController", script)
        self.assertIn("3000", script)
        self.assertIn("30000", script)
        self.assertIn("patchRunNodes", script)
        self.assertIn("patchResults", script)
        self.assertIn("progress.percent == null", script)
        self.assertIn("downloadCount", script)
        self.assertIn("runCardAction", script)
        self.assertIn("gameRunAction", script)
        self.assertIn("Export to GameTracker", script)
        self.assertIn("Connect TruMedia", script)
        self.assertIn("Revalidate current", script)
        self.assertIn("python src/trumedia_auth.py", script)
        self.assertIn("retain_sources", script)
        self.assertIn("function stanceLabel", script)
        self.assertIn("game-context-strip", script)
        self.assertIn("resultAttention", script)
        self.assertIn("Knee-down Left", script)
        self.assertIn("Review action", script)
        self.assertIn("Download Excel for official sheet", script)
        self.assertIn("Save final stance", script)
        self.assertIn("Recover full source", script)
        self.assertIn("KneeDown - Left", script)
        self.assertIn("savePitchReview", script)
        self.assertIn("openAdminUnlock", script)
        self.assertIn("checked-in sample is read-only", script)
        self.assertIn("loadScheduleForSeason", script)
        self.assertIn("schedule_unpublished", script)
        self.assertIn("recoverPitchSource", script)
        self.assertNotIn("innerHTML", script)

    def test_baseball_operations_visual_contract(self):
        index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
        script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("ops-ui-4", index)
        self.assertIn("game-context-strip", styles)
        self.assertIn("run-relationship", styles)
        self.assertIn("needs-attention", styles)
        self.assertIn("prefers-reduced-motion", styles)
        self.assertIn("Manage schedule", script)
        self.assertIn("Refresh from Duke", script)
        self.assertIn("getScheduleSettings", script)
        self.assertIn("schedule_unpublished", script)

    def test_static_and_run_routes_smoke(self):
        schedule_payload = {"team": "Duke", "season": 2026, "games": []}
        with patch("backend.routes.completed_schedule", return_value=schedule_payload):
            client = create_app().test_client()
            for path in (
                "/",
                "/styles.css",
                "/app.js",
                "/fonts/barlow-condensed-700.woff2",
                "/api/schedule",
                "/api/teams",
                "/api/teams/duke/seasons",
                "/api/teams/duke/schedule-settings",
                "/api/teams/duke/schedule?season=2026",
                "/api/integrations/trumedia/status",
                "/api/runs",
                "/api/runs?view=summary",
            ):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(200, response.status_code)
                    response.close()


if __name__ == "__main__":
    unittest.main()
