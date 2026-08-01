from __future__ import annotations

import unittest
from pathlib import Path

from backend.app import create_app
from project_paths import WEB_DIR


class WebContractTests(unittest.TestCase):
    def test_app_shell_uses_module_frontend_and_accessible_status(self):
        index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        self.assertIn('type="module"', index)
        self.assertIn('href="#/games"', index)
        self.assertIn('href="#/runs"', index)
        self.assertIn('aria-live="polite"', index)
        self.assertIn("Skip to content", index)

    def test_frontend_monitors_visibility_and_avoids_inner_html(self):
        script = (WEB_DIR / "app.js").read_text(encoding="utf-8")
        self.assertIn("visibilitychange", script)
        self.assertIn("schedulePoll", script)
        self.assertNotIn("innerHTML", script)

    def test_static_and_run_routes_smoke(self):
        client = create_app().test_client()
        for path in (
            "/",
            "/styles.css",
            "/app.js",
            "/api/schedule",
            "/api/runs",
        ):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(200, response.status_code)
                response.close()


if __name__ == "__main__":
    unittest.main()
