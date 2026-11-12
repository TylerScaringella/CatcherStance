from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app import create_app


ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_healthz_is_public_and_does_not_expose_runtime_details(self):
        response = create_app().test_client().get("/healthz")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"status": "ok"}, response.get_json())

    def test_local_entrypoint_uses_configurable_host_and_port(self):
        from backend import app as backend_app

        with patch.dict(
            os.environ,
            {"CATCHER_STANCE_HOST": "0.0.0.0", "CATCHER_STANCE_PORT": "8123"},
            clear=False,
        ), patch.object(backend_app.Flask, "run") as run:
            backend_app.main()
        run.assert_called_once_with(host="0.0.0.0", port=8123, threaded=True)

    def test_deployment_files_contain_public_contract(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        caddy = (ROOT / "Caddyfile").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("catchers.duke.tylerscaringella.com", caddy)
        self.assertIn("tyler.scaringella@duke.edu", caddy)
        self.assertIn("reverse_proxy app:8000", caddy)
        self.assertIn('X-Frame-Options "SAMEORIGIN"', caddy)
        self.assertIn("./data:/app/data", compose)
        self.assertIn('"80:80"', compose)
        self.assertIn('"443:443"', compose)
        self.assertIn('"443:443/udp"', compose)
        self.assertIn("condition: service_healthy", compose)
        app_section = compose.split("\n  caddy:", 1)[0]
        self.assertNotIn("\n    ports:", app_section)
        self.assertIn("gunicorn", dockerfile)
        self.assertIn("playwright install --with-deps chromium", dockerfile)
        for token in ("data/auth", "data/runs", "data/tmp", ".env", "*.mp4"):
            self.assertIn(token, ignore)
        self.assertIn(".env", gitignore)
        self.assertIn("!.env.example", gitignore)


if __name__ == "__main__":
    unittest.main()
