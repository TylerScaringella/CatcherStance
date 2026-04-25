# Codex attribution: OpenAI Codex generated this backend configuration module
# for the Flask web app, with project-specific review and edits.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "src" / "web"
RUNS_DIR = ROOT / "data" / "runs"
SCHEDULE_PATH = ROOT / "data" / "schedules" / "duke_baseball_2026.json"
TRUMEDIA_DEFAULT_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
SCHEDULE_REFRESH_INTERVAL_SECONDS = 60 * 30
