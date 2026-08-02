import os

from project_paths import AUTH_DIR, RAW_DOWNLOADS_DIR

START_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
DOWNLOAD_DIR = str(RAW_DOWNLOADS_DIR)
MANIFEST_PATH = str(RAW_DOWNLOADS_DIR / "video_manifest.csv")
STORAGE_STATE_PATH = str(AUTH_DIR / "playwright_state.json")

REQUEST_TIMEOUT_SECONDS = 180
DOWNLOAD_WORKERS = max(1, int(os.environ.get("CATCHER_STANCE_DOWNLOAD_WORKERS", "8")))
CHUNK_SIZE = 1024 * 1024
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 2
