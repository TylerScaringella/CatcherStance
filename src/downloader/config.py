from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "downloader"
START_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
DOWNLOAD_DIR = str(DATA_DIR / "downloads")
MANIFEST_PATH = str(DATA_DIR / "video_manifest.csv")
STORAGE_STATE_PATH = str(DATA_DIR / "playwright_state.json")

GRID_SELECTOR = "tmn-grid.contents-container"
VIEWPORT_SELECTOR = "tmn-grid.contents-container .viewport"
CARD_SELECTOR = "tmn-grid.contents-container tmn-paper"

CLICK_TIMEOUT_MS = 5000
POST_CLICK_WAIT_MS = 150
SCROLL_WAIT_MS = 250
MAX_NO_PROGRESS_SCROLLS = 4

REQUEST_TIMEOUT_SECONDS = 180
DOWNLOAD_WORKERS = 8
CHUNK_SIZE = 1024 * 1024
RETRY_COUNT = 3
RETRY_SLEEP_SECONDS = 2
MAX_NEW_URLS_PER_RUN = None
