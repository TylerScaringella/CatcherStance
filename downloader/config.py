from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
START_URL = "https://duke-ncaabaseball.trumedianetworks.com/baseball/"
DOWNLOAD_DIR = str(MODULE_DIR / "downloads")
MANIFEST_PATH = str(MODULE_DIR / "video_manifest.csv")
STORAGE_STATE_PATH = str(MODULE_DIR / "playwright_state.json")

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
