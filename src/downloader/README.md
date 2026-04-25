# Downloader

Collects pitch video URLs from TruMedia and downloads the pitch-by-pitch clips.

Main entry point: `main.py`

Important modules:
- `crawler.py`: opens TruMedia with Playwright and captures S3 video URLs.
- `files.py`: downloads individual MP4 files.
- `manifest.py`: reads and writes the resumable video manifest.
- `config.py`: downloader paths, selectors, and crawl limits.
