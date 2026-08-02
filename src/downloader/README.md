# Downloader

Reads authenticated TruMedia pitch data and downloads Duke-catching broadcast clips.

Main entry point: `main.py`

Important modules:
- `crawler.py`: parses complete pitch datasets, filters Duke as the catching team, and requests temporary media signatures.
- `files.py`: downloads individual MP4 files.
- `manifest.py`: reads and writes the resumable video manifest.
- `config.py`: downloader paths, timeouts, concurrency, and retry limits.

The durable manifest stores stable pitch IDs, catcher/team metadata, the selected camera angle, and private `s3://` media references. Signed AWS URLs are transient and refreshed in bounded download batches, so interrupted runs can resume after earlier signatures expire. Legacy manifests containing signed `s3_url` values remain readable, but their next atomic write removes those credentials.
