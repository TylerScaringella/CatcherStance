# Setup

These instructions cover installation, credentials, local runtime setup, and smoke tests. Read the [README](README.md) for the human-language workflow, model explanation, source-of-truth rules, and repository map.

## Requirements

- Python 3.12 or newer
- `pip`
- Internet access
- Access to TruMedia
- Optional: `ffmpeg` on `PATH` for faster H.264 compact review encoding; OpenCV is the fallback
- The project model files:
  - `models/classifier/catcher_stance_mlp.pt`
  - `models/classifier/label_encoder.pkl`
  - `models/classifier/standard_scaler.pkl`
  - `models/pose/yolo26n-pose.pt`

## Install

From the project root, create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies.

```bash
pip install -r requirements.txt
```

Install the Playwright Chromium browser. This is required because the downloader opens authenticated TruMedia game pages and reads their game and pitch datasets.

```bash
python -m playwright install chromium
```

## Run The App

Start the Flask web app from the project root:

```bash
python src/app.py
```

The terminal should print:

```text
Catcher Stance web app running at http://127.0.0.1:8000
```

Open this URL in a browser:

```text
http://127.0.0.1:8000
```

Keep the terminal running while you use the app. Stop the server with `Ctrl+C`.

The app opens on the Games view. Active Runs continues monitoring work while you
navigate elsewhere, and Results exposes pitch-level timing, temporal votes,
quality flags, detector provenance, and source/overlay replay.

## TruMedia Session Setup

The web pipeline is headless and never prompts in its background worker. Create a transferable Playwright session on a trusted workstation:

```bash
python src/trumedia_auth.py
```

Log in in the opened browser, complete verification, and return to the terminal when the baseball workspace is visible. The ignored export is written to `data/auth/playwright_state.export.json` by default.

Configure independent application secrets before starting Flask:

```bash
export CATCHER_STANCE_ADMIN_TOKEN='replace-with-a-long-random-value'
export CATCHER_STANCE_SECRET_KEY='replace-with-an-independent-random-value'
python src/app.py
```

Open the application and select **Connect TruMedia** in the header. Enter the admin token and upload `data/auth/playwright_state.export.json`. You do not need to start a game first. The app validates it in headless Chromium and atomically installs it as `data/auth/playwright_state.json` with owner-only permissions. The same dialog can revalidate or replace the session later.

Treat both session files as bearer credentials. They are ignored by Git, must not be committed or shared in logs, and must not be baked into a container image. If the header reports an expired session, export a replacement and install it through the same dialog.

For containers, mount `data/auth/` as a private writable volume. Provide both secrets through the deployment secret manager, enable `CATCHER_STANCE_SECURE_COOKIES=1`, and terminate HTTPS before the app. Never bake either session JSON into an image.

## Docker Compose and Caddy deployment

The repository includes `Dockerfile`, `docker-compose.yml`, and `Caddyfile` for a source-based deployment. The app runs in a private container network and Caddy publishes HTTPS for:

```text
https://catchers.duke.tylerscaringella.com
```

### First deployment

1. Install Docker Engine and the Docker Compose plugin on the host.
2. Clone the repository and enter its root directory.
3. Create the deployment environment file:

   ```bash
   cp .env.example .env
   chmod 600 .env
   ```

4. Edit `.env` and set independent values for `CATCHER_STANCE_ADMIN_TOKEN` and `CATCHER_STANCE_SECRET_KEY`. Set `APP_UID` and `APP_GID` to the numeric host user that owns the checkout (`id -u` and `id -g`) so the container can write the bind-mounted data directory.
5. Create private runtime directories:

   ```bash
   mkdir -p data/auth data/cache data/runs data/tmp
   chmod 700 data/auth data/cache data/runs data/tmp
   ```

6. Create a DNS A and/or AAAA record for `catchers.duke.tylerscaringella.com` pointing to the deployment host.
7. Allow inbound TCP ports 80 and 443. Allow UDP 443 if HTTP/3 is desired.
8. Start the deployment:

   ```bash
   docker compose up -d --build
   docker compose ps
   curl https://catchers.duke.tylerscaringella.com/healthz
   ```

Caddy obtains and renews the public certificate automatically using `tyler.scaringella@duke.edu` from `Caddyfile`. Certificate state is stored in the `caddy_data` Docker volume. The Flask/Gunicorn port is intentionally not published to the host; only Caddy is public.

### Updates, backups, and operations

After a source update:

```bash
git pull
docker compose up -d --build
```

Useful commands:

```bash
docker compose logs -f app
docker compose logs -f caddy
docker compose restart app
docker compose restart caddy
docker compose down
```

Back up the application data and Caddy certificate state separately:

```bash
tar -czf catcher-data-backup.tgz data/
docker run --rm -v catcherstance_caddy_data:/data -v "$PWD":/backup alpine \
  tar -czf /backup/caddy-data-backup.tgz -C /data .
```

`data/auth/` contains bearer credentials, including the TruMedia Playwright session. Restrict host permissions, do not include it in source archives or images, and rotate the admin token and Flask secret independently when access changes.

### Deployment troubleshooting

- **DNS does not resolve:** verify the A/AAAA record and wait for DNS propagation before starting or retrying Caddy.
- **Certificate issuance fails:** confirm ports 80 and 443 reach this host, the hostname is public, and no other service owns those ports. Inspect `docker compose logs caddy`.
- **Port conflict:** stop the process already using 80/443 or change the deployment host; do not publish the app's internal 8000 port as a workaround.
- **Permission errors:** set `APP_UID` and `APP_GID` to the checkout owner's IDs and ensure `data/auth`, `data/cache`, `data/runs`, and `data/tmp` are writable by that user.
- **Chromium is missing:** rebuild with `docker compose build --no-cache app`; the Dockerfile installs Playwright Chromium and its Linux dependencies.
- **TruMedia is expired:** export a fresh Playwright storage state and upload it through the app's **Connect TruMedia** workflow. Never copy session JSON into the image.

Gunicorn is configured with one worker and multiple threads because background jobs and polling state are process-local. Do not increase the worker count without first moving job state and coordination to a shared service.

## How To Use The App

1. Start the app with `python src/app.py`.
2. Open `http://127.0.0.1:8000`.
3. Select the included Duke at Liberty sample game from the schedule on the left.
4. Confirm that the header reports `TruMedia ready`.
5. Click `Start Detection`. The server resolves the game from TruMedia Scores.
6. For a doubleheader, confirm the exact game only when game number, opponent, and result do not produce one unambiguous match.
7. Monitor the independent discovery, download, detection, and compact-review stages.
8. Review available Duke-catching pitches while the remaining pitches continue.
9. Use `Export CSV` or `Export JSON` to save the predictions.
10. Open a pitch row to view its source or compact review clip and optional pose overlay.

Completed games reuse their latest result. `Reprocess` creates a versioned run while preserving previous revisions.

The repository includes a sample completed run at `data/examples/duke-2026-04-21-liberty-sample/` with five downloaded videos, `video_manifest.csv`, `detections.csv`, `detections.json`, `pitch_features.csv`, and `job.json`. It can be inspected without TruMedia access. Running a fresh download still requires TruMedia access.

## Outputs

Each game run is written to:

```text
data/runs/<run-id>/
```

Important local output files:

- `video_manifest.csv`: stable pitch/media identifiers, catcher metadata, local video paths, and download statuses; signed URLs are never persisted
- `pitch_features.csv`: extracted catcher keypoint features for each pitch
- `detections.csv`: pitch-level stance predictions in table format
- `detections.json`: pitch-level stance predictions in JSON format
- `job.json`: saved app job status and metadata
- `state/pitches/`: atomic internal per-pitch records used for restart-safe resume
- `downloads/`: downloaded pitch-by-pitch video clips
- `artifacts/`: durable generated diagnostics and reusable run-specific assets

The default storage policy creates `artifacts/review-<clip-id>.mp4`, validates every review, and then removes full source downloads. Enable **Retain full source clips** before starting to preserve them. Cleanup failure never deletes source clips.

CSV and JSON exports remain unavailable until final aggregation completes. Partial
results are read from the internal per-pitch state through the run-detail API; they
are not duplicated inside `job.json`.
If discovery, download, or processing fails, the partial records remain resumable but
no canonical export replaces the last valid output. Public API responses omit local
paths and provider media references.

## Performance Configuration

The balanced defaults share resources across every live run:

```bash
export CATCHER_STANCE_DOWNLOAD_WORKERS=8
export CATCHER_STANCE_INFERENCE_WORKERS=1
export CATCHER_STANCE_REVIEW_WORKERS=1
```

Keep inference at one worker on Apple MPS unless a target-specific benchmark proves
otherwise. Multiple games can prepare downloads concurrently, but they receive the
accelerator in FIFO order after their first validated clip is ready. Each slot reuses
its initialized analyzer across runs. The app remains responsive because polling is summary-only,
conditional, paused in hidden tabs, and applied through targeted DOM updates.

Temporary processing files are created under `data/tmp/<run-id>-*` and removed
after success or failure. Refreshed schedule data is stored in `data/cache/`
rather than overwriting the checked-in schedule source.

Generated research reports are written under ignored `research/outputs/generated/`.
The main prediction fields in `detections.csv` are:

- `pitch_index`: pitch number within the processed run
- `clip_id`: TruMedia clip identifier
- `stance`: predicted catcher stance
- `confidence`: model confidence for the predicted stance
- `status`: whether detection/classification succeeded

## Troubleshooting

If the app starts but downloads do not begin, check the persisted run phase in Active Runs. Background jobs never wait on terminal input.

If TruMedia validation fails, export a fresh session with `python src/trumedia_auth.py` and upload it through **Connect TruMedia** in the header.

If Playwright cannot launch Chromium, rerun:

```bash
python -m playwright install chromium
```

If imports fail, make sure the virtual environment is activated and dependencies were installed with:

```bash
pip install -r requirements.txt
```

If model loading fails, confirm that the files in `models/classifier/` and `models/pose/yolo26n-pose.pt` are present.

## Smoke Tests

Run the automated suite:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

Run the deterministic polling and partial-results browser test:

```bash
python -m playwright install chromium
RUN_BROWSER_TESTS=1 PYTHONPATH=src ./.venv/bin/python \
  -m unittest discover -s tests -p 'test_web_browser.py' -v
```

This test uses temporary run storage and does not modify checked-in examples or require
a TruMedia session.

Start the app and verify the checked-in sample without TruMedia:

```bash
python src/app.py
```

Open `http://127.0.0.1:8000/#/games`, choose the Liberty sample, and select
`Review results`. Expected labels are `LKD, LKD, RKD, LKD, LKD`.

Additional browser smoke checks:

1. Confirm the 2026 selector and completed Duke games load.
2. Search for `Liberty` and filter by site, result, conference status, and date.
3. Temporarily relocate the ignored installed session, select **Connect TruMedia** in the header, and confirm the protected dialog opens without starting a game.
4. Open the April 21 sample, verify five results, and filter to `RKD`.
5. Open **Export to GameTracker**, enter a URL shaped like `https://docs.google.com/spreadsheets/d/test/edit`, preview simulated tabs, and simulate an export. Confirm the dialog states that no data was written.
6. Resize to 1440px, 840px, and 390px and confirm navigation, schedule cards, result rows, and dialogs remain usable without horizontal overflow.

Credential-dependent live smoke test:

1. Upload a fresh TruMedia session.
2. Select a completed game and verify automatic or confirmed matching.
3. Monitor discovery, download, detection, review generation, and cleanup in Active Runs.
4. Open partial results before the run completes and confirm new pitches appear without a page flicker.
5. Confirm CSV/JSON exports enable only after completion, then verify compact replay, storage metadata, and versioned reprocessing.
