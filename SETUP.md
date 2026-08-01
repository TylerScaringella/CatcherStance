# Setup

These instructions set up the catcher stance detection web app, install the Python dependencies, install the Playwright browser used for downloading TruMedia clips, and explain how to run and use the app.

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

Install the Playwright Chromium browser. This is required because the downloader opens TruMedia, clicks through the pitch cards, and captures the video URLs.

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

Select an unanalyzed game. When prompted, enter the admin token and upload the exported JSON. The app validates it in headless Chromium and installs it as `data/auth/playwright_state.json` with owner-only permissions. Repeat the export and upload process when the status reports an expired session.

For containers, mount `data/auth/` as a private writable volume. Provide both secrets through the deployment secret manager, enable `CATCHER_STANCE_SECURE_COOKIES=1`, and terminate HTTPS before the app. Never bake either session JSON into an image.

## How To Use The App

1. Start the app with `python src/app.py`.
2. Open `http://127.0.0.1:8000`.
3. Select the included Duke at Liberty sample game from the schedule on the left.
4. Click `Start Detection`. The server automatically resolves the Duke schedule game to TruMedia.
5. Confirm the exact TruMedia game only when multiple candidates are shown.
6. Wait for the job status to move through downloading, detection, and completion.
7. Review the pitch-level results table.
8. Use `Export CSV` or `Export JSON` to save the predictions.
9. Open a pitch row to view its source or compact review clip and optional pose overlay.

Completed games reuse their latest result. `Reprocess` creates a versioned run while preserving previous revisions.

The repository includes a sample completed run at `data/examples/duke-2026-04-21-liberty-sample/` with five downloaded videos, `video_manifest.csv`, `detections.csv`, `detections.json`, `pitch_features.csv`, and `job.json`. Graders can inspect that run without TruMedia access. Running a fresh download still requires TruMedia access.

## Outputs

Each game run is written to:

```text
data/runs/<run-id>/
```

Important output files:

- `video_manifest.csv`: the pitch video URLs, local video paths, and download statuses
- `pitch_features.csv`: extracted catcher keypoint features for each pitch
- `detections.csv`: pitch-level stance predictions in table format
- `detections.json`: pitch-level stance predictions in JSON format
- `job.json`: saved app job status and metadata
- `downloads/`: downloaded pitch-by-pitch video clips
- `artifacts/`: durable generated diagnostics and reusable run-specific assets

The default storage policy creates `artifacts/review-<clip-id>.mp4`, validates every review, and then removes full source downloads. Enable **Retain full source clips** before starting to preserve them. Cleanup failure never deletes source clips.

Temporary processing files are created under `data/tmp/<run-id>-*` and removed
after success or failure. Refreshed schedule data is stored in `data/cache/`
rather than overwriting the checked-in schedule source.

The main prediction fields in `detections.csv` are:

- `pitch_index`: pitch number within the processed run
- `clip_id`: TruMedia clip identifier
- `stance`: predicted catcher stance
- `confidence`: model confidence for the predicted stance
- `status`: whether detection/classification succeeded

## Troubleshooting

If the app starts but downloads do not begin, check the persisted run phase in Active Runs. Background jobs never wait on terminal input.

If TruMedia validation fails, export a fresh session with `python src/trumedia_auth.py` and upload it again through the protected integration dialog.

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

Start the app and verify the checked-in sample without TruMedia:

```bash
python src/app.py
```

Open `http://127.0.0.1:8000/#/games`, choose the Liberty sample, and select
`Review results`. Expected labels are `LKD, LKD, RKD, LKD, LKD`.

Additional browser smoke checks:

1. Confirm the 2026 selector and completed Duke games load.
2. Search for `Liberty` and filter by site, result, conference status, and date.
3. Start the April 14 Liberty game without a session and confirm the protected authentication dialog appears.
4. Open the April 21 sample, verify five results, and filter to `RKD`.
5. Open **Export to GameTracker**, enter a URL shaped like `https://docs.google.com/spreadsheets/d/test/edit`, preview simulated tabs, and simulate an export. Confirm the dialog states that no data was written.
6. Resize to 1440px, 840px, and 390px and confirm navigation, schedule cards, result rows, and dialogs remain usable without horizontal overflow.

Credential-dependent live smoke test:

1. Upload a fresh TruMedia session.
2. Select a completed game and verify automatic or confirmed matching.
3. Monitor discovery, download, detection, review generation, and cleanup in Active Runs.
4. Confirm CSV/JSON exports, compact replay, storage metadata, and versioned reprocessing.
