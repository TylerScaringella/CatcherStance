# Catcher Stance Detection

My final project for Duke CS372 - Introduction to Applied Machine Learning.

## What it Does

Catcher Stance Detection is a multi-stage machine learning pipeline built to automate one of Duke Baseball's game tracking tasks: identifying the catcher's stance on every pitch. The project downloads pitch-by-pitch video clips for a selected game, runs YOLO pose detection on each clip, isolates the catcher from the detected players, converts the catcher's pose into fixed-length keypoint features, and classifies each pitch as `Knee-down Left`, `Knee-down Right`, or `Squat` with an MLP classifier. The goal is to reduce manual tracking work while preserving pitch-level information that can help identify whether a catcher may be unintentionally tipping pitch type through stance.

## Quick Start

After completing the environment setup, run the web app from the project root:

```bash
python src/app.py
```

Then open `http://127.0.0.1:8000`. The Games workspace loads Duke's official completed-game schedule for the selected season, joins each game to prior analysis runs, and starts the TruMedia-to-detection workflow without requiring analysts to paste media URLs.

New live runs require `CATCHER_STANCE_ADMIN_TOKEN` and an uploaded Playwright storage-state export. Existing results and the checked-in sample remain available without TruMedia access.

The refreshed interface has three linkable work areas:

- `#/games`: choose a game and start, resume, or review a run
- `#/runs`: monitor queued and active work while continuing to use the app
- `#/results/<run-id>`: review stance totals, quality flags, timing, votes, and video

The season selector defaults to 2026 and is ready for 2027. Completed games can be filtered by opponent, date, site, conference status, result, and analysis status. TruMedia games are resolved from the selected date's Scores dataset using Duke's team ID, opponent, result, and doubleheader game number. Ambiguous matches require explicit confirmation.

Run progress is persisted on disk and refreshed when the browser regains focus, so
switching views, changing browser tabs, or reloading does not detach the job. The
browser polls lightweight run summaries every three seconds while work is active,
every 30 seconds while idle, and pauses while hidden. Updates patch the existing
progress elements rather than rebuilding the page, so search focus, dialogs, scroll
position, and video playback remain stable.

The repository includes a five-pitch sample run from Duke at Liberty on April 21, 2026 in `data/examples/duke-2026-04-21-liberty-sample/`. This lets graders inspect videos, manifests, and stance predictions without TruMedia access.

For the accuracy-first detector, download the optional BaseballCV pitcher/hitter/catcher
and glove/ball models:

```bash
python src/download_models.py all
```

The staged analyzer identifies the catcher and pitch impact first, then classifies only
the stationary pre-impact set stance. `LKD` and `RKD` refer to the catcher's anatomical side.
External weights are cached under the ignored `models/external/` directory.

Run the fast unit and contract tests:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

After downloading the BaseballCV assets, run the five-clip model regression:

```bash
python src/download_models.py all
RUN_SAMPLE_MODEL_TESTS=1 CATCHER_STANCE_DEVICE=mps \
  PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

`CATCHER_STANCE_DEVICE=mps` is optional and specific to Apple Silicon. Omit it to
let Ultralytics choose the available device. Custom model locations can be provided
with `CATCHER_STANCE_PHC_MODEL` and `CATCHER_STANCE_EVENT_MODEL`.

## Repository Layout

- `src/`: Flask app, downloader, curator, pose pipeline, and catcher detection code
- `data/auth/`: saved Playwright session state
- `data/raw/`: source inputs and downloaded clip media
- `data/processed/`: generated datasets and derived training files
- `data/runs/`: active app output for new detection runs
- `data/tmp/`: disposable per-job intermediate files, cleaned automatically
- `data/cache/`: ignored runtime cache such as refreshed schedule data
- `data/examples/`: checked-in sample outputs for review and grading
- `models/classifier/`: trained stance classifier artifacts
- `models/pose/`: pose model weights used by the curator and overlay code
- `models/external/`: ignored cache for optional third-party detection weights
- `research/`: notebooks and exploratory analysis

## Runtime Storage Contract

Production code resolves writable paths through `project_paths.py` and
`backend/storage.py`; it must not write next to source modules or notebooks.
Live outputs stay inside their owning `data/runs/<run-id>/` directory, reusable
artifacts belong in that run's `artifacts/` folder, and incomplete work belongs
under `data/tmp/`. Atomic per-pitch resume records live under the owning run's
`state/pitches/` directory and are consolidated into the normal detection exports
when processing completes. Writes to job state, resume records, and exports are atomic.

The API resolves media from server-owned manifests and rejects traversal,
absolute-path escapes, and symlink escapes. `data/examples/` is always read-only.
Authentication state, runtime caches, temporary files, live runs, and external
model weights are ignored by Git.

## TruMedia And Media Lifecycle

TruMedia credentials are never entered into the application. Export a browser session on a trusted workstation:

```bash
python src/trumedia_auth.py
```

Start the app with independent deployment secrets:

```bash
export CATCHER_STANCE_ADMIN_TOKEN='replace-with-a-long-random-value'
export CATCHER_STANCE_SECRET_KEY='replace-with-an-independent-random-value'
python src/app.py
```

Open the application and select **Connect TruMedia** in the persistent header. Enter the admin token and upload `data/auth/playwright_state.export.json`. The session can be configured, replaced, or revalidated before selecting a game; an expired session also reopens the same workflow when a run starts.

The server validates the state in isolated headless Chromium before atomically installing it as `data/auth/playwright_state.json` with owner-only permissions. Both files are bearer credentials, are ignored by Git, and must never be committed or baked into a container image. Container deployments must mount `data/auth/` as a private writable volume and terminate TLS before Flask.

For each matched game, the downloader reads the complete TruMedia pitch dataset and selects only pitches where Duke (`730336256`) is the catching team. Stable `s3://` media references are stored in the private run manifest; expiring signed AWS URLs are generated in short batches and never persisted. Downloads use atomic `.part` promotion and resumable manifests. Full source clips are removed by default only after detections and compact review MP4s validate. Failed or interrupted runs receive fresh media signatures when resumed.

The GameTracker dialog is an interactive prototype. It validates a Google Sheets URL and simulates tab selection and export, but it does not contact Google or persist spreadsheet settings.

## Throughput And Progress

Downloads, inference, and compact-review generation form a bounded staged pipeline.
Validated clips enter detection without waiting for the full game to download, and
completed pitches become reviewable while later pitches continue processing. Multiple
games may download concurrently, but game-level inference uses a FIFO accelerator lease.

The balanced defaults are eight total download slots, one inference worker, and one
review encoder. Override them only after benchmarking the target machine:

```bash
export CATCHER_STANCE_DOWNLOAD_WORKERS=8
export CATCHER_STANCE_INFERENCE_WORKERS=1
export CATCHER_STANCE_REVIEW_WORKERS=1
```

One MPS inference worker is intentional. On the development Apple M3, processing the
same two clips with competing pitch threads took `61.2s`, compared with `40.5s`
sequentially. The pipeline gains throughput by overlapping independent stages rather
than contending for the same accelerator and model objects.

## Video Links

- Demo video: [YouTube](https://youtu.be/cJP-MdaAUTA)
- Technical walkthrough: [YouTube](https://youtu.be/flwCT_3V4m4)

## Evaluation

The final MLP classifier was evaluated on a held-out validation set of 374 labeled pitch clips.

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| Knee-down Left | 0.51 | 0.72 | 0.60 | 57 |
| Knee-down Right | 0.95 | 0.88 | 0.91 | 315 |
| Squat | 0.00 | 0.00 | 0.00 | 2 |

MLP validation accuracy: `85.3%`

For comparison, I also tested a simple logistic regression baseline on the same stance classification task using the validation split.

| Model | Validation Accuracy | Weighted Precision | Weighted Recall | Weighted F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 42.5% | 0.74 | 0.42 | 0.54 |
| MLP Classifier | 85.3% | 0.87 | 0.85 | 0.86 |

The MLP substantially outperformed the logistic regression baseline, especially in overall accuracy and weighted F1-score. The model performed best on `Knee-down Right`, which was also the most common class in the dataset. `Squat` performance is not reliable yet because there were only two squat examples in the test set.
