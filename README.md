# Catcher Stance Detection

My final project for Duke CS372 - Introduction to Applied Machine Learning.

## What it Does

Catcher Stance Detection is a multi-stage machine learning pipeline built to automate one of Duke Baseball's game tracking tasks: identifying the catcher's stance on every pitch. The project downloads pitch-by-pitch video clips for a selected game, runs YOLO pose detection on each clip, isolates the catcher from the detected players, converts the catcher's pose into fixed-length keypoint features, and classifies each pitch as `Knee-down Left`, `Knee-down Right`, or `Squat` with an MLP classifier. The goal is to reduce manual tracking work while preserving pitch-level information that can help identify whether a catcher may be unintentionally tipping pitch type through stance.

## Quick Start

After completing the environment setup, run the web app from the project root:

```bash
python src/app.py
```

Then open `http://127.0.0.1:8000`, select a Duke Baseball game, and run the detection pipeline. The app downloads available pitch videos, processes each clip through catcher detection and stance classification, and writes results to `data/runs/<run-id>/detections.csv`, `detections.json`, `pitch_features.csv`, and `video_manifest.csv`.

The refreshed interface has three linkable work areas:

- `#/games`: choose a game and start, resume, or review a run
- `#/runs`: monitor queued and active work while continuing to use the app
- `#/results/<run-id>`: review stance totals, quality flags, timing, votes, and video

Run progress is persisted on disk and refreshed when the browser regains focus, so
switching views, changing browser tabs, or reloading does not detach the job.

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
under `data/tmp/`. Writes to job state and detection exports are atomic.

The API resolves media from server-owned manifests and rejects traversal,
absolute-path escapes, and symlink escapes. `data/examples/` is always read-only.
Authentication state, runtime caches, temporary files, live runs, and external
model weights are ignored by Git.

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
