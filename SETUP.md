# Setup

These instructions set up the catcher stance detection web app, install the Python dependencies, install the Playwright browser used for downloading TruMedia clips, and explain how to run and use the app.

## Requirements

- Python 3.12 or newer
- `pip`
- Internet access
- Access to TruMedia
- The project model files:
  - `model/catcher_stance_mlp.pt`
  - `model/label_encoder.pkl`
  - `model/standard_scaler.pkl`
  - `notebook/yolo26n-pose.pt`

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
python app.py
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

## First TruMedia Login

The first time the downloader runs, Playwright may open a Chromium window and the terminal may ask you to log in to TruMedia.

When prompted:

1. Log in to TruMedia in the Chromium window.
2. Complete any email verification or authentication steps.
3. Navigate to the page that contains the pitch cards for the game.
4. Return to the terminal and press `Enter`.

The app saves the authenticated browser session to:

```text
downloader/playwright_state.json
```

After that file exists, future runs should reuse the saved session. If the session expires, the downloader will ask you to log in again and refresh the saved state.

## How To Use The App

1. Start the app with `python app.py`.
2. Open `http://127.0.0.1:8000`.
3. Select a game from the schedule on the left.
4. Confirm the TruMedia game or pitch-card URL in the URL field.
5. Click `Run Detection`.
6. Wait for the job status to move through downloading, detection, and completion.
7. Review the pitch-level results table.
8. Use `Export CSV` or `Export JSON` to save the predictions.
9. Click `Play` on a pitch row to view the video replay with the catcher pose overlay.

The `Force redownload` checkbox ignores the latest completed run for that game and starts a fresh download/detection run.

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

The main prediction fields in `detections.csv` are:

- `pitch_index`: pitch number within the processed run
- `clip_id`: TruMedia clip identifier
- `stance`: predicted catcher stance
- `confidence`: model confidence for the predicted stance
- `status`: whether detection/classification succeeded

## Troubleshooting

If the app starts but downloads do not begin, check the terminal. The downloader may be waiting for TruMedia login confirmation.

If TruMedia login keeps failing, delete `downloader/playwright_state.json`, restart the app, and log in again when the Chromium window opens.

If Playwright cannot launch Chromium, rerun:

```bash
python -m playwright install chromium
```

If imports fail, make sure the virtual environment is activated and dependencies were installed with:

```bash
pip install -r requirements.txt
```

If model loading fails, confirm that the files in `model/` and `notebook/yolo26n-pose.pt` are present.
