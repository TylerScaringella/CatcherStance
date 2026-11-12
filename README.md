# Catcher Stance Detection

Catcher Stance Detection helps Duke baseball staff turn pitch video into a reviewable catcher-stance record. It gathers pitch clips, runs a staged detector, keeps evidence and provenance with every result, and lets an analyst correct or complete decisions before exporting a paste-ready workbook.

## The workflow

1. Open the Games workspace and select a completed Duke game.
2. Authenticate the TruMedia session when the game needs new footage.
3. Gather pitch clips and start an analysis run.
4. The pipeline identifies the catcher, anchors the pitch event, selects a pre-delivery window, and classifies the stance.
5. Review accepted and abstained pitches in Results. The original model output stays visible; a reviewed value becomes the final value.
6. Download the Excel handoff after every pitch has a final stance.

The supported final labels are `KneeDown - Left`, `KneeDown - Right`, and `Squat`. Existing compact review clips, full source clips when available, pose overlays, quality flags, timing, votes, and provenance remain available in pitch detail.

## Quick start

From the project root, install the dependencies described in [`SETUP.md`](SETUP.md), then run:

```bash
python src/app.py
```

Open `http://127.0.0.1:8000`. The main routes are:

- `#/games` — choose a game, manage the active season, and start or resume analysis.
- `#/runs` — monitor queued and active work.
- `#/results/<run-id>` — filter, review, and export pitch results.

Live runs require `CATCHER_STANCE_ADMIN_TOKEN` and a Playwright storage-state export for TruMedia. The checked-in Liberty sample and existing results can be inspected without external credentials. Optional BaseballCV weights can be downloaded with:

```bash
python src/download_models.py all
```

Useful environment variables include `CATCHER_STANCE_DEVICE`, `CATCHER_STANCE_PHC_MODEL`, and `CATCHER_STANCE_EVENT_MODEL`. See [`SETUP.md`](SETUP.md) for authentication, model setup, and troubleshooting.

To exercise the complete live workflow without processing a whole game, open Run settings in the selected completed game, enter a Test run pitch limit from 1 to 25, and start detection. The run still discovers the game, downloads real pitch clips, classifies them, builds review media, supports analyst review, and enables Excel export after every selected pitch is finalized. Leave the field blank for a normal full-game run.

While a live run is shown in Active Runs, use Pause to finish the current pitch and keep the run resumable, Stop to finish the current pitch and leave the run stopped, or Cancel to prevent resuming it. Each action asks for confirmation. Paused and stopped runs can be resumed from the same card; canceled runs retain their results until an authorized deletion. Deleting a terminal live run requires the admin token and a second confirmation. The checked-in sample remains read-only.

The application is branded **Duke Catcher Detection** with the **Baseball Operations** workspace label. The route and API behavior remain unchanged.

## How the staged model works

The production analyzer is deliberately staged so a pose is not classified just because a person happens to be visible in one frame:

1. Scene and camera handling removes unusable frames and records cuts or coverage changes.
2. Catcher proposals come from the catcher detector and pose candidates. Geometry, location, scale, and temporal continuity help reject the pitcher, batter, or a transient person.
3. Ball/glove evidence is smoothed to estimate the pitch delivery and arrival event. A verified plate or receiving-zone event is preferred; bounded fallbacks are marked with lower confidence.
4. A bounded window before the event is searched for a stable set stance. Post-receive movement must not enter the classification window.
5. The selected catcher pose is normalized into temporal features, including a short sequence rather than a single image.
6. The trained MLP classifier produces a stance, confidence, vote information, quality metadata, and provenance.
7. Weak identity, timing, coverage, or classifier evidence causes abstention. An abstention is a review task, not a claim that the pitch has no stance.

The current artifact and feature contract are described in `models/classifier/metadata.json` and the pipeline specification at [`research/specs/pitch_stance_pipeline_spec.md`](research/specs/pitch_stance_pipeline_spec.md). API payloads and stored result fields are compatibility contracts; presentation changes must not silently change them.

## What it does well and where to be careful

The system is useful for Duke-catching pitch clips when the catcher is visible and the camera view is stable. It separates catcher evidence from other detected players, uses temporal context, preserves timing/quality/provenance, supports resumable work, and allows analysts to correct both wrong classifications and abstentions.

It is intentionally conservative. The labels are heavily imbalanced toward Knee-down Right, Squat has limited support, camera angle and occlusion affect catcher identity, and weak timing or coverage can produce an abstention. TruMedia session and media availability are also prerequisites for new footage. Official exports therefore require human review of every pitch.

Historical notebook metrics are not the production quality claim. Current research uses source-group-aware evaluation and generated local outputs; do not promote a model from a row-wise split or a tiny rare-class sample.

## Repository map and source of truth

- `src/app.py`, `src/backend/`, and `src/stance_pipeline/` are production behavior.
- `src/web/` is the vanilla DOM frontend; routes and API contracts remain stable.
- `tests/` contains unit, contract, and optional browser coverage.
- `data/examples/` is the checked-in Liberty sample for local inspection.
- `models/classifier/` and `models/pose/` contain checked-in production artifacts; `models/external/` is an ignored cache.
- `research/scripts/` contains active reproducible audits and benchmarks, not runtime behavior.
- `research/outputs/generated/` and `research/annotations/generated/` are local generated outputs and are intentionally ignored.
- `data/auth/`, `data/cache/`, `data/runs/`, `data/tmp/`, downloaded video, and processed datasets are runtime or private data and must remain ignored.
- `docs/` contains durable project documentation, not a screenshot gallery or a generated report archive.

Production paths are resolved through `src/project_paths.py` and backend storage helpers. Run state and exports belong inside their owning run directory and are written atomically. Never commit credentials, signed media URLs, videos, generated research reports, or local runtime state.

## Tests and research commands

Run the default suite from a clean checkout:

```bash
PYTHONPATH=src ./.venv/bin/python -m unittest discover -s tests -v
```

If Chromium is installed:

```bash
RUN_BROWSER_TESTS=1 PYTHONPATH=src ./.venv/bin/python \
  -m unittest discover -s tests -p 'test_web_browser.py' -v
```

Active research entry points write generated results locally:

```bash
PYTHONPATH=src ./.venv/bin/python research/scripts/audit_model.py
PYTHONPATH=src ./.venv/bin/python research/scripts/audit_abstentions.py
PYTHONPATH=src ./.venv/bin/python research/scripts/benchmark_staged_pipeline.py
```

Read the nearest `AGENTS.md` before modifying code. The root guide explains project-wide rules; guides under `src/backend`, `src/stance_pipeline`, `src/web`, and `research` add subsystem constraints. Future work is recorded as a roadmap in [`docs/FUTURE_DIRECTIONS.md`](docs/FUTURE_DIRECTIONS.md), not as completed functionality.
