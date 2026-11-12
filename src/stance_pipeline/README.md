# Stance Pipeline

Orchestrates the full stance detection workflow after videos are available.

Main modules:
- `runner.py`: runs downloading plus stance detection for a game/run.
- `detect.py`: extracts pitch features and writes stance prediction outputs.
- `model.py`: loads the trained MLP classifier and predicts stance labels.
- `overlay.py`: renders pose overlays for the web app video preview.
- `analyzer.py`: staged, frame-accurate clip analysis and temporal stance voting.
- `assets.py`: verified download/lookup for optional BaseballCV model weights.
- `temporal.py`: camera-cut/gap-safe stable-window utilities.

## Optional Accuracy Models

Download the BaseballCV PHC and glove/ball models into the ignored model cache:

```bash
python src/download_models.py all
```

The analyzer works without the BaseballCV Python package. If the cached models are
unavailable, it falls back to full-frame catcher pose gating and motion-based windowing.

## Reusable API

```python
from stance_pipeline import analyze_pitch_clip

result = analyze_pitch_clip("pitch.mp4")
print(result.label, result.confidence, result.quality_flags)
```

`LKD` and `RKD` always refer to the catcher's anatomical side.

For multiple clips on one inference worker, reuse initialized components:

```python
from stance_pipeline import PitchStanceAnalyzer

analyzer = PitchStanceAnalyzer()
results = [analyzer.analyze(path) for path in clip_paths]
```

Do not create an unbounded thread per pitch on a single GPU or MPS device. The web
runner uses one FIFO inference lease and overlaps downloads and compact-review encoding
around it. A run requests the lease only after its first clip is ready, and each lease
slot reuses one initialized analyzer without sharing mutable Ultralytics state with
another active slot. Each completed pitch is atomically persisted before the next pitch begins.

## Abstention audit

The current staged analyzer emits structured diagnostics for every accepted or
abstained clip. These diagnostics identify the pipeline stage and stable reason,
along with event-anchor, pose coverage, temporal-window, and catcher-candidate
evidence. To reproduce the current-pipeline baseline and generate a review manifest:

```bash
PYTHONPATH=src ./.venv/bin/python research/scripts/audit_abstentions.py
```

The audit keeps legacy frame-level research summaries separate from pitch-level
results and includes unmatched local clips with an explicit empty label join. Do not
relax a rejection gate solely to increase coverage; review identity continuity and
pre-event timing evidence first.
