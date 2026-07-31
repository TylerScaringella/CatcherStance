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
