# Stance Pipeline

Orchestrates the full stance detection workflow after videos are available.

Main modules:
- `runner.py`: runs downloading plus stance detection for a game/run.
- `detect.py`: extracts pitch features and writes stance prediction outputs.
- `model.py`: loads the trained MLP classifier and predicts stance labels.
- `overlay.py`: renders pose overlays for the web app video preview.
