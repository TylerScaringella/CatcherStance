# Curator

Builds the labeled keypoint dataset used to train and evaluate the stance classifier.

Main entry point: `video_to_keypoint.py`

Important modules:
- `features.py`: extracts catcher keypoint features from one video.
- `dataset.py`: runs feature extraction over labeled videos and writes the dataset CSV.
- `config.py`: central paths and dataset schema.
