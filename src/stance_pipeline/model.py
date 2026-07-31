from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import sklearn
import torch
from torch import nn
from sklearn.exceptions import InconsistentVersionWarning

from project_paths import CLASSIFIER_METADATA_PATH

from .config import CLASSIFIER_PATH, LABEL_ENCODER_PATH, SCALER_PATH


class CatcherMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class StanceClassifier:
    def __init__(self):
        self.metadata = self._load_metadata()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        self.model = CatcherMLP(
            input_dim=int(getattr(self.scaler, "n_features_in_", 238)),
            num_classes=len(self.label_encoder.classes_),
        )
        state = torch.load(CLASSIFIER_PATH, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        self._validate_metadata()

    @staticmethod
    def _load_metadata() -> dict:
        if not CLASSIFIER_METADATA_PATH.exists():
            return {}
        return json.loads(CLASSIFIER_METADATA_PATH.read_text(encoding="utf-8"))

    def _validate_metadata(self) -> None:
        expected_features = int(self.metadata.get("feature_count", 238))
        actual_features = int(getattr(self.scaler, "n_features_in_", expected_features))
        if expected_features != actual_features:
            raise ValueError(
                f"Classifier feature schema mismatch: metadata={expected_features}, scaler={actual_features}"
            )
        expected_classes = list(self.metadata.get("class_order", []))
        actual_classes = [str(value) for value in self.label_encoder.classes_]
        if expected_classes and expected_classes != actual_classes:
            raise ValueError(
                f"Classifier class ordering mismatch: metadata={expected_classes}, encoder={actual_classes}"
            )
        trained_version = self.metadata.get("training_runtime", {}).get("scikit_learn")
        self.runtime_version_mismatch = bool(
            trained_version and trained_version != sklearn.__version__
        )

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        scaled = self.scaler.transform(np.asarray([features], dtype=np.float32))
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return str(self.label_encoder.inverse_transform([idx])[0]), float(probs[idx])
