from __future__ import annotations

import joblib
import numpy as np
import torch
from torch import nn

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

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        scaled = self.scaler.transform(np.asarray([features], dtype=np.float32))
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return str(self.label_encoder.inverse_transform([idx])[0]), float(probs[idx])
