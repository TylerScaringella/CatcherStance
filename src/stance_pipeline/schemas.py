from dataclasses import dataclass


@dataclass
class PitchDetection:
    pitch_index: int
    clip_id: str
    video_path: str
    stance: str
    confidence: float
    status: str
    error: str = ""


@dataclass
class PitchFeature:
    choice: str
    id: str
    features: list[float] | str
    status: str
