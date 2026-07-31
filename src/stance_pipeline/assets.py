from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from project_paths import EXTERNAL_MODELS_DIR


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    environment_variable: str


MODEL_ASSETS = {
    "baseballcv_phc": ModelAsset(
        name="baseballcv_phc",
        filename="baseballcv-pitcher-hitter-catcher-v3.pt",
        url=(
            "https://data.balldatalab.com/index.php/s/KP5ZqJKEfjQ785X/"
            "download/pitcher_hitter_catcher_detector_v3.pt"
        ),
        sha256="9484df1c7ef3a432e82114fbc540e7d20a9d6dc76149187742ef698d69a96e81",
        environment_variable="CATCHER_STANCE_PHC_MODEL",
    ),
    "baseballcv_glove": ModelAsset(
        name="baseballcv_glove",
        filename="baseballcv-glove-tracking-v4.pt",
        url=(
            "https://data.balldatalab.com/index.php/s/BwwWJbSsesFSBDa/"
            "download/glove_tracking_v4_YOLOv11.pt"
        ),
        sha256="2c3de8c47180645da20edc252344a1deb7da56591e0740caf325d4df5eb0e5be",
        environment_variable="CATCHER_STANCE_EVENT_MODEL",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_asset(name: str) -> Path | None:
    asset = MODEL_ASSETS[name]
    override = os.environ.get(asset.environment_variable)
    candidate = Path(override).expanduser() if override else EXTERNAL_MODELS_DIR / asset.filename
    if not candidate.exists():
        return None
    if sha256_file(candidate) != asset.sha256:
        raise ValueError(f"Checksum mismatch for model asset: {candidate}")
    return candidate


def download_model_asset(name: str, destination_dir: Path = EXTERNAL_MODELS_DIR) -> Path:
    asset = MODEL_ASSETS[name]
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / asset.filename
    if destination.exists() and sha256_file(destination) == asset.sha256:
        return destination

    with tempfile.NamedTemporaryFile(dir=destination_dir, suffix=".download", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        urllib.request.urlretrieve(asset.url, temporary_path)
        actual_hash = sha256_file(temporary_path)
        if actual_hash != asset.sha256:
            raise ValueError(
                f"Checksum mismatch for {name}: expected {asset.sha256}, received {actual_hash}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Download optional catcher stance model assets.")
    parser.add_argument(
        "models",
        nargs="*",
        choices=[*MODEL_ASSETS, "all"],
        default=["all"],
        help="Assets to download (default: all).",
    )
    args = parser.parse_args()
    names = list(MODEL_ASSETS) if "all" in args.models else args.models
    for name in names:
        print(download_model_asset(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
