from __future__ import annotations

import os
import uuid
from pathlib import Path

import joblib


def save_artifact(artifact: dict, model_dir: Path, run_id: str) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    versioned = model_dir / f"{run_id}.joblib"
    latest = model_dir / "latest.joblib"
    joblib.dump(artifact, versioned)
    temporary = model_dir / f".latest.{uuid.uuid4().hex}.tmp"
    try:
        joblib.dump(artifact, temporary)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)
    return versioned, latest


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}. Execute ons-train primeiro.")
    return joblib.load(path)

