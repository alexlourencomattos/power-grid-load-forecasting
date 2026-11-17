from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .features import SUBSYSTEM_COLUMNS, make_prediction_features
from .model_io import load_artifact


def generate_forecast(gold_path: Path, model_path: Path, origin: pd.Timestamp | None = None) -> pd.DataFrame:
    gold = pd.read_parquet(gold_path).sort_values("timestamp").set_index("timestamp")
    artifact = load_artifact(model_path)
    origin = pd.Timestamp(origin) if origin is not None else gold.index.max()
    if origin not in gold.index:
        raise ValueError(f"Origem {origin} não existe na camada gold")

    forecast: pd.DataFrame | None = None
    for column in SUBSYSTEM_COLUMNS:
        examples = make_prediction_features(gold[column], origin, artifact["horizon_hours"])
        if artifact["selection"][column] == "weekly_baseline":
            values = examples.features["lag_168"].to_numpy()
        else:
            baseline = examples.features["lag_168"].to_numpy()
            values = baseline + artifact["models"][column].predict(examples.features)
        if forecast is None:
            forecast = pd.DataFrame({"timestamp": examples.timestamps})
        forecast[column] = values.clip(min=0)
    if forecast is None:
        raise RuntimeError("Nenhuma previsão foi gerada")
    forecast["load_sin_mwmed"] = forecast[SUBSYSTEM_COLUMNS].sum(axis=1)
    forecast.insert(0, "origin", origin)
    forecast.insert(0, "model_run_id", artifact["run_id"])
    forecast.insert(0, "generated_at_utc", datetime.now(UTC).isoformat())
    return forecast


def save_forecast(forecast: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(forecast["origin"].iloc[0]).strftime("%Y%m%dT%H%M%S")
    versioned = output_dir / f"forecast-{origin}.parquet"
    latest = output_dir / "latest.parquet"
    forecast.to_parquet(versioned, index=False)
    temporary = output_dir / f".latest.{uuid.uuid4().hex}.tmp"
    try:
        forecast.to_parquet(temporary, index=False)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)
    return versioned, latest


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera previsão horária de carga para sete dias")
    parser.add_argument("--gold-path", type=Path, default=Path("data/gold/ons_load_hourly.parquet"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/latest.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/forecasts"))
    parser.add_argument("--origin", type=pd.Timestamp)
    parser.add_argument("--json", action="store_true", help="Imprime a previsão como JSON")
    args = parser.parse_args()
    forecast = generate_forecast(args.gold_path, args.model_path, args.origin)
    versioned, _ = save_forecast(forecast, args.output_dir)
    if args.json:
        print(json.dumps(forecast.assign(timestamp=forecast["timestamp"].astype(str)).to_dict("records"), ensure_ascii=False))
    else:
        print(f"Previsão salva em {versioned} ({len(forecast)} horas)")


if __name__ == "__main__":
    main()

