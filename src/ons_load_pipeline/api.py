from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query

from .forecasting import generate_forecast
from .model_io import load_artifact

GOLD_PATH = Path(os.getenv("ONS_GOLD_PATH", "data/gold/ons_load_hourly.parquet"))
MODEL_PATH = Path(os.getenv("ONS_MODEL_PATH", "artifacts/models/latest.joblib"))

app = FastAPI(title="Power Grid Load Forecasting API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    try:
        artifact = load_artifact(MODEL_PATH)
        return {"status": "ok", "model_run_id": artifact["run_id"]}
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/v1/forecast")
def forecast(origin: str | None = Query(default=None)) -> dict:
    try:
        frame = generate_forecast(GOLD_PATH, MODEL_PATH, pd.Timestamp(origin) if origin else None)
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    records = frame.copy()
    for column in ("origin", "timestamp"):
        records[column] = records[column].astype(str)
    return {
        "model_run_id": frame["model_run_id"].iloc[0],
        "origin": str(frame["origin"].iloc[0]),
        "horizon_hours": len(frame),
        "forecast": records.to_dict("records"),
    }


def main() -> None:
    uvicorn.run("ons_load_pipeline.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
