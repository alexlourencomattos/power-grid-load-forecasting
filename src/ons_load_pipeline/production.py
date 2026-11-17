from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

from .config import Settings
from .forecasting import generate_forecast, save_forecast
from .pipeline import run_pipeline
from .training import train_models

LOGGER = logging.getLogger(__name__)


def main() -> None:
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Atualiza dados, treina e publica a previsão de 168 horas")
    parser.add_argument("--start-year", type=int, default=current_year - 1)
    parser.add_argument("--end-year", type=int, default=current_year)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--forecast-dir", type=Path, default=Path("artifacts/forecasts"))
    parser.add_argument("--validation-days", type=int, default=56)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = Settings(data_dir=args.data_dir)
    ingestion = run_pipeline(args.start_year, args.end_year, settings)
    training = train_models(settings.gold_path, args.model_dir, args.validation_days)
    model_path = args.model_dir / "latest.joblib"
    forecast = generate_forecast(settings.gold_path, model_path)
    versioned, latest = save_forecast(forecast, args.forecast_dir)
    LOGGER.info(
        "Produção concluída: ingestão=%s modelo=%s previsão=%s latest=%s",
        ingestion["run_id"],
        training["run_id"],
        versioned,
        latest,
    )


if __name__ == "__main__":
    main()
