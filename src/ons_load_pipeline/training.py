from __future__ import annotations

import argparse
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FEATURE_COLUMNS, SUBSYSTEM_COLUMNS, make_training_examples
from .model_io import save_artifact

LOGGER = logging.getLogger(__name__)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    nonzero = np.abs(actual) > 1e-9
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mape_percent": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100),
    }


def train_models(gold_path: Path, model_dir: Path, validation_days: int = 56) -> dict:
    gold = pd.read_parquet(gold_path).sort_values("timestamp").set_index("timestamp")
    if gold.index.duplicated().any():
        raise ValueError("A camada gold contém timestamps duplicados")
    missing = set(SUBSYSTEM_COLUMNS) - set(gold.columns)
    if missing:
        raise ValueError(f"Colunas ausentes na gold: {sorted(missing)}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    cutoff = gold.index.max() - pd.Timedelta(days=validation_days)
    artifact: dict = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_start": gold.index.min().isoformat(),
        "data_end": gold.index.max().isoformat(),
        "validation_cutoff": cutoff.isoformat(),
        "horizon_hours": 168,
        "feature_columns": FEATURE_COLUMNS,
        "prediction_strategy": "weekly_baseline_plus_residual",
        "models": {},
        "selection": {},
        "metrics": {},
    }

    for column in SUBSYSTEM_COLUMNS:
        LOGGER.info("Gerando exemplos para %s", column)
        examples = make_training_examples(gold[column])
        train_mask = examples.timestamps <= cutoff
        validation_mask = examples.origins > cutoff
        if train_mask.sum() == 0 or validation_mask.sum() == 0:
            raise ValueError("Histórico insuficiente para treino e validação temporal")
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=42,
        )
        train_baseline = examples.features.loc[train_mask, "lag_168"].to_numpy()
        train_residual = examples.target[train_mask] - train_baseline
        model.fit(examples.features.loc[train_mask], train_residual)
        actual = examples.target[validation_mask]
        baseline = examples.features.loc[validation_mask, "lag_168"].to_numpy()
        predicted = baseline + model.predict(examples.features.loc[validation_mask])
        model_metrics = _metrics(actual, predicted)
        baseline_metrics = _metrics(actual, baseline)
        selection = "model" if model_metrics["mae"] < baseline_metrics["mae"] else "weekly_baseline"
        artifact["models"][column] = model
        artifact["selection"][column] = selection
        artifact["metrics"][column] = {
            "model": model_metrics,
            "weekly_baseline": baseline_metrics,
            "selected": selection,
            "validation_rows": int(validation_mask.sum()),
        }
        LOGGER.info(
            "%s: MAE modelo %.2f, baseline %.2f, selecionado %s",
            column,
            model_metrics["mae"],
            baseline_metrics["mae"],
            selection,
        )

    versioned, latest = save_artifact(artifact, model_dir, run_id)
    report = {key: value for key, value in artifact.items() if key != "models"}
    report["versioned_model"] = str(versioned)
    report["latest_model"] = str(latest)
    report_path = model_dir / f"{run_id}.metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelos de carga para horizonte de 168 horas")
    parser.add_argument("--gold-path", type=Path, default=Path("data/gold/ons_load_hourly.parquet"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--validation-days", type=int, default=56)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = train_models(args.gold_path, args.model_dir, args.validation_days)
    LOGGER.info("Treino concluído: %s", report["run_id"])


if __name__ == "__main__":
    main()

