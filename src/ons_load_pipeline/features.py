from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SUBSYSTEM_COLUMNS = ["load_n_mwmed", "load_ne_mwmed", "load_s_mwmed", "load_se_mwmed"]
FEATURE_COLUMNS = [
    "horizon",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "lag_168",
    "lag_336",
    "lag_504",
    "weekly_mean_3",
    "week_delta_1",
    "week_delta_2",
    "origin_lag_1",
    "origin_lag_24",
    "origin_lag_168",
]


@dataclass(frozen=True)
class Examples:
    features: pd.DataFrame
    target: np.ndarray | None
    origins: pd.DatetimeIndex
    timestamps: pd.DatetimeIndex


def _calendar_features(timestamps: pd.DatetimeIndex, horizon: np.ndarray) -> pd.DataFrame:
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "horizon": horizon,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        }
    )


def _one_origin(series: pd.Series, origin: pd.Timestamp, horizon: int) -> Examples:
    timestamps = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
    horizons = np.arange(1, horizon + 1, dtype=np.int16)
    features = _calendar_features(timestamps, horizons)
    weekly_lags = []
    for hours in (168, 336, 504):
        values = series.reindex(timestamps - pd.Timedelta(hours=hours)).to_numpy(dtype=float)
        features[f"lag_{hours}"] = values
        weekly_lags.append(values)
    features["weekly_mean_3"] = np.nanmean(np.vstack(weekly_lags), axis=0)
    features["week_delta_1"] = weekly_lags[0] - weekly_lags[1]
    features["week_delta_2"] = weekly_lags[1] - weekly_lags[2]
    for hours in (1, 24, 168):
        value = series.get(origin - pd.Timedelta(hours=hours), np.nan)
        features[f"origin_lag_{hours}"] = value
    target = series.reindex(timestamps).to_numpy(dtype=float)
    return Examples(
        features=features[FEATURE_COLUMNS],
        target=target,
        origins=pd.DatetimeIndex([origin] * horizon),
        timestamps=timestamps,
    )


def make_training_examples(
    series: pd.Series,
    horizon: int = 168,
    origin_step_hours: int = 24,
) -> Examples:
    series = series.sort_index().astype(float)
    first_origin = series.index.min() + pd.Timedelta(hours=504)
    last_origin = series.index.max() - pd.Timedelta(hours=horizon)
    origins = pd.date_range(first_origin, last_origin, freq=f"{origin_step_hours}h")
    batches = [_one_origin(series, origin, horizon) for origin in origins]
    if not batches:
        raise ValueError("Histórico insuficiente para construir exemplos de treinamento")
    features = pd.concat([batch.features for batch in batches], ignore_index=True)
    target = np.concatenate([batch.target for batch in batches if batch.target is not None])
    all_origins = batches[0].origins.append([batch.origins for batch in batches[1:]])
    timestamps = batches[0].timestamps.append([batch.timestamps for batch in batches[1:]])
    valid = features.notna().all(axis=1).to_numpy() & ~np.isnan(target)
    return Examples(features.loc[valid].reset_index(drop=True), target[valid], all_origins[valid], timestamps[valid])


def make_prediction_features(series: pd.Series, origin: pd.Timestamp, horizon: int = 168) -> Examples:
    batch = _one_origin(series.sort_index().astype(float), origin, horizon)
    if batch.features.isna().any().any():
        missing = list(batch.features.columns[batch.features.isna().any()])
        raise ValueError(f"Histórico insuficiente na origem {origin}; atributos ausentes: {missing}")
    return Examples(batch.features, None, batch.origins, batch.timestamps)

