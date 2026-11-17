from __future__ import annotations

import io

import pandas as pd

SOURCE_COLUMNS = {
    "id_subsistema": "subsystem",
    "nom_subsistema": "subsystem_name",
    "din_instante": "timestamp",
    "val_cargaenergiahomwmed": "load_mwmed",
}
EXPECTED_SUBSYSTEMS = {"N", "NE", "S", "SE"}


def normalize_csv(content: bytes, expected_year: int) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content), sep=";", encoding="utf-8-sig", decimal=",")
    missing = set(SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV de {expected_year}: {sorted(missing)}")

    frame = frame[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    frame["subsystem"] = frame["subsystem"].astype("string").str.strip().str.upper()
    frame["subsystem_name"] = frame["subsystem_name"].astype("string").str.strip()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["load_mwmed"] = pd.to_numeric(frame["load_mwmed"], errors="raise")
    frame["year"] = frame["timestamp"].dt.year.astype("int16")

    if frame.empty:
        raise ValueError(f"CSV de {expected_year} vazio")
    if set(frame["year"].unique()) != {expected_year}:
        raise ValueError(f"O recurso de {expected_year} contém datas de outro ano")
    unknown = set(frame["subsystem"].unique()) - EXPECTED_SUBSYSTEMS
    if unknown:
        raise ValueError(f"Subsistemas desconhecidos: {sorted(unknown)}")
    missing_subsystems = EXPECTED_SUBSYSTEMS - set(frame["subsystem"].unique())
    if missing_subsystems:
        raise ValueError(f"Subsistemas ausentes: {sorted(missing_subsystems)}")
    if frame[["timestamp", "subsystem", "load_mwmed"]].isna().any().any():
        raise ValueError("Há valores nulos em campos obrigatórios")
    if (frame["load_mwmed"] < 0).any():
        raise ValueError("Há carga negativa")
    if frame.duplicated(["timestamp", "subsystem"]).any():
        raise ValueError("Há chaves timestamp/subsystem duplicadas")
    if not (frame["timestamp"].dt.minute.eq(0) & frame["timestamp"].dt.second.eq(0)).all():
        raise ValueError("Há observações fora da granularidade horária")

    return frame.sort_values(["timestamp", "subsystem"], ignore_index=True)


def build_gold(silver: pd.DataFrame) -> pd.DataFrame:
    pivot = silver.pivot(index="timestamp", columns="subsystem", values="load_mwmed")
    pivot = pivot.rename(columns={code: f"load_{code.lower()}_mwmed" for code in EXPECTED_SUBSYSTEMS})
    subsystem_columns = [f"load_{code.lower()}_mwmed" for code in sorted(EXPECTED_SUBSYSTEMS)]
    pivot = pivot.reindex(columns=subsystem_columns)
    pivot["load_sin_mwmed"] = pivot[subsystem_columns].sum(axis=1, min_count=4)
    return pivot.reset_index().sort_values("timestamp", ignore_index=True)

