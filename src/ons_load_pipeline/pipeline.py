from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .client import ONSClient
from .config import Settings
from .transform import build_gold, normalize_csv

LOGGER = logging.getLogger(__name__)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        for attempt in range(5):
            try:
                os.replace(temporary, destination)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.25 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def _acquire_lock(data_dir: Path) -> tuple[int, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".ons_load_pipeline.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"Outra execução está usando {data_dir}. "
            f"Se nenhuma pipeline estiver ativa, remova o lock obsoleto: {lock_path}"
        ) from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    return descriptor, lock_path


def _replace_years(existing_path: Path, incoming: pd.DataFrame, years: set[int]) -> pd.DataFrame:
    if not existing_path.exists():
        return incoming
    existing = pd.read_parquet(existing_path)
    timestamp = pd.to_datetime(existing["timestamp"])
    return pd.concat([existing.loc[~timestamp.dt.year.isin(years)], incoming], ignore_index=True)


def run_pipeline(start_year: int, end_year: int, settings: Settings) -> dict:
    current_year = datetime.now().year
    if start_year < 2000 or end_year < start_year or end_year > current_year:
        raise ValueError(f"Intervalo inválido: use 2000 <= início <= fim <= {current_year}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    manifest: dict = {
        "run_id": run_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "years": list(range(start_year, end_year + 1)),
        "resources": [],
    }
    settings.manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.manifest_dir / f"{run_id}.json"
    lock_descriptor: int | None = None
    lock_path: Path | None = None

    try:
        lock_descriptor, lock_path = _acquire_lock(settings.data_dir)
        client = ONSClient(settings)
        available = client.list_csv_resources()
        frames: list[pd.DataFrame] = []
        for year in range(start_year, end_year + 1):
            if year not in available:
                raise ValueError(f"Recurso CSV não encontrado no catálogo para {year}")
            resource = available[year]
            LOGGER.info("Baixando carga horária de %s", year)
            content = client.download(resource)
            digest = hashlib.sha256(content).hexdigest()
            raw_dir = settings.bronze_dir / f"year={year}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{resource.resource_id}-{digest[:16]}.csv"
            if not raw_path.exists():
                raw_path.write_bytes(content)
            normalized = normalize_csv(content, year)
            normalized["source_resource_id"] = resource.resource_id
            normalized["source_sha256"] = digest
            normalized["ingested_at_utc"] = manifest["started_at_utc"]
            frames.append(normalized)
            manifest["resources"].append(
                {
                    "year": year,
                    "resource_id": resource.resource_id,
                    "url": resource.url,
                    "last_modified": resource.last_modified,
                    "sha256": digest,
                    "rows": len(normalized),
                    "raw_path": str(raw_path),
                }
            )

        incoming = pd.concat(frames, ignore_index=True)
        years = set(range(start_year, end_year + 1))
        silver = _replace_years(settings.silver_path, incoming, years)
        silver = silver.sort_values(["timestamp", "subsystem"], ignore_index=True)
        if silver.duplicated(["timestamp", "subsystem"]).any():
            raise ValueError("Duplicidade detectada ao consolidar a camada silver")
        gold = build_gold(silver)
        _atomic_parquet(silver, settings.silver_path)
        _atomic_parquet(gold, settings.gold_path)
        manifest.update(
            {
                "status": "success",
                "silver_rows": len(silver),
                "gold_rows": len(gold),
                "silver_path": str(settings.silver_path),
                "gold_path": str(settings.gold_path),
            }
        )
        return manifest
    except Exception as error:
        manifest.update({"status": "failed", "error": str(error)})
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)

