from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    dataset_id: str = "curva-carga"
    ckan_url: str = "https://dados.ons.org.br/api/3/action/package_show"
    timeout_seconds: int = 60
    max_attempts: int = 4

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze" / "ons_load"

    @property
    def silver_path(self) -> Path:
        return self.data_dir / "silver" / "ons_load_hourly.parquet"

    @property
    def gold_path(self) -> Path:
        return self.data_dir / "gold" / "ons_load_hourly.parquet"

    @property
    def manifest_dir(self) -> Path:
        return self.data_dir / "manifests"

