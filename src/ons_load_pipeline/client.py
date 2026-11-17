from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Settings


@dataclass(frozen=True)
class Resource:
    year: int
    resource_id: str
    name: str
    url: str
    last_modified: str | None


class ONSClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "power-grid-load-forecasting/0.1"})

    def _request(self, url: str, **kwargs: Any) -> requests.Response:
        if urlparse(url).scheme != "https":
            raise ValueError(f"A fonte precisa usar HTTPS: {url}")
        last_error: Exception | None = None
        for attempt in range(self.settings.max_attempts):
            try:
                response = self.session.get(url, timeout=self.settings.timeout_seconds, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 < self.settings.max_attempts:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Falha ao consultar {url}") from last_error

    def list_csv_resources(self) -> dict[int, Resource]:
        response = self._request(self.settings.ckan_url, params={"id": self.settings.dataset_id})
        payload = response.json()
        if payload.get("success") is not True:
            raise RuntimeError("A API CKAN do ONS retornou success=false")

        discovered: dict[int, Resource] = {}
        for item in payload["result"]["resources"]:
            if str(item.get("format", "")).upper() != "CSV":
                continue
            match = re.search(r"(?<!\d)(20\d{2})(?!\d)", item.get("name", ""))
            if not match:
                continue
            year = int(match.group(1))
            if year in discovered:
                raise RuntimeError(f"Mais de um recurso CSV encontrado para {year}")
            discovered[year] = Resource(
                year=year,
                resource_id=item["id"],
                name=item["name"],
                url=item["url"],
                last_modified=item.get("last_modified"),
            )
        return discovered

    def download(self, resource: Resource) -> bytes:
        return self._request(resource.url).content

