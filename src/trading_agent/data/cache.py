"""Small, reproducible file cache for exact daily-data requests."""

from hashlib import sha256
from pathlib import Path

import pandas as pd

from trading_agent.data.models import DailyDataRequest


class FileSystemCache:
    """Persist normalized frames as pickle files keyed by provider and request.

    A cache hit occurs only when provider identity, symbols, timeframe, start,
    and end are identical. This avoids accidental reuse of a broader or
    narrower date range. The cache is local and can safely be deleted.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, provider_key: str, request: DailyDataRequest) -> Path:
        payload = "|".join((provider_key, ",".join(request.symbols), request.start.isoformat(), request.end.isoformat(), request.timeframe))
        return self.directory / f"{sha256(payload.encode()).hexdigest()}.pkl"

    def get(self, provider_key: str, request: DailyDataRequest) -> pd.DataFrame | None:
        path = self._path(provider_key, request)
        return pd.read_pickle(path) if path.exists() else None

    def set(self, provider_key: str, request: DailyDataRequest, data: pd.DataFrame) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        data.to_pickle(self._path(provider_key, request))
