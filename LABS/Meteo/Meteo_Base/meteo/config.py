"""Configurazione e percorsi del progetto."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    location_name: str = "Roma"
    latitude: float = 41.9028
    longitude: float = 12.4964
    timezone: str = "UTC"
    history_years: int = 10
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    request_timeout_seconds: int = 30
    hybrid_ml_weight: float = 0.5

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def hourly_csv(self) -> Path:
        return self.processed_dir / "weather_hourly.csv"

    @property
    def features_csv(self) -> Path:
        return self.processed_dir / "training_features.csv"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "weather_models.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / "metrics.json"


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Carica la configurazione JSON, risolvendo i percorsi relativi alla root."""
    if config_path is None:
        return Settings()

    path = Path(config_path).expanduser().resolve()
    try:
        values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Impossibile leggere la configurazione {path}: {exc}") from exc

    allowed = set(Settings.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Chiavi di configurazione non riconosciute: {sorted(unknown)}")

    for key in ("data_dir", "artifacts_dir"):
        if key in values:
            candidate = Path(values[key]).expanduser()
            values[key] = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate

    settings = Settings(**values)
    if settings.history_years < 1:
        raise ValueError("history_years deve essere almeno 1")
    if not 0 <= settings.hybrid_ml_weight <= 1:
        raise ValueError("hybrid_ml_weight deve essere compreso tra 0 e 1")
    if settings.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds deve essere positivo")
    if not -90 <= settings.latitude <= 90 or not -180 <= settings.longitude <= 180:
        raise ValueError("Coordinate geografiche non valide")
    if settings.timezone != "UTC":
        raise ValueError("Il progetto usa timestamp UTC; timezone deve essere 'UTC'")
    return settings
