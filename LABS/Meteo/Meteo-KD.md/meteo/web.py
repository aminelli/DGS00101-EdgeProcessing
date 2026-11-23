"""API FastAPI e dashboard per previsioni, storico e valutazioni."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from meteo.config import load_settings
from meteo.data import DAILY_TARGETS, read_hourly_csv
from meteo.features import aggregate_daily
from meteo.inference import make_forecast
from meteo.national import active_distillation_dir, load_catalog

LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Previsioni meteo Roma", version="0.1.0")
SETTINGS = load_settings()
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

REPORTS = {
    "operativo": ("metrics.json", "plots"),
    "sperimentale": ("experiments/metrics.json", "experiments/plots"),
    "avanzato": ("advanced_experiments/metrics.json", "advanced_experiments/plots"),
}


@app.get("/health")
def health() -> dict[str, str]:
    model_status = "ready" if SETTINGS.model_path.is_file() else "not_trained"
    return {"status": "ok", "model": model_status}


@app.get("/api/forecast")
def forecast() -> dict:
    try:
        return make_forecast(SETTINGS)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        LOGGER.warning("Richiesta previsione non valida: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.error("Servizio di previsione non disponibile: %s", exc)
        raise HTTPException(status_code=502, detail="Servizio dati meteo non disponibile") from exc
    except Exception as exc:
        LOGGER.exception("Errore inatteso durante la previsione")
        raise HTTPException(status_code=500, detail="Errore interno durante la previsione") from exc


@app.get("/api/history")
def history(days: int = Query(default=90, ge=1, le=3650)) -> dict:
    try:
        daily = aggregate_daily(read_hourly_csv(SETTINGS.hourly_csv))
        if daily.empty:
            raise ValueError("Non sono disponibili giornate complete")
        daily = daily.loc[
            daily["date"] >= daily["date"].max() - pd.Timedelta(days=days - 1)
        ]
        values = daily[["date", *DAILY_TARGETS]].copy()
        if not np.isfinite(values[DAILY_TARGETS].to_numpy(dtype=float)).all():
            raise ValueError("Il periodo storico selezionato contiene valori giornalieri non validi")
        values["date"] = values["date"].dt.strftime("%Y-%m-%d")
        return {
            "location": SETTINGS.location_name,
            "timezone": "UTC",
            "source": "Open-Meteo Historical Weather API: rianalisi/modello",
            "requested_days": days,
            "records": values.to_dict(orient="records"),
        }
    except FileNotFoundError as exc:
        LOGGER.warning("Storico non disponibile: %s", exc)
        raise HTTPException(status_code=503, detail="Storico non disponibile. Eseguire fetch-history.") from exc
    except (ValueError, KeyError) as exc:
        LOGGER.warning("Storico non valido: %s", exc)
        raise HTTPException(status_code=422, detail="Dati storici incompleti o non validi.") from exc
    except OSError as exc:
        LOGGER.exception("Impossibile leggere lo storico")
        raise HTTPException(status_code=500, detail="Errore nella lettura dei dati storici.") from exc


@app.get("/api/evaluations/{report_name}")
def evaluation(report_name: str) -> dict:
    if report_name not in REPORTS:
        raise HTTPException(status_code=404, detail="Valutazione non trovata.")
    path = SETTINGS.artifacts_dir / REPORTS[report_name][0]
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Risultati non disponibili: eseguire il relativo training o esperimento.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.exception("Impossibile leggere la valutazione %s", report_name)
        raise HTTPException(status_code=500, detail="Impossibile leggere i risultati della valutazione.") from exc
    if not isinstance(report, dict) or not isinstance(report.get("targets"), dict):
        LOGGER.error("Formato valutazione non valido: %s", path)
        raise HTTPException(status_code=500, detail="Formato della valutazione non valido.")
    return report


@app.get("/api/evaluations/{report_name}/plots/{target}")
def evaluation_plot(report_name: str, target: str) -> FileResponse:
    if report_name not in REPORTS or target not in DAILY_TARGETS:
        raise HTTPException(status_code=404, detail="Grafico non trovato.")
    path = SETTINGS.artifacts_dir / REPORTS[report_name][1] / f"{target}_test.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Grafico non disponibile per questa valutazione.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/national/catalog")
def national_catalog() -> dict:
    try:
        snapshot = active_distillation_dir(SETTINGS) / "catalog.json"
        return load_catalog(SETTINGS, snapshot if snapshot.is_file() else None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        LOGGER.error("Catalogo nazionale non valido: %s", exc)
        raise HTTPException(status_code=500, detail="Catalogo nazionale non valido.") from exc


@app.get("/api/national/forecast")
def national_forecast(level: str, territory: str, point: str) -> dict:
    from meteo.distillation import forecast_student

    try:
        return forecast_student(SETTINGS, level, territory, point)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        LOGGER.warning("Previsione nazionale non valida: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/national/evaluation")
def national_evaluation(level: str, territory: str) -> dict:
    if level not in ("region", "province"):
        raise HTTPException(status_code=422, detail="Livello territoriale non valido.")
    try:
        path = active_distillation_dir(SETTINGS) / "metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Valutazione nazionale non disponibile: eseguire national-train.") from exc
    except (OSError, ValueError) as exc:
        LOGGER.exception("Impossibile leggere la valutazione nazionale")
        raise HTTPException(status_code=500, detail="Valutazione nazionale non leggibile.") from exc
    if not isinstance(report, dict) or not isinstance(report.get("territories"), dict):
        raise HTTPException(status_code=500, detail="Formato della valutazione nazionale non valido.")
    result = report["territories"].get(f"{level}:{territory}")
    if result is None:
        raise HTTPException(status_code=404, detail="Valutazione territoriale non trovata.")
    return {
        "territory": result, "split": report["split"],
        "distillation_weight": report["distillation_weight"],
        "teacher_size_bytes": report["teacher_size_bytes"], "limitations": report["limitations"],
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/storico", include_in_schema=False)
def history_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "storico.html")


@app.get("/valutazioni", include_in_schema=False)
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "valutazioni.html")


@app.get("/italia", include_in_schema=False)
def national_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "italia.html")
