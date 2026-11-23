"""Comandi per acquisizione, preparazione, training ed esportazione."""

from __future__ import annotations

import argparse
import logging
from datetime import date

from meteo.config import load_settings
from meteo.data import fetch_history, read_hourly_csv
from meteo.features import build_training_frame
from meteo.training import export_onnx, format_test_results, train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline meteo per Roma")
    parser.add_argument("--config", help="Percorso a un file JSON di configurazione")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("fetch-history", help="Scarica lo storico Open-Meteo")
    subparsers.add_parser("prepare", help="Crea CSV giornaliero e feature di training")
    subparsers.add_parser("train", help="Addestra e valuta i modelli")
    subparsers.add_parser("experiment", help="Confronta modelli sperimentali senza sostituire quelli operativi")
    nwp_parser = subparsers.add_parser("fetch-nwp-history", help="Scarica le previsioni NWP archiviate")
    nwp_parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 1))
    nwp_parser.add_argument("--end-date", type=date.fromisoformat)
    subparsers.add_parser("advanced-experiment", help="Confronta modelli e NWP con validazione rolling")
    subparsers.add_parser("export", help="Esporta e verifica i modelli ONNX")
    subparsers.add_parser("national-catalog", help="Prepara il catalogo territoriale nazionale")
    national_fetch = subparsers.add_parser("national-fetch", help="Acquisisce lo storico nazionale con budget giornaliero")
    national_fetch.add_argument("--daily-budget", type=int, default=4000, help="Massimo giornaliero locale di chiamate equivalenti (1-4000)")
    national_refresh = subparsers.add_parser("national-refresh", help="Aggiorna in modo incrementale gli storici nazionali")
    national_refresh.add_argument("--daily-budget", type=int, default=4000, help="Massimo locale sulle 24 ore (1-4000)")
    subparsers.add_parser("national-train", help="Addestra teacher nazionale e student territoriali")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(args.config)

    if args.command == "fetch-history":
        frame = fetch_history(settings)
        print(f"Salvate {len(frame)} righe orarie in {settings.hourly_csv}")
    elif args.command == "prepare":
        hourly = read_hourly_csv(settings.hourly_csv)
        frame, columns = build_training_frame(hourly)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(settings.features_csv, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        print(f"Salvati {len(frame)} esempi e {len(columns)} feature in {settings.features_csv}")
    elif args.command == "train":
        hourly = read_hourly_csv(settings.hourly_csv)
        metrics = train_models(hourly, settings)
        print(format_test_results(metrics))
    elif args.command == "experiment":
        from meteo.experiments import format_experiment_results, run_experiments

        hourly = read_hourly_csv(settings.hourly_csv)
        print(format_experiment_results(run_experiments(hourly, settings)))
    elif args.command == "fetch-nwp-history":
        from meteo.nwp_history import default_nwp_end_date, fetch_nwp_history

        result = fetch_nwp_history(settings, args.start_date, args.end_date or default_nwp_end_date())
        print(f"Salvate {len(result)} giornate di previsioni NWP archiviate.")
    elif args.command == "advanced-experiment":
        from meteo.advanced import format_advanced_results, run_advanced_experiment

        hourly = read_hourly_csv(settings.hourly_csv)
        print(format_advanced_results(run_advanced_experiment(hourly, settings)))
    elif args.command == "export":
        hourly = read_hourly_csv(settings.hourly_csv)
        report = export_onnx(settings, hourly)
        print(f"Report ONNX: {settings.artifacts_dir / 'export_report.json'}")
        for target, result in report["onnx"].items():
            print(f"{target}: verified={result['verified']}, max_abs_error={result['max_absolute_error']:.6g}")
    elif args.command == "national-catalog":
        from meteo.national import build_catalog

        catalog = build_catalog(settings)
        print(f"Catalogo: {len(catalog['regions'])} regioni, {len(catalog['provinces'])} province, {len(catalog['points'])} punti.")
    elif args.command == "national-fetch":
        from meteo.national import NationalBudgetExceeded, fetch_national_history, load_catalog

        try:
            fetch_national_history(settings, load_catalog(settings), args.daily_budget)
        except NationalBudgetExceeded as exc:
            parser.exit(2, f"{exc}\n")
    elif args.command == "national-refresh":
        from meteo.national import NationalBudgetExceeded, load_catalog, refresh_national_history

        try:
            refresh_national_history(settings, load_catalog(settings), args.daily_budget)
        except NationalBudgetExceeded as exc:
            parser.exit(2, f"{exc}\n")
    elif args.command == "national-train":
        from meteo.distillation import train_distillation

        report = train_distillation(settings)
        print(f"Teacher e {len(report['territories'])} student territoriali addestrati.")


if __name__ == "__main__":
    main()
