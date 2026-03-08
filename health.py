"""System health check for the Macro Market Portfolio Alignment Tool.

Checks all external connections and dependencies before a run.
Exit code 0 = all required services reachable.
Exit code 1 = at least one required service unreachable.

Usage:
    python health.py          # full check
    python health.py --json   # machine-readable JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

import config
from storage.database import health_check, init_db

REQUIRED = {"ollama_primary", "fred", "yfinance", "database"}
OPTIONAL = {"ollama_secondary", "gdelt", "world_bank", "postgresql"}


def check_ollama(url: str, label: str) -> dict:
    """Check if an Ollama instance is reachable and has models."""

    try:
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        models = [m["name"] for m in (resp.json().get("models") or [])]
        return {
            "status": "ok",
            "url": url,
            "label": label,
            "models": models,
            "model_count": len(models),
        }
    except requests.ConnectionError:
        return {"status": "unreachable", "url": url, "label": label, "error": "Connection refused"}
    except Exception as exc:
        return {"status": "error", "url": url, "label": label, "error": str(exc)}


def check_fred() -> dict:
    """Check FRED API connectivity and key validity."""

    if not config.FRED_API_KEY:
        return {"status": "no_key", "message": "FRED_API_KEY not set in .env — live data unavailable."}

    try:
        from fredapi import Fred
        client = Fred(api_key=config.FRED_API_KEY)
        series = client.get_series("SOFR", observation_start="2024-01-01")
        if series is None or series.empty:
            return {"status": "empty", "message": "FRED returned no data for SOFR."}
        return {
            "status": "ok",
            "latest_sofr": float(series.iloc[-1]),
            "latest_date": str(series.index[-1].date()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_yfinance() -> dict:
    """Check yfinance can fetch a ticker."""

    try:
        import yfinance as yf
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period="5d")
        if hist.empty:
            return {"status": "empty", "message": "yfinance returned no data for SPY."}
        return {
            "status": "ok",
            "spy_latest_close": float(hist["Close"].iloc[-1]),
            "latest_date": str(hist.index[-1].date()),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_gdelt() -> dict:
    """Check GDELT Full Text Search API connectivity."""

    try:
        params = {
            "query": "geopolitics",
            "mode": "artlist",
            "maxrecords": 1,
            "format": "json",
        }
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        count = len(data.get("articles") or [])
        return {"status": "ok", "sample_articles_returned": count}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_world_bank() -> dict:
    """Check World Bank API connectivity."""

    try:
        resp = requests.get(
            "https://api.worldbank.org/v2/country/US/indicator/NY.GDP.MKTP.KD.ZG"
            "?date=2022:2023&format=json&per_page=2",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        has_data = bool(data and len(data) > 1 and data[1])
        return {"status": "ok", "sample_data_available": has_data}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_database() -> dict:
    """Check database connectivity and return table row counts."""

    try:
        init_db()
        counts = health_check()
        db_url = config.DB_URL or f"sqlite:///{config.DB_PATH}"
        db_type = "postgresql" if db_url.startswith("postgresql") else "sqlite"
        return {"status": "ok", "db_type": db_type, "table_counts": counts}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_kb_files() -> dict:
    """Verify all expected KB memory.md files exist."""

    from pathlib import Path

    expected = [
        "kb/signal_watcher/memory.md",
        "kb/market_watcher/memory.md",
        "kb/historian/memory.md",
        "kb/forecaster/memory.md",
        "kb/skeptic/memory.md",
        "kb/recommender/memory.md",
        "kb/meta_evaluator/memory.md",
        "kb/geopolitical_historian/memory.md",
        "kb/geo_economic_historian/memory.md",
    ]
    base = config.BASE_DIR
    missing = [p for p in expected if not (base / p).exists()]
    present = [p for p in expected if (base / p).exists()]
    return {
        "status": "ok" if not missing else "warning",
        "present": len(present),
        "missing": missing,
    }


def run_checks() -> dict:
    """Run all health checks and return a structured results dict."""

    t0 = time.time()
    results = {}

    print("Checking Ollama primary (ZBook)…", end=" ", flush=True)
    results["ollama_primary"] = check_ollama(config.OLLAMA_PRIMARY_URL, "ZBook 70b")
    print(results["ollama_primary"]["status"])

    secondary_url = config.OLLAMA_SECONDARY_URL
    is_same = secondary_url == config.OLLAMA_PRIMARY_URL
    print(f"Checking Ollama secondary (EliteBook){' [same as primary]' if is_same else ''}…", end=" ", flush=True)
    results["ollama_secondary"] = check_ollama(secondary_url, "EliteBook 8b")
    print(results["ollama_secondary"]["status"])

    print("Checking FRED API…", end=" ", flush=True)
    results["fred"] = check_fred()
    print(results["fred"]["status"])

    print("Checking yfinance…", end=" ", flush=True)
    results["yfinance"] = check_yfinance()
    print(results["yfinance"]["status"])

    print("Checking GDELT…", end=" ", flush=True)
    results["gdelt"] = check_gdelt()
    print(results["gdelt"]["status"])

    print("Checking World Bank API…", end=" ", flush=True)
    results["world_bank"] = check_world_bank()
    print(results["world_bank"]["status"])

    print("Checking database…", end=" ", flush=True)
    results["database"] = check_database()
    print(results["database"]["status"])

    print("Checking KB files…", end=" ", flush=True)
    results["kb_files"] = check_kb_files()
    print(results["kb_files"]["status"])

    required_failures = [k for k in REQUIRED if results.get(k, {}).get("status") not in {"ok", "no_key"}]
    optional_warnings = [k for k in OPTIONAL if results.get(k, {}).get("status") not in {"ok", "warning"}]
    elapsed = round(time.time() - t0, 2)

    results["_summary"] = {
        "elapsed_sec": elapsed,
        "required_failures": required_failures,
        "optional_warnings": optional_warnings,
        "overall": "PASS" if not required_failures else "FAIL",
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Health check for the Macro Market Portfolio Alignment Tool.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = parser.parse_args()

    print("\n=== Macro Market Portfolio Alignment Tool — Health Check ===\n")
    results = run_checks()
    summary = results["_summary"]

    print(f"\n{'='*50}")
    print(f"Overall: {summary['overall']}  ({summary['elapsed_sec']}s)")
    if summary["required_failures"]:
        print(f"REQUIRED failures: {', '.join(summary['required_failures'])}")
        print("  → Fix these before running the pipeline.")
    if summary["optional_warnings"]:
        print(f"Optional warnings: {', '.join(summary['optional_warnings'])}")
        print("  → Historian agents will degrade gracefully.")
    print(f"{'='*50}\n")

    if args.json:
        print(json.dumps(results, indent=2, default=str))

    sys.exit(0 if summary["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
