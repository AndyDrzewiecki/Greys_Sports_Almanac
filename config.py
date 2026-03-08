"""Runtime configuration for the Macro Market Portfolio Alignment Tool."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── Database ──────────────────────────────────────────────────────────────────
# SQLite by default; override with DB_URL in .env for PostgreSQL.
# PostgreSQL example: postgresql://mwagent:password@192.168.1.xxx/market_watch
DB_URL = os.getenv("DB_URL", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Kept for optional fallback — not used by default in this build.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Ollama LLM endpoints ──────────────────────────────────────────────────────
# OLLAMA_PRIMARY_URL   → ZBook 17 (70b model, heavy reasoning)
# OLLAMA_SECONDARY_URL → EliteBook 840 G2 (8b model, historian/extraction)
OLLAMA_PRIMARY_URL = os.getenv("OLLAMA_PRIMARY_URL", "http://localhost:11434")
OLLAMA_SECONDARY_URL = os.getenv("OLLAMA_SECONDARY_URL", "http://localhost:11434")

# Model names pulled from Ollama on each machine.
# Override via .env when you have larger models available.
OLLAMA_MODELS: dict[str, str] = {
    "reasoning": os.getenv("OLLAMA_REASONING_MODEL", "llama3.1:8b"),    # 70b on ZBook
    "extraction": os.getenv("OLLAMA_EXTRACTION_MODEL", "llama3.1:8b"),  # 8b on EliteBook
    "fast": os.getenv("OLLAMA_FAST_MODEL", "llama3.1:8b"),
}

# Which model tier routes to which Ollama URL.
# "reasoning" → primary (ZBook 70b), everything else → secondary (EliteBook 8b).
OLLAMA_TIER_ROUTING: dict[str, str] = {
    "reasoning": OLLAMA_PRIMARY_URL,
    "extraction": OLLAMA_SECONDARY_URL,
    "fast": OLLAMA_SECONDARY_URL,
}

# Legacy alias — used by any code that still imports MODEL directly.
MODEL = OLLAMA_MODELS["reasoning"]

# ── Forecaster cadence ────────────────────────────────────────────────────────
# The Forecaster runs in 30-day increments, not daily.
# The orchestrator checks this before invoking the forecaster.
FORECAST_CADENCE_DAYS = 30

# ── Regime thresholds ────────────────────────────────────────────────────────
REGIME_THRESHOLDS = {
    "hy_spread_stress": 400,
    "hy_spread_dislocated": 600,
    "move_index_stress": 120,
    "move_index_dislocated": 150,
    "vix_stress": 25,
    "vix_dislocated": 35,
}

# ── Scheduling ────────────────────────────────────────────────────────────────
CADENCE = {
    "intraday_interval_minutes": 60,
    "daily_hour": 18,
    "weekly_day": "Sunday",
}

DATA_SOURCES = ["FRED", "YFINANCE"]

# ── File paths ────────────────────────────────────────────────────────────────
DB_PATH = BASE_DIR / "storage" / "market_watch.db"
LATEST_RUN_PATH = BASE_DIR / "storage" / "latest_run.json"

# ── Scenarios ────────────────────────────────────────────────────────────────
FORECAST_SCENARIOS = [
    "null_hypothesis",
    "soft_landing",
    "stagflation_drift",
    "credit_accident",
    "energy_shock",
    "treasury_plumbing_crisis",
    "dollar_down_rotation",
]

# ── News source credibility weights ──────────────────────────────────────────
# Higher = more weight in historian context. Add new domains as discovered.
SOURCE_WEIGHTS: dict[str, float] = {
    # Tier 1 — Investigative / independent
    "ICIJ": 1.0,
    "icij.org": 1.0,
    "ProPublica": 0.95,
    "propublica.org": 0.95,
    "theintercept.com": 0.90,
    "OCCRP": 0.95,
    "occrp.org": 0.95,
    "bellingcat.com": 0.90,
    # Tier 2 — Quality international
    "reuters.com": 0.80,
    "apnews.com": 0.80,
    "bbc.com": 0.75,
    "bbc.co.uk": 0.75,
    "ft.com": 0.80,
    "economist.com": 0.80,
    "bloomberg.com": 0.78,
    "wsj.com": 0.75,
    # Tier 3 — Mainstream (use for corroboration)
    "cnn.com": 0.55,
    "foxnews.com": 0.50,
    "msnbc.com": 0.55,
    "nytimes.com": 0.65,
    "washingtonpost.com": 0.65,
    # Tier 4 — Official (factual data, self-serving narrative)
    "federalreserve.gov": 0.70,
    "treasury.gov": 0.70,
    "imf.org": 0.75,
    "worldbank.org": 0.75,
    "un.org": 0.72,
}

DEFAULT_SOURCE_WEIGHT = 0.60

# ── Conspiracy Theorist configuration ─────────────────────────────────────────
# Countries monitored by the CT agent (12 base + 4 extended)
CT_COUNTRIES = [
    "US", "CN", "RU", "DE", "JP", "SA", "IR", "VE", "BR", "IN", "GB", "FR",
    "IL", "UA", "MX", "NG",
]

# CT source credibility — investigative-first hierarchy
CT_SOURCE_WEIGHTS: dict[str, float] = {
    "ICIJ": 1.0,
    "icij.org": 1.0,
    "OCCRP": 0.97,
    "occrp.org": 0.97,
    "ProPublica": 0.97,
    "propublica.org": 0.97,
    "The Intercept": 0.95,
    "theintercept.com": 0.95,
    "Bellingcat": 0.93,
    "bellingcat.com": 0.93,
    "Forbidden Stories": 0.93,
    "declassified.uk": 0.90,
    "opendemocracy.net": 0.88,
    "reuters.com": 0.82,
    "apnews.com": 0.82,
    "ft.com": 0.80,
    "bloomberg.com": 0.78,
    "wsj.com": 0.75,
    "nytimes.com": 0.70,
    "washingtonpost.com": 0.70,
    "guardian.com": 0.72,
    "bbc.com": 0.70,
    "cnn.com": 0.50,
    "foxnews.com": 0.45,
    "government_statement": 0.40,   # always cross-check
}

CT_DEFAULT_SOURCE_WEIGHT = 0.55     # more skeptical default than general agents

# Minimum confidence threshold to persist a CT finding
CT_FINDING_CONFIDENCE_THRESHOLD = 0.55

# Minimum anomaly level to include CT output in Orchestrator synthesis
CT_ANOMALY_LEVEL_THRESHOLD = "moderate"  # "low" | "moderate" | "elevated" | "high"

# CT agent cadence — runs every pipeline cycle but deep country analysis cached monthly
CT_CADENCE_DAYS = 30
