"""Database models and helpers for the Macro Market Portfolio Alignment Tool.

Supports both SQLite (default, local dev) and PostgreSQL (production, Mini PC).
Set DB_URL in .env to switch:
    DB_URL=postgresql://mwagent:password@192.168.1.xxx/market_watch
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator

from sqlalchemy import (
    Boolean, DateTime, Float, Index, Integer, JSON,
    String, Text, UniqueConstraint, create_engine, func, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

import config


class Base(DeclarativeBase):
    """Base SQLAlchemy declarative model."""


# ── Existing tables (unchanged schema) ───────────────────────────────────────

class SignalReading(Base):
    """Stores normalized macro and market stress indicator readings."""

    __tablename__ = "signal_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    indicator_name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    pct_change_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_change_20d: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_change_90d: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_flag: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    data_quality_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class MarketReading(Base):
    """Stores market price snapshots and derived risk metrics."""

    __tablename__ = "market_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    pct_change_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    pct_change_5d: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_on_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime_contribution: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)


class Forecast(Base):
    """Stores scenario probability outputs and later scoring metadata."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    agent_version: Mapped[str] = mapped_column(String(64), nullable=False)
    scenarios_json: Mapped[str] = mapped_column(Text, nullable=False)
    top_drivers_json: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    invalidations_json: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class Recommendation(Base):
    """Stores ETF, stock, and macro recommendation outputs and realized outcomes."""

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    rec_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ticker_or_theme: Mapped[str] = mapped_column(String(128), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    scenario_context: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    skeptic_approved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    outcome_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_90d: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)


class AuditLog(Base):
    """Stores audit trail events, errors, and data quality messages."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    input_summary: Mapped[str] = mapped_column(Text, nullable=False)
    output_summary: Mapped[str] = mapped_column(Text, nullable=False)
    data_quality_flags: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class KBUpdate(Base):
    """Stores KB change requests and their review status."""

    __tablename__ = "kb_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False)
    update_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


# ── New tables for historian and backtesting ──────────────────────────────────

class HistoricalContext(Base):
    """Cache for historian research results, keyed by month + context type + country.

    Avoids re-fetching expensive GDELT/World Bank data on repeated runs.
    context_type: 'political' | 'economic'
    context_coverage: 'full' (2015+) | 'partial' (1979-2015) | 'sparse' (pre-1979)
    """

    __tablename__ = "historical_context"
    __table_args__ = (
        UniqueConstraint("year_month", "context_type", "country", name="uq_historical_context"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year_month: Mapped[str] = mapped_column(String(7), nullable=False)    # "1972-03"
    context_type: Mapped[str] = mapped_column(String(50), nullable=False) # "political" | "economic"
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_events: Mapped[Any] = mapped_column(JSON, nullable=True)
    source_weight_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_coverage: Mapped[str] = mapped_column(String(20), nullable=False, default="partial")
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BacktestRun(Base):
    """Stores one backtesting run per month — the model's forecast given only historical data."""

    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year_month: Mapped[str] = mapped_column(String(7), nullable=False, unique=True)
    fred_snapshot: Mapped[Any] = mapped_column(JSON, nullable=True)
    political_context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    economic_context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    portfolio_recommendation: Mapped[Any] = mapped_column(JSON, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    context_coverage: Mapped[str] = mapped_column(String(20), nullable=False, default="partial")
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BacktestScore(Base):
    """Scores a backtest run against actual market data 10 years later.

    Filled in once `run_at.year + 10` <= today.
    """

    __tablename__ = "backtest_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)   # FK → backtest_runs.id
    score_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_10yr_data: Mapped[Any] = mapped_column(JSON, nullable=True)
    return_vs_spy: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    wins: Mapped[Any] = mapped_column(JSON, nullable=True)
    losses: Mapped[Any] = mapped_column(JSON, nullable=True)
    root_cause_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    missing_signals: Mapped[str | None] = mapped_column(Text, nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class BacktestLesson(Base):
    """Accumulated wisdom from backtest wins and losses.

    Each lesson captures which signal combination produced a good or bad call
    and what additional signals would have improved it.
    confidence: 'provisional' (1-2 examples) | 'established' (3+ examples)
    """

    __tablename__ = "backtest_lessons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lesson_type: Mapped[str] = mapped_column(String(20), nullable=False)     # "win" | "loss"
    era: Mapped[str] = mapped_column(String(50), nullable=False)             # "Cold War" | "Post-Soviet" | etc.
    signal_combination: Mapped[Any] = mapped_column(JSON, nullable=True)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    root_cause: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_new_signals: Mapped[str | None] = mapped_column(Text, nullable=True)
    applicable_scenarios: Mapped[Any] = mapped_column(JSON, nullable=True)
    confidence: Mapped[str] = mapped_column(String(20), nullable=False, default="provisional")
    validation_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Conspiracy Theorist tables ────────────────────────────────────────────────

class ConspiracyFinding(Base):
    """Stores anomaly findings from the Conspiracy Theorist agent.

    finding_type: 'financial_anomaly' | 'narrative_gap' | 'money_flow' | 'document_link'
    status: 'flagged' | 'reviewed' | 'confirmed' | 'dismissed'
    """

    __tablename__ = "conspiracy_findings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    finding_type: Mapped[str] = mapped_column(String(64), nullable=False)
    year_month: Mapped[str] = mapped_column(String(7), nullable=False)           # "YYYY-MM"
    countries_involved: Mapped[Any] = mapped_column(JSON, nullable=True)
    source_documents: Mapped[Any] = mapped_column(JSON, nullable=True)           # Panama Papers, Epstein, etc.
    anomaly_description: Mapped[str] = mapped_column(Text, nullable=False)
    market_event_within_6mo: Mapped[str | None] = mapped_column(Text, nullable=True)
    financial_flow_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    economist_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    political_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="flagged")
    corroborated_by: Mapped[Any] = mapped_column(JSON, nullable=True)            # other agents that agree
    forecaster_impact: Mapped[str | None] = mapped_column(Text, nullable=True)  # how this changes forecast


class CTCountryAnalysis(Base):
    """Cache for Conspiracy Theorist per-country 'follow the money' analysis.

    Follows same caching pattern as HistoricalContext.
    """

    __tablename__ = "ct_country_analyses"
    __table_args__ = (
        UniqueConstraint("year_month", "country", name="uq_ct_country_analysis"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    year_month: Mapped[str] = mapped_column(String(7), nullable=False)
    country: Mapped[str] = mapped_column(String(10), nullable=False)
    financial_flows: Mapped[Any] = mapped_column(JSON, nullable=True)           # tracked money movements
    anomalies: Mapped[Any] = mapped_column(JSON, nullable=True)                 # flagged events
    official_narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    documented_reality: Mapped[str | None] = mapped_column(Text, nullable=True) # what was later revealed
    key_actors: Mapped[Any] = mapped_column(JSON, nullable=True)
    market_correlation: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_weight_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LivePortfolioRecommendation(Base):
    """Daily portfolio balance recommendations with 1/5/7/10 year self-review tracking.

    Separate from backtest_runs. This is the live self-review database.
    data_source: 'live' | 'ct_secondary' (CT agent's alternative theory)
    """

    __tablename__ = "live_portfolio_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    year_month: Mapped[str] = mapped_column(String(7), nullable=False)
    portfolio_json: Mapped[Any] = mapped_column(JSON, nullable=False)           # {sector: weight, ...}
    macro_thesis: Mapped[str] = mapped_column(Text, nullable=False)
    top_scenario: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ct_flags_json: Mapped[Any] = mapped_column(JSON, nullable=True)             # CT anomalies considered
    data_source: Mapped[str] = mapped_column(String(32), nullable=False, default="live")
    # Outcome tracking — filled in asynchronously at each horizon
    return_1yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_5yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_7yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    return_10yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_spy_1yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_spy_5yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_spy_7yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    vs_spy_10yr: Mapped[float | None] = mapped_column(Float, nullable=True)
    scored_1yr_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scored_5yr_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scored_7yr_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scored_10yr_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BacktestScoreHorizon(Base):
    """Multi-horizon scoring for backtest runs: 1, 5, 7, and 10 years.

    Supplements BacktestScore with finer-grained horizon tracking.
    horizon_years: 1 | 5 | 7 | 10
    """

    __tablename__ = "backtest_score_horizons"
    __table_args__ = (
        UniqueConstraint("run_id", "horizon_years", name="uq_backtest_horizon"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, nullable=False)                # FK → backtest_runs.id
    horizon_years: Mapped[int] = mapped_column(Integer, nullable=False)
    score_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_data: Mapped[Any] = mapped_column(JSON, nullable=True)
    return_vs_spy: Mapped[float | None] = mapped_column(Float, nullable=True)
    brier_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    wins: Mapped[Any] = mapped_column(JSON, nullable=True)
    losses: Mapped[Any] = mapped_column(JSON, nullable=True)
    root_cause_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    ct_anomaly_flags: Mapped[Any] = mapped_column(JSON, nullable=True)          # CT findings for this period
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── Engine setup ──────────────────────────────────────────────────────────────

def _db_url() -> str:
    """Return the database URL.

    Priority: DB_URL env var (PostgreSQL) → SQLite local file.
    """

    if config.DB_URL:
        return config.DB_URL
    db_path = Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def _make_engine():
    """Create a SQLAlchemy engine appropriate for the configured DB."""

    url = _db_url()
    if url.startswith("postgresql"):
        return create_engine(url, future=True, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return create_engine(url, future=True, connect_args={"check_same_thread": False})


ENGINE = _make_engine()
SessionLocal = sessionmaker(bind=ENGINE, class_=Session, expire_on_commit=False)


def init_db() -> None:
    """Create all database tables if they do not already exist."""

    if not _db_url().startswith("postgresql"):
        Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(ENGINE)


@contextmanager
def get_session() -> Iterator[Session]:
    """Yield a database session with commit/rollback handling."""

    init_db()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def health_check() -> Dict[str, int]:
    """Return a row count per table as a lightweight health check."""

    init_db()
    table_models = {
        "signal_readings": SignalReading,
        "market_readings": MarketReading,
        "forecasts": Forecast,
        "recommendations": Recommendation,
        "audit_log": AuditLog,
        "kb_updates": KBUpdate,
        "historical_context": HistoricalContext,
        "backtest_runs": BacktestRun,
        "backtest_scores": BacktestScore,
        "backtest_lessons": BacktestLesson,
        "conspiracy_findings": ConspiracyFinding,
        "ct_country_analyses": CTCountryAnalysis,
        "live_portfolio_recommendations": LivePortfolioRecommendation,
        "backtest_score_horizons": BacktestScoreHorizon,
    }
    counts: Dict[str, int] = {}
    with get_session() as session:
        for name, model in table_models.items():
            counts[name] = session.scalar(select(func.count()).select_from(model)) or 0
    return counts
