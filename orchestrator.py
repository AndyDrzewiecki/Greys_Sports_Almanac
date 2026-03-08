"""Orchestrator — the intelligent brain of the Macro Market Portfolio Alignment Tool.

Responsibilities:
  1. Run the full agent pipeline in correct dependency order.
  2. Enforce the 30-day forecast cadence.
  3. Synthesize all active KB files into cross-agent patterns (synthesize_kbs).
  4. Generate a one-page structured Daily Brief from all agent outputs.
  5. Log everything to the audit trail.
  6. Fall back to last successful output on stage failures.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import schedule

import config
from agents.base_agent import BaseAgent
from agents.forecaster.agent import Forecaster
from agents.geo_economic_historian.agent import GeoEconomicHistorian
from agents.geopolitical_historian.agent import GeopoliticalHistorian
from agents.historian.agent import Historian
from agents.market_watcher.agent import MarketWatcher
from agents.meta_evaluator.agent import MetaEvaluator
from agents.recommender.agent import Recommender
from agents.signal_watcher.agent import SignalWatcher
from agents.skeptic.agent import Skeptic
from data.connectors.fred_connector import FREDConnector
from data.connectors.yfinance_connector import YFinanceConnector
from data.validator import DataValidator
from storage.database import AuditLog, Forecast, get_session, init_db


def log_orchestrator_event(
    event_type: str,
    input_summary: str,
    output_summary: str,
    error: str | None = None,
    data_quality_flags: str | None = None,
) -> None:
    """Write orchestrator events to the audit log."""

    with get_session() as session:
        session.add(
            AuditLog(
                agent_name="orchestrator",
                event_type=event_type,
                input_summary=input_summary,
                output_summary=output_summary,
                data_quality_flags=data_quality_flags,
                error=error,
            )
        )


def load_last_run() -> dict[str, Any]:
    """Load the last successful combined run, if present."""

    if not Path(config.LATEST_RUN_PATH).exists():
        return {}
    return json.loads(Path(config.LATEST_RUN_PATH).read_text(encoding="utf-8"))


def save_latest_run(payload: dict[str, Any]) -> None:
    """Persist the latest combined run JSON payload."""

    Path(config.LATEST_RUN_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(config.LATEST_RUN_PATH).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def precheck_data_sources() -> dict[str, Any]:
    """Run a lightweight data source pre-check before the full pipeline."""

    validator = DataValidator(agent_name="orchestrator")
    fred = FREDConnector()
    yfinance = YFinanceConnector()
    sample_readings: dict[str, Any] = {}

    try:
        sample_readings["SOFR"] = {"data": fred.fetch("SOFR", periods=10), "source": fred.source_name}
    except Exception as exc:
        log_orchestrator_event("precheck_fetch_error", "SOFR", "", error=str(exc), data_quality_flags="true")

    try:
        sample_readings["SPY"] = {"data": yfinance.fetch("SPY", period="1mo"), "source": yfinance.source_name}
    except Exception as exc:
        log_orchestrator_event("precheck_fetch_error", "SPY", "", error=str(exc), data_quality_flags="true")

    result = validator.validate_all(sample_readings)
    severe_failure = len(result["passed"]) == 0
    return {
        "severe_failure": severe_failure,
        "passed": list(result["passed"].keys()),
        "failed": result["failed"],
        "warnings": result["warnings"],
    }


def forecaster_should_run() -> bool:
    """Enforce the 30-day forecast cadence.

    Returns True if no forecast exists or the most recent one is older than
    FORECAST_CADENCE_DAYS.
    """

    with get_session() as session:
        last = (
            session.query(Forecast)
            .order_by(Forecast.timestamp.desc())
            .first()
        )

    if not last:
        return True

    days_since = (datetime.now(timezone.utc) - last.timestamp.replace(tzinfo=timezone.utc)).days
    return days_since >= config.FORECAST_CADENCE_DAYS


def synthesize_kbs() -> str:
    """Read all active KB files and synthesize cross-agent patterns.

    This is the "meta-intelligence" call — the reasoning model reads every
    agent's knowledge base and surfaces contradictions, consensus signals,
    and emerging themes that no single agent would see alone.

    Returns:
        Prose synthesis string for use in the Daily Brief.
    """

    kb_root = Path("kb")
    kb_contents: dict[str, str] = {}

    for md_file in sorted(kb_root.rglob("memory.md")):
        agent_name = md_file.parent.name
        try:
            content = md_file.read_text(encoding="utf-8")
            # Limit per-KB content to avoid token overload; take first 800 chars
            kb_contents[agent_name] = content[:800]
        except OSError:
            pass

    if not kb_contents:
        return "No KB files available for synthesis."

    # Use a lightweight base agent to make the synthesis call
    _synth_agent = BaseAgent("kb_synthesizer", "kb/meta_evaluator")

    user_message = json.dumps(
        {
            "kb_snapshots": kb_contents,
            "instruction": (
                "Read all agent knowledge bases above.  Identify: "
                "(1) consensus signals where 3+ agents agree, "
                "(2) contradictions where agents disagree, "
                "(3) blind spots — risks mentioned by only one agent, "
                "(4) emerging themes appearing for the first time across KBs, "
                "(5) any KB that appears stale or hasn't been updated recently. "
                "Return a structured prose synthesis in 4-6 paragraphs."
            ),
        },
        default=str,
    )

    try:
        synthesis = _synth_agent.call_llm(
            system_prompt=(
                "You are the Chief Intelligence Officer reviewing all research team knowledge bases. "
                "Your synthesis shapes the final investment brief. "
                "Be specific, cite agent names, and flag actionable divergences."
            ),
            user_message=user_message,
            include_kb=False,
            model_tier="reasoning",
        )
        log_orchestrator_event("kb_synthesis_complete", "all_kbs", synthesis[:200])
        return synthesis
    except Exception as exc:
        log_orchestrator_event("kb_synthesis_error", "all_kbs", "", error=str(exc))
        return f"KB synthesis failed: {exc}"


def generate_daily_brief(
    daily_summary: dict[str, Any],
    kb_synthesis: str,
) -> str:
    """Generate a one-page structured Daily Brief from all agent outputs.

    Args:
        daily_summary: The full pipeline output dict.
        kb_synthesis: Output from synthesize_kbs().

    Returns:
        Formatted plain-text brief suitable for display or PDF export.
    """

    signal = daily_summary.get("signal_output") or {}
    market = daily_summary.get("market_output") or {}
    forecast = daily_summary.get("forecast_output") or {}
    geo_pol = daily_summary.get("geo_political_output") or {}
    geo_econ = daily_summary.get("geo_economic_output") or {}
    recs = daily_summary.get("recommendations") or {}
    skeptic = daily_summary.get("skeptic_output") or {}

    _brief_agent = BaseAgent("daily_brief_writer", "kb/meta_evaluator")

    user_message = json.dumps(
        {
            "timestamp": daily_summary.get("timestamp"),
            "macro_regime": signal.get("regime", "unknown"),
            "risk_label": market.get("risk_label", "unknown"),
            "top_scenario": forecast.get("top_scenario", "unknown"),
            "scenario_probabilities": forecast.get("scenarios", {}),
            "top_drivers": forecast.get("top_drivers", []),
            "geopolitical_top_risks": geo_pol.get("top_risks", []),
            "economic_top_risks": geo_econ.get("top_risks", []),
            "recommendations_summary": {
                "etfs": recs.get("etfs", [])[:3],
                "themes": recs.get("themes", [])[:3],
            },
            "skeptic_challenges": skeptic.get("challenges", [])[:3] if isinstance(skeptic, dict) else [],
            "kb_synthesis": kb_synthesis[:600],
            "instruction": (
                "Write a concise one-page macro investment daily brief. "
                "Structure: "
                "MACRO REGIME | RISK POSTURE | TOP SCENARIO | "
                "KEY GEOPOLITICAL RISKS | KEY ECONOMIC RISKS | "
                "PORTFOLIO POSITIONING | SKEPTIC CHALLENGES | "
                "CROSS-AGENT INTELLIGENCE | ACTION ITEMS. "
                "Keep each section to 2-4 bullet points. "
                "Tone: senior PM morning memo."
            ),
        },
        default=str,
    )

    try:
        brief = _brief_agent.call_llm(
            system_prompt=(
                "You are a macro investment strategist writing a daily portfolio brief. "
                "Be concise, specific, and actionable. "
                "No fluff. Every sentence must inform a decision."
            ),
            user_message=user_message,
            include_kb=False,
            model_tier="reasoning",
        )
        log_orchestrator_event("daily_brief_generated", "all_outputs", brief[:200])
        return brief
    except Exception as exc:
        log_orchestrator_event("daily_brief_error", "all_outputs", "", error=str(exc))
        return f"Daily brief generation failed: {exc}"


def run_once() -> dict[str, Any]:
    """Run the full pipeline once, using fallbacks when intermediate agents fail."""

    init_db()
    last_run = load_last_run()
    timestamp = datetime.now(timezone.utc).isoformat()

    daily_summary: dict[str, Any] = {
        "timestamp": timestamp,
        "data_quality_report": {},
        "signal_output": None,
        "market_output": None,
        "historian_output": None,
        "geo_political_output": None,
        "geo_economic_output": None,
        "forecast_output": None,
        "skeptic_output": None,
        "recommendations": None,
        "meta_evaluation": None,
        "kb_synthesis": None,
        "daily_brief": None,
    }

    precheck = precheck_data_sources()
    daily_summary["data_quality_report"] = precheck
    if precheck["severe_failure"]:
        regime = (last_run.get("signal_output") or {}).get("regime", "unknown")
        daily_summary["signal_output"] = {
            "timestamp": timestamp,
            "regime": regime,
            "regime_summary": "Pipeline aborted during pre-check because data sources failed validation.",
            "indicators": {},
            "top_anomalies": [],
            "data_quality_warnings": precheck["warnings"],
        }
        log_orchestrator_event(
            "pipeline_aborted",
            "precheck_data_sources",
            json.dumps({"regime": regime}),
            error="DataValidator pre-check failed.",
            data_quality_flags=json.dumps(precheck),
        )
        save_latest_run(daily_summary)
        return daily_summary

    # ── Stage 1: Macro signal and market risk ─────────────────────────────────
    signal_output = _run_stage("signal_output", lambda: SignalWatcher().run(), last_run, [])
    daily_summary["signal_output"] = signal_output

    market_output = _run_stage("market_output", lambda: MarketWatcher().run(), last_run, [])
    daily_summary["market_output"] = market_output

    # ── Stage 2: Historian (analog matching) ─────────────────────────────────
    if signal_output and market_output:
        historian_output = _run_stage(
            "historian_output",
            lambda: Historian().run(signal_output, market_output),
            last_run,
            ["signal_output", "market_output"],
        )
    else:
        historian_output = _skip_stage("historian_output", last_run, "Missing signal or market output.")
    daily_summary["historian_output"] = historian_output

    # ── Stage 3: Geopolitical and Geo-Economic historians ────────────────────
    if signal_output and market_output:
        geo_political_output = _run_stage(
            "geo_political_output",
            lambda: GeopoliticalHistorian().run(signal_output, market_output),
            last_run,
            ["signal_output", "market_output"],
        )
    else:
        geo_political_output = _skip_stage("geo_political_output", last_run, "Missing macro inputs.")
    daily_summary["geo_political_output"] = geo_political_output

    if signal_output and market_output:
        geo_economic_output = _run_stage(
            "geo_economic_output",
            lambda: GeoEconomicHistorian().run(signal_output, market_output),
            last_run,
            ["signal_output", "market_output"],
        )
    else:
        geo_economic_output = _skip_stage("geo_economic_output", last_run, "Missing macro inputs.")
    daily_summary["geo_economic_output"] = geo_economic_output

    # ── Stage 4: Forecaster (cadence-gated) ──────────────────────────────────
    if forecaster_should_run() and signal_output and market_output and historian_output:
        # Merge geo context into historian_output for the Forecaster prompt
        enriched_historian = dict(historian_output) if historian_output else {}
        if geo_political_output:
            enriched_historian["geopolitical_context"] = geo_political_output.get("synthesis", {}).get("summary", "")
            enriched_historian["geopolitical_top_risks"] = geo_political_output.get("top_risks", [])
        if geo_economic_output:
            enriched_historian["economic_context"] = geo_economic_output.get("synthesis", {}).get("summary", "")
            enriched_historian["economic_top_risks"] = geo_economic_output.get("top_risks", [])

        forecast_output = _run_stage(
            "forecast_output",
            lambda: Forecaster().run(signal_output, market_output, enriched_historian),
            last_run,
            ["signal_output", "market_output", "historian_output"],
        )
    elif not forecaster_should_run():
        forecast_output = last_run.get("forecast_output")
        log_orchestrator_event(
            "forecast_cadence_skip",
            "forecaster",
            "Forecast skipped — cadence not reached.",
        )
    else:
        forecast_output = _skip_stage("forecast_output", last_run, "Missing upstream context.")
    daily_summary["forecast_output"] = forecast_output

    # ── Stage 5: Skeptic ─────────────────────────────────────────────────────
    if forecast_output and signal_output and market_output:
        skeptic_output = _run_stage(
            "skeptic_output",
            lambda: Skeptic().run(forecast_output, signal_output, market_output),
            last_run,
            ["forecast_output", "signal_output", "market_output"],
        )
    else:
        skeptic_output = _skip_stage("skeptic_output", last_run, "Missing forecast or market context.")
    daily_summary["skeptic_output"] = skeptic_output

    # ── Stage 6: Recommender ──────────────────────────────────────────────────
    if forecast_output and signal_output and market_output and skeptic_output:
        recommendations = _run_stage(
            "recommendations",
            lambda: Recommender().run(
                forecast_output, signal_output, market_output, skeptic_output,
                geo_political_output=geo_political_output,
                geo_economic_output=geo_economic_output,
            ),
            last_run,
            ["forecast_output", "signal_output", "market_output", "skeptic_output"],
        )
    else:
        recommendations = _skip_stage("recommendations", last_run, "Missing skeptic-reviewed forecast.")
    daily_summary["recommendations"] = recommendations

    # ── Stage 7: Weekly meta-evaluation ───────────────────────────────────────
    if datetime.now().strftime("%A") == config.CADENCE["weekly_day"]:
        daily_summary["meta_evaluation"] = _run_stage("meta_evaluation", lambda: MetaEvaluator().run(), last_run, [])

    # ── Stage 8: KB synthesis and Daily Brief (reasoning model) ───────────────
    kb_synthesis = synthesize_kbs()
    daily_summary["kb_synthesis"] = kb_synthesis

    daily_brief = generate_daily_brief(daily_summary, kb_synthesis)
    daily_summary["daily_brief"] = daily_brief

    # ── Persist ───────────────────────────────────────────────────────────────
    save_latest_run(daily_summary)
    log_orchestrator_event(
        "pipeline_run_success",
        input_summary="full_pipeline",
        output_summary=json.dumps(
            {
                "regime": (signal_output or {}).get("regime"),
                "risk_label": (market_output or {}).get("risk_label"),
                "top_scenario": (forecast_output or {}).get("top_scenario"),
                "geo_regime": (geo_political_output or {}).get("synthesis", {}).get("geopolitical_regime"),
            },
            default=str,
        ),
    )
    return daily_summary


def _run_stage(
    key: str,
    runner: Any,
    last_run: dict[str, Any],
    dependencies: list[str],
) -> dict[str, Any] | None:
    """Run one pipeline stage with error handling and fallback to last successful output."""

    try:
        result = runner()
        log_orchestrator_event("stage_success", key, "fresh_output")
        return result
    except Exception as exc:
        fallback = last_run.get(key)
        if fallback is not None:
            log_orchestrator_event(
                "stage_fallback",
                key,
                "using_last_successful_output",
                error=str(exc),
                data_quality_flags=json.dumps({"dependencies": dependencies}),
            )
            return fallback
        log_orchestrator_event(
            "stage_failure",
            key,
            "no_fallback_available",
            error=str(exc),
            data_quality_flags=json.dumps({"dependencies": dependencies}),
        )
        return None


def _skip_stage(key: str, last_run: dict[str, Any], reason: str) -> dict[str, Any] | None:
    """Skip a stage and fall back to last successful output when possible."""

    fallback = last_run.get(key)
    log_orchestrator_event("stage_skipped", key, "skipped", error=reason)
    return fallback


def run_scheduled() -> None:
    """Run the orchestrator daily at the configured hour."""

    hour = int(config.CADENCE["daily_hour"])
    schedule.every().day.at(f"{hour:02d}:00").do(run_once)
    log_orchestrator_event("scheduler_started", "run_scheduled", f"daily at {hour:02d}:00")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Macro Market Portfolio Alignment pipeline.")
    parser.add_argument("--once", action="store_true", help="Run the pipeline once immediately.")
    parser.add_argument("--scheduled", action="store_true", help="Run the daily scheduler loop.")
    args = parser.parse_args()

    if args.scheduled:
        run_scheduled()
    else:
        print(json.dumps(run_once(), indent=2, default=str))
