"""Geopolitical Historian agent — parent orchestrator over country sub-agents.

Coordinates per-country research, synthesizes cross-country dynamics,
and caches results to the historical_context table.

Synthesis uses model_tier='reasoning' → routes to the ZBook 70b model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agents.base_agent import BaseAgent
from agents.geopolitical_historian.country_agent import GeopoliticalCountryAgent
from storage.database import HistoricalContext, get_session

# Countries to track: (country_code, full_name_for_GDELT)
TRACKED_COUNTRIES: list[tuple[str, str]] = [
    ("US", "United States"),
    ("CN", "China"),
    ("RU", "Russia"),
    ("DE", "Germany"),
    ("JP", "Japan"),
    ("SA", "Saudi Arabia"),
    ("IR", "Iran"),
    ("VE", "Venezuela"),
    ("BR", "Brazil"),
    ("IN", "India"),
    ("GB", "United Kingdom"),
    ("FR", "France"),
]


class GeopoliticalHistorian(BaseAgent):
    """Synthesizes geopolitical context across all tracked countries."""

    def __init__(self) -> None:
        super().__init__("geopolitical_historian", "kb/geopolitical_historian")
        self.country_agents: dict[str, GeopoliticalCountryAgent] = {
            code: GeopoliticalCountryAgent(code, name)
            for code, name in TRACKED_COUNTRIES
        }

    def run(
        self,
        signal_output: dict[str, Any],
        market_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Run current-month geopolitical analysis.

        Uses the current calendar month.  Results are cached and used
        by the Forecaster and Backtesting Engine.

        Args:
            signal_output: Output from SignalWatcher.run().
            market_output: Output from MarketWatcher.run().

        Returns:
            Dict with keys: year_month, country_analyses, synthesis,
            top_risks, market_impact_assessment.
        """

        year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        return self.run_month(year_month, macro_context={"signal": signal_output, "market": market_output})

    def run_month(
        self,
        year_month: str,
        macro_context: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Run geopolitical analysis for a specific month.

        Checks the DB cache first.  Skips countries that are already cached
        unless force_refresh is True.

        Args:
            year_month: "YYYY-MM" format.
            macro_context: Optional macro regime/market data to include in synthesis.
            force_refresh: Re-fetch even when cached.

        Returns:
            Dict with country_analyses, synthesis, top_risks, market_impact_assessment,
            context_coverage.
        """

        country_analyses: dict[str, Any] = {}
        for code, name in TRACKED_COUNTRIES:
            cached = self._load_cached(year_month, code)
            if cached and not force_refresh:
                country_analyses[code] = cached
                continue
            try:
                agent = self.country_agents[code]
                analysis = agent.research_month(year_month)
                country_analyses[code] = analysis
                self._save_cached(year_month, code, analysis)
            except Exception as exc:
                self.log_audit(
                    "country_research_error",
                    f"{code} {year_month}",
                    str(exc),
                    error=str(exc),
                )
                country_analyses[code] = {"country_code": code, "error": str(exc), "context_coverage": "none"}

        synthesis = self._synthesize(year_month, country_analyses, macro_context or {}, country_analyses)
        output = {
            "year_month": year_month,
            "country_analyses": country_analyses,
            "synthesis": synthesis,
            "top_risks": synthesis.get("top_risks", []),
            "market_impact_assessment": synthesis.get("market_impact_assessment", ""),
            "context_coverage": self._overall_coverage(country_analyses),
        }

        self.write_kb_staging(
            note=f"Geopolitical synthesis for {year_month}: top_risks={synthesis.get('top_risks', [])}",
            category="monthly_synthesis",
        )
        self.log_audit(
            "geopolitical_historian_run",
            f"month={year_month}",
            f"countries={len(country_analyses)} top_risks={synthesis.get('top_risks', [])}",
        )
        return output

    def get_monthly_summary(self, year_month: str) -> str:
        """Return a cached summary string for a month, or trigger a fresh run.

        Args:
            year_month: "YYYY-MM" format.

        Returns:
            Prose summary string for use in prompts.
        """

        with get_session() as session:
            row = (
                session.query(HistoricalContext)
                .filter_by(year_month=year_month, context_type="political", country="ALL")
                .first()
            )
        if row:
            return row.summary
        result = self.run_month(year_month)
        return result.get("synthesis", {}).get("summary", "No summary available.")

    def query(self, question: str, context: dict[str, Any] | None = None) -> str:
        """Answer a free-form geopolitical question using the KB and LLM.

        Args:
            question: Natural-language question.
            context: Optional additional context dict.

        Returns:
            String answer.
        """

        user_msg = question
        if context:
            user_msg = f"{question}\n\nAdditional context: {json.dumps(context, default=str)}"

        return self.call_llm(
            system_prompt=(
                "You are a senior geopolitical analyst with 40 years of experience. "
                "Use your knowledge base and training data to answer the question. "
                "Focus on specific historical facts, actors, dates, and market implications."
            ),
            user_message=user_msg,
            include_kb=True,
            model_tier="reasoning",
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _synthesize(
        self,
        year_month: str,
        country_analyses: dict[str, Any],
        macro_context: dict[str, Any],
        all_analyses: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synthesize cross-country analysis into a single geopolitical picture."""

        user_message = json.dumps(
            {
                "year_month": year_month,
                "country_analyses": country_analyses,
                "macro_context": macro_context,
                "instruction": (
                    "Synthesize the geopolitical situation across all countries. "
                    "Identify cross-country tensions, alliances, and contagion risks. "
                    "Return JSON with keys: summary (str), top_risks (list of str), "
                    "market_impact_assessment (str), geopolitical_regime (str: stable/tense/crisis), "
                    "hotspots (list of str), safe_haven_drivers (list of str), "
                    "scenario_modifiers (dict mapping scenario names to probability adjustments)."
                ),
            },
            default=str,
        )

        raw = self.call_llm(
            system_prompt=(
                "You are the lead geopolitical strategist for a macro investment firm. "
                "Synthesize country-level research into actionable investment signals. "
                "Return valid JSON only."
            ),
            user_message=user_message,
            include_kb=True,
            model_tier="reasoning",
        )

        parsed = self.validate_json_output(
            raw,
            required_keys=["summary", "top_risks", "market_impact_assessment"],
        )

        self._save_synthesis(year_month, parsed, all_analyses or country_analyses)
        return parsed

    def _save_cached(self, year_month: str, country_code: str, analysis: dict[str, Any]) -> None:
        """Upsert a country analysis into the historical_context table."""

        summary = analysis.get("market_relevance", "") or str(analysis.get("key_themes", []))
        coverage = analysis.get("context_coverage", "partial")
        with get_session() as session:
            existing = (
                session.query(HistoricalContext)
                .filter_by(year_month=year_month, context_type="political", country=country_code)
                .first()
            )
            if existing:
                existing.summary = summary
                existing.key_events = analysis.get("events")
                existing.context_coverage = coverage
            else:
                session.add(HistoricalContext(
                    year_month=year_month,
                    context_type="political",
                    country=country_code,
                    summary=summary,
                    key_events=analysis.get("events"),
                    context_coverage=coverage,
                ))

    def _save_synthesis(
        self,
        year_month: str,
        synthesis: dict[str, Any],
        country_analyses: dict[str, Any],
    ) -> None:
        """Upsert the overall synthesis to the historical_context table."""

        summary = synthesis.get("summary", "")
        coverage = self._overall_coverage(country_analyses)
        with get_session() as session:
            existing = (
                session.query(HistoricalContext)
                .filter_by(year_month=year_month, context_type="political", country="ALL")
                .first()
            )
            if existing:
                existing.summary = summary
                existing.key_events = synthesis.get("top_risks")
                existing.context_coverage = coverage
            else:
                session.add(HistoricalContext(
                    year_month=year_month,
                    context_type="political",
                    country="ALL",
                    summary=summary,
                    key_events=synthesis.get("top_risks"),
                    context_coverage=coverage,
                ))

    def _load_cached(self, year_month: str, country_code: str) -> dict[str, Any] | None:
        """Load a cached country analysis from the DB."""

        with get_session() as session:
            row = (
                session.query(HistoricalContext)
                .filter_by(year_month=year_month, context_type="political", country=country_code)
                .first()
            )
        if row is None:
            return None
        return {
            "summary": row.summary,
            "events": row.key_events or [],
            "context_coverage": row.context_coverage,
            "country_code": country_code,
        }

    @staticmethod
    def _overall_coverage(country_analyses: dict[str, Any]) -> str:
        """Determine the minimum coverage tier across all countries."""

        tiers = [v.get("context_coverage", "partial") for v in country_analyses.values()]
        if "full" in tiers and not any(t == "sparse" for t in tiers):
            return "full"
        if "sparse" in tiers:
            return "sparse"
        return "partial"
