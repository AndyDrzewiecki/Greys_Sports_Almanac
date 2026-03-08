"""Country-level sub-agent for the Geopolitical Historian.

Each instance is scoped to one country.  The parent GeopoliticalHistorian
creates and manages these at runtime.

Uses model_tier='extraction' → routes to the EliteBook 8b model.
"""

from __future__ import annotations

import json
from typing import Any

from agents.base_agent import BaseAgent
from data.connectors.gdelt_connector import GDELTConnector


class GeopoliticalCountryAgent(BaseAgent):
    """Analyzes geopolitical events for a single country in a given month."""

    def __init__(self, country_code: str, country_name: str) -> None:
        """Initialize the country agent.

        Args:
            country_code: ISO 3166-1 alpha-2 code, e.g. "RU".
            country_name: Full name for GDELT queries, e.g. "Russia".
        """

        super().__init__(
            name=f"geo_pol_{country_code.lower()}",
            kb_path="kb/geopolitical_historian",
        )
        self.country_code = country_code
        self.country_name = country_name
        self.gdelt = GDELTConnector()

    def research_month(self, year_month: str) -> dict[str, Any]:
        """Research geopolitical events for this country in one calendar month.

        For pre-2015 dates, GDELT returns no articles.  The agent returns
        a sparse analysis based on general knowledge encoded in the KB.

        Args:
            year_month: "YYYY-MM" format.

        Returns:
            Dict with keys: country, year_month, events, risk_level,
            key_themes, market_relevance, context_coverage.
        """

        coverage = self.gdelt.coverage_tier(year_month)
        events = self.gdelt.get_country_events(self.country_name, year_month)
        events_for_prompt = events[:40] if events else []

        # Build weighted summary from credible sources first
        high_cred = [e for e in events_for_prompt if e.get("source_weight", 0) >= 0.8]
        low_cred = [e for e in events_for_prompt if e.get("source_weight", 0) < 0.8]
        ordered_events = (high_cred + low_cred)[:30]

        user_message = json.dumps(
            {
                "country": self.country_name,
                "country_code": self.country_code,
                "year_month": year_month,
                "coverage_tier": coverage,
                "gdelt_articles": ordered_events,
                "instruction": (
                    "Analyze the geopolitical situation for this country in this month. "
                    "If articles are empty (pre-2015), rely on your historical knowledge. "
                    "Return JSON with keys: events (list of str), risk_level (low/medium/high/critical), "
                    "key_themes (list of str), market_relevance (str), regime_stability (str), "
                    "notable_actors (list of str), sanctions_flags (list of str). "
                    "Be specific and factual."
                ),
            },
            default=str,
        )

        raw = self.call_llm(
            system_prompt=(
                f"You are a geopolitical analyst specializing in {self.country_name}. "
                "You track political risk, regime stability, foreign policy, and military actions. "
                "Your analysis feeds into a macro investment model. "
                "Always return valid JSON only — no markdown, no extra text."
            ),
            user_message=user_message,
            include_kb=True,
            model_tier="extraction",
        )

        parsed = self.validate_json_output(
            raw,
            required_keys=["events", "risk_level", "key_themes", "market_relevance"],
        )
        parsed["country"] = self.country_name
        parsed["country_code"] = self.country_code
        parsed["year_month"] = year_month
        parsed["context_coverage"] = coverage
        return parsed

    def query_context(self, question: str) -> str:
        """Answer a specific question about this country's geopolitical history.

        Args:
            question: Free-text question about historical or current events.

        Returns:
            String answer drawing on KB and LLM knowledge.
        """

        return self.call_llm(
            system_prompt=(
                f"You are an expert in {self.country_name} geopolitics and political history. "
                "Answer based on your knowledge and the KB context provided. "
                "Be specific, cite approximate dates and actors where possible."
            ),
            user_message=question,
            include_kb=True,
            model_tier="extraction",
        )
