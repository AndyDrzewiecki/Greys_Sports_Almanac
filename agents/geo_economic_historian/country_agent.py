"""Country-level sub-agent for the Geo Economic Historian.

Fetches World Bank indicators and FRED data for a single country,
interprets the economic health, and flags debt/currency/trade risks.

Uses model_tier='extraction' → routes to the EliteBook 8b model.
"""

from __future__ import annotations

import json
from typing import Any

from agents.base_agent import BaseAgent
from data.connectors.world_bank_connector import WorldBankConnector

# FRED series that have international coverage (select, not comprehensive)
FRED_GLOBAL_INDICATORS: dict[str, str] = {
    "DEXUSEU": "usd_eur",
    "DEXJPUS": "usd_jpy",
    "DEXCHUS": "usd_cny",
    "DEXBZUS": "usd_brl",
    "DEXINUS": "usd_inr",
    "DEXUSNB": "usd_nok",
    "DCOILWTICO": "wti_oil_price",
    "GOLDAMGBD228NLBM": "gold_usd",
}


class GeoEconomicCountryAgent(BaseAgent):
    """Analyzes economic conditions for a single country in a given year."""

    def __init__(self, country_code: str, country_name: str) -> None:
        """Initialize the country agent.

        Args:
            country_code: ISO 3166-1 alpha-2 code, e.g. "CN".
            country_name: Full name, e.g. "China".
        """

        super().__init__(
            name=f"geo_econ_{country_code.lower()}",
            kb_path="kb/geo_economic_historian",
        )
        self.country_code = country_code
        self.country_name = country_name
        self.wb = WorldBankConnector()

    def research_month(self, year_month: str) -> dict[str, Any]:
        """Research economic conditions for this country in a given month.

        World Bank data is annual; we use the year from year_month and
        ±1 year to give the LLM trend context.

        Args:
            year_month: "YYYY-MM" format.

        Returns:
            Dict with keys: country, year_month, wb_snapshot, economic_health,
            trade_risks, currency_stress, debt_level, key_concerns,
            market_relevance, context_coverage.
        """

        year = int(year_month[:4])
        snapshot = self.wb.fetch_country_snapshot(self.country_code, year)
        snapshot_prev = self.wb.fetch_country_snapshot(self.country_code, year - 1)

        user_message = json.dumps(
            {
                "country": self.country_name,
                "country_code": self.country_code,
                "year_month": year_month,
                "current_year_indicators": snapshot,
                "prior_year_indicators": snapshot_prev,
                "instruction": (
                    "Analyze the economic health and risks for this country. "
                    "Consider: GDP growth, inflation, debt burden, trade dependency, "
                    "currency stability, and FDI trends. "
                    "Return JSON with keys: economic_health (strong/stable/stressed/crisis), "
                    "trade_risks (list of str), currency_stress (low/medium/high), "
                    "debt_level (low/moderate/high/critical), "
                    "key_concerns (list of str), market_relevance (str), "
                    "contagion_risk (str), export_concentration_risk (str)."
                ),
            },
            default=str,
        )

        raw = self.call_llm(
            system_prompt=(
                f"You are a sovereign credit analyst and macro economist specializing in {self.country_name}. "
                "Interpret World Bank economic data in the context of the global macro cycle. "
                "Your analysis feeds into a portfolio risk model. "
                "Always return valid JSON only."
            ),
            user_message=user_message,
            include_kb=True,
            model_tier="extraction",
        )

        parsed = self.validate_json_output(
            raw,
            required_keys=["economic_health", "key_concerns", "market_relevance"],
        )
        parsed["country"] = self.country_name
        parsed["country_code"] = self.country_code
        parsed["year_month"] = year_month
        parsed["wb_snapshot"] = snapshot
        parsed["context_coverage"] = "full" if snapshot.get("gdp_growth_pct") is not None else "sparse"
        return parsed

    def query_context(self, question: str) -> str:
        """Answer a specific question about this country's economic history.

        Args:
            question: Free-text question.

        Returns:
            String answer.
        """

        return self.call_llm(
            system_prompt=(
                f"You are an expert in {self.country_name} economic history, policy, and markets. "
                "Draw on World Bank data context and your training knowledge. "
                "Be specific about numbers, dates, and causes."
            ),
            user_message=question,
            include_kb=True,
            model_tier="extraction",
        )
