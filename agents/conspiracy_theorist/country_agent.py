"""CTCountryAgent — per-country 'follow the money' analysis agent.

Each instance tracks financial flows, anomalies, and the gap between official
narratives and documented reality for a single country. Results are cached in
the ct_country_analyses table to avoid redundant LLM calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import config
from agents.base_agent import BaseAgent
from storage.database import CTCountryAnalysis, get_session


class CTCountryAgent(BaseAgent):
    """Tracks financial flows and narrative gaps for a single country."""

    # Source credibility weights — higher weight = more trustworthy
    SOURCE_TIERS = {
        "ICIJ": 1.0,
        "ProPublica": 0.97,
        "The Intercept": 0.95,
        "Bellingcat": 0.93,
        "Organized Crime and Corruption Reporting Project": 0.93,
        "Reuters Investigates": 0.88,
        "AP Investigations": 0.87,
        "Bloomberg Investigates": 0.85,
        "NYT Investigations": 0.82,
        "Washington Post Investigations": 0.82,
        "Guardian Investigations": 0.82,
        "mainstream wire": 0.65,
        "government statement": 0.45,
    }

    def __init__(self, country_code: str) -> None:
        super().__init__(f"ct_country_{country_code.lower()}", "kb/conspiracy_theorist")
        self.country_code = country_code

    def research_month(self, year_month: str, force_refresh: bool = False) -> dict[str, Any]:
        """Produce a 'follow the money' analysis for this country in the given month.

        Checks cache first. If cached and not force_refresh, returns cached result.
        Otherwise runs LLM research and stores result.

        Args:
            year_month: "YYYY-MM" format.
            force_refresh: Re-run even if cache exists.

        Returns:
            Dict with financial_flows, anomalies, official_narrative,
            documented_reality, key_actors, market_correlation.
        """
        if not force_refresh:
            cached = self._load_from_cache(year_month)
            if cached:
                return cached

        result = self._run_research(year_month)
        self._save_to_cache(year_month, result)
        return result

    def _run_research(self, year_month: str) -> dict[str, Any]:
        system_prompt = f"""You are a PhD-level economist, historian, and investigative journalist
specializing in financial intelligence for {self.country_code}.

Your methodology: Follow the money. Find financial announcements, capital flows,
and corporate/government transactions within 6 months of significant market events.
Identify gaps between official narratives and documented reality.

Source credibility hierarchy you MUST apply:
- ICIJ / OCCRP / Bellingcat / investigative outlets: HIGH credibility (weight 0.90-1.0)
- Independent investigative journalists: HIGH credibility (0.85-0.95)
- Established international press investigative units: MEDIUM-HIGH (0.80-0.88)
- Mainstream national media: MEDIUM (0.60-0.75)
- Government statements and press releases: LOW-MEDIUM (0.40-0.60, verify independently)

Only flag findings with at least 2 independent credible sources.

Your constitutional framework: The concentration of financial and political power in
unaccountable structures (offshore vehicles, classified programs, shell companies)
distorts market outcomes. Your job is to detect these distortions before they
become public knowledge.

Return ONLY valid JSON with keys:
financial_flows, anomalies, official_narrative, documented_reality,
key_actors, market_correlation, confidence, source_weight_avg, coverage_notes"""

        user_msg = f"""Research {self.country_code} for {year_month}.

Identify:
1. Major financial flows — capital movement, FDI, debt issuance, commodity transactions
2. Anomalies — transactions or events that don't fit the official narrative
3. Official narrative — what governments/mainstream media said was happening
4. Documented reality — what investigative journalism or subsequent declassification revealed
5. Key actors — names, institutions, and their roles in any anomalies
6. Market correlation — how these flows and anomalies correlated with asset price moves

Focus on: offshore capital flows, defense/energy contracting, political funding trails,
regulatory actions that benefited specific actors, and any documents later revealed
by ICIJ, FOIA, or court proceedings.

Return JSON:
{{
  "financial_flows": [
    {{"description": str, "amount_estimate": str, "source": str, "credibility": float}}
  ],
  "anomalies": [
    {{"description": str, "official_explanation": str, "documented_alternative": str,
      "sources": [str], "confidence": float}}
  ],
  "official_narrative": str,
  "documented_reality": str,
  "key_actors": [{{"name": str, "role": str, "institution": str}}],
  "market_correlation": str,
  "confidence": float,
  "source_weight_avg": float,
  "coverage_notes": str
}}"""

        raw = self.call_llm(
            system_prompt,
            user_msg,
            include_kb=True,
            model_tier="reasoning",
        )
        parsed = self.validate_json_output(raw, required_keys=[
            "financial_flows", "anomalies", "official_narrative",
            "documented_reality", "key_actors", "market_correlation", "confidence",
        ])
        if parsed is None:
            return {
                "financial_flows": [],
                "anomalies": [],
                "official_narrative": f"Research failed for {self.country_code} {year_month}",
                "documented_reality": "",
                "key_actors": [],
                "market_correlation": "",
                "confidence": 0.0,
                "source_weight_avg": 0.0,
                "coverage_notes": "LLM parse failure",
                "country": self.country_code,
                "year_month": year_month,
            }
        parsed["country"] = self.country_code
        parsed["year_month"] = year_month
        return parsed

    def _load_from_cache(self, year_month: str) -> dict[str, Any] | None:
        try:
            with get_session() as session:
                row = session.query(CTCountryAnalysis).filter_by(
                    year_month=year_month,
                    country=self.country_code,
                ).first()
                if row:
                    return {
                        "financial_flows": row.financial_flows or [],
                        "anomalies": row.anomalies or [],
                        "official_narrative": row.official_narrative or "",
                        "documented_reality": row.documented_reality or "",
                        "key_actors": row.key_actors or [],
                        "market_correlation": row.market_correlation or "",
                        "source_weight_avg": row.source_weight_avg or 0.0,
                        "country": self.country_code,
                        "year_month": year_month,
                        "_cached": True,
                    }
        except Exception as exc:
            self.log_audit("cache_read_error", year_month, "", str(exc))
        return None

    def _save_to_cache(self, year_month: str, result: dict[str, Any]) -> None:
        try:
            with get_session() as session:
                existing = session.query(CTCountryAnalysis).filter_by(
                    year_month=year_month,
                    country=self.country_code,
                ).first()
                if existing:
                    existing.financial_flows = result.get("financial_flows")
                    existing.anomalies = result.get("anomalies")
                    existing.official_narrative = result.get("official_narrative")
                    existing.documented_reality = result.get("documented_reality")
                    existing.key_actors = result.get("key_actors")
                    existing.market_correlation = result.get("market_correlation")
                    existing.source_weight_avg = result.get("source_weight_avg")
                    existing.generated_at = datetime.now(timezone.utc)
                else:
                    session.add(CTCountryAnalysis(
                        year_month=year_month,
                        country=self.country_code,
                        financial_flows=result.get("financial_flows"),
                        anomalies=result.get("anomalies"),
                        official_narrative=result.get("official_narrative"),
                        documented_reality=result.get("documented_reality"),
                        key_actors=result.get("key_actors"),
                        market_correlation=result.get("market_correlation"),
                        source_weight_avg=result.get("source_weight_avg"),
                    ))
        except Exception as exc:
            self.log_audit("cache_write_error", year_month, "", str(exc))
