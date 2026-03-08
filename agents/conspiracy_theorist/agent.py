"""ConspiracyTheorist — follows money trails and identifies financial anomalies.

Core identity:
  - PhD-level economist AND historian
  - Tracks the gap between official narratives and documented reality
  - Believes the NSA Act of 1947 created a structural constitutional black hole
    (Article I) that enables unaccountable financial and political actions
  - Focuses on ICIJ-quality investigative sources; dismisses unverified speculation
  - Works with the Orchestrator and Forecaster to flag anomalies that don't fit
    the consensus narrative

Documented historical framework (public record):
  - CIA origins: Operation Paperclip, Vatican ratlines, Wall Street law firm networks
  - Guatemala 1954, JFK 1963, Watergate, Iran-Contra, BCCI, PROMIS/Maxwell
  - Panama Papers, Pandora Papers, Epstein network, Cyprus Papers
  - Methodology: financial announcements within 6 months of market swings
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import config
from agents.base_agent import BaseAgent
from agents.conspiracy_theorist.country_agent import CTCountryAgent
from agents.conspiracy_theorist.economist_agent import CTEconomistAgent
from agents.conspiracy_theorist.political_scientist_agent import CTPoliticalScientistAgent
from storage.database import ConspiracyFinding, get_session


# Countries monitored — base 12 + extended for CT (adds Israel, Ukraine, Mexico, Nigeria)
CT_COUNTRIES = [
    "US", "CN", "RU", "DE", "JP", "SA", "IR", "VE", "BR", "IN", "GB", "FR",
    "IL", "UA", "MX", "NG",
]

# Documented reference events the CT agent uses as pattern anchors
ANCHOR_EVENTS = [
    {"name": "Panama Papers", "year": 2016, "type": "offshore_leak"},
    {"name": "Pandora Papers", "year": 2021, "type": "offshore_leak"},
    {"name": "Cyprus Papers", "year": 2020, "type": "passport_scheme"},
    {"name": "Epstein Network", "year": 2019, "type": "financial_intelligence"},
    {"name": "Iran-Contra", "year": 1986, "type": "arms_drugs_finance"},
    {"name": "BCCI Collapse", "year": 1991, "type": "intelligence_banking"},
    {"name": "PROMIS/Maxwell", "year": 1991, "type": "intelligence_technology"},
    {"name": "Kissinger Documents", "year": 1973, "type": "covert_policy"},
]


class ConspiracyTheorist(BaseAgent):
    """Follows money trails and flags anomalies that don't fit official narratives."""

    def __init__(self) -> None:
        super().__init__("conspiracy_theorist", "kb/conspiracy_theorist")
        self.country_agents: dict[str, CTCountryAgent] = {
            c: CTCountryAgent(c) for c in CT_COUNTRIES
        }
        self.economist = CTEconomistAgent()
        self.political_scientist = CTPoliticalScientistAgent()

    # ── Main entry points ──────────────────────────────────────────────────────

    def run(
        self,
        forecast_output: dict[str, Any] | None,
        market_output: dict[str, Any] | None,
        signal_output: dict[str, Any] | None,
        geo_political_output: dict[str, Any] | None = None,
        geo_economic_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Main pipeline run — analyze current month for anomalies and cross-reference forecast.

        This runs on the current month. It is called after the Forecaster and Skeptic
        so it can flag where the consensus forecast may be blind to documented anomalies.
        """
        year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        self.log_audit("run_start", f"CT analysis for {year_month}", "")

        try:
            # Run country-level analysis (top-priority countries only for live runs)
            priority_countries = ["US", "CN", "RU", "SA", "IR", "IL", "GB", "DE"]
            country_analyses = {}
            for country in priority_countries:
                try:
                    country_analyses[country] = self.country_agents[country].research_month(year_month)
                except Exception as exc:
                    self.log_audit("country_agent_error", country, "", str(exc))
                    country_analyses[country] = {"anomalies": [], "financial_flows": [], "confidence": 0.0}

            # Synthesize anomalies across all countries
            synthesis = self._synthesize_findings(
                year_month, country_analyses, forecast_output, market_output, signal_output
            )

            # Cross-reference with forecast
            forecast_cross_ref = self._cross_reference_forecast(
                forecast_output, synthesis, market_output
            )

            # Generate CT's alternative portfolio theory (secondary hypothesis)
            ct_portfolio_theory = self._generate_ct_portfolio_theory(
                synthesis, forecast_cross_ref, geo_political_output, geo_economic_output
            )

            result = {
                "year_month": year_month,
                "country_analyses": country_analyses,
                "synthesis": synthesis,
                "forecast_cross_reference": forecast_cross_ref,
                "ct_portfolio_theory": ct_portfolio_theory,
                "top_anomalies": synthesis.get("top_anomalies", []),
                "overall_anomaly_level": synthesis.get("anomaly_level", "low"),
                "confidence": synthesis.get("confidence", 0.5),
            }

            # Persist significant findings
            self._persist_findings(year_month, synthesis)

            # Stage KB note if significant anomalies found
            if synthesis.get("anomaly_level") in ("elevated", "high"):
                self.write_kb_staging(
                    f"[{year_month}] Anomaly level: {synthesis.get('anomaly_level')}. "
                    f"Top finding: {synthesis.get('top_anomalies', [{}])[0].get('description', '')[:200]}",
                    category="live_anomaly",
                )

            self.log_audit("run_success", f"CT {year_month}", json.dumps(result)[:500])
            return result

        except Exception as exc:
            self.log_audit("run_error", f"CT {year_month}", "", str(exc))
            return {
                "year_month": year_month,
                "error": str(exc),
                "top_anomalies": [],
                "overall_anomaly_level": "unknown",
                "confidence": 0.0,
            }

    def analyze_period(
        self,
        year_month: str,
        forecast: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Analyze a historical period — used by the backtest engine.

        Runs all country agents for the given month and produces a full synthesis.
        Results are cached at the country level to avoid redundant LLM calls.
        """
        self.log_audit("historical_analysis_start", year_month, "")

        country_analyses = {}
        for country in CT_COUNTRIES:
            try:
                country_analyses[country] = self.country_agents[country].research_month(
                    year_month, force_refresh=force_refresh
                )
            except Exception as exc:
                self.log_audit("country_agent_error", f"{country}/{year_month}", "", str(exc))
                country_analyses[country] = {"anomalies": [], "financial_flows": [], "confidence": 0.0}

        synthesis = self._synthesize_findings(
            year_month, country_analyses, forecast, None, None
        )

        return {
            "year_month": year_month,
            "country_analyses": country_analyses,
            "synthesis": synthesis,
            "top_anomalies": synthesis.get("top_anomalies", []),
            "overall_anomaly_level": synthesis.get("anomaly_level", "low"),
        }

    def get_monthly_summary(self, year_month: str) -> str:
        """Return a text summary of CT findings for a given month.

        Used by the Orchestrator for KB synthesis and daily briefs.
        """
        result = self.analyze_period(year_month)
        synthesis = result.get("synthesis", {})
        top = result.get("top_anomalies", [])[:3]
        top_str = "; ".join(a.get("description", "")[:100] for a in top) if top else "None flagged"
        return (
            f"CT Analysis {year_month}: anomaly_level={result.get('overall_anomaly_level', 'unknown')}, "
            f"top_findings=[{top_str}], "
            f"official_narrative_gaps={synthesis.get('narrative_gaps_count', 0)}"
        )

    # ── Internal synthesis ─────────────────────────────────────────────────────

    def _synthesize_findings(
        self,
        year_month: str,
        country_analyses: dict[str, Any],
        forecast_output: dict[str, Any] | None,
        market_output: dict[str, Any] | None,
        signal_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Synthesize country-level findings into a cross-country anomaly picture."""

        all_anomalies = []
        for country, analysis in country_analyses.items():
            for anomaly in analysis.get("anomalies", []):
                anomaly["country"] = country
                all_anomalies.append(anomaly)

        anomaly_str = json.dumps(all_anomalies[:20], indent=2) if all_anomalies else "[]"
        forecast_str = json.dumps(forecast_output, indent=2)[:800] if forecast_output else "none"
        market_str = json.dumps(market_output, indent=2)[:400] if market_output else "none"

        system_prompt = """You are a PhD-level economist and historian with expertise in
following money trails and detecting the gap between official narratives and documented reality.

You maintain a constitutional framework: the NSA Act of 1947 created unaccountable structures
that distort market outcomes. Your job is to detect when these structures are active.

Anchor your analysis to documented events (Panama Papers, ICIJ database, court records,
Senate/Congressional investigations, FOIA releases) rather than speculation.

Source credibility: ICIJ/OCCRP/Bellingcat > independent investigative > establishment press > mainstream.

Return ONLY valid JSON."""

        user_msg = f"""Synthesize CT findings for {year_month}.

Country-level anomalies found:
{anomaly_str}

Current forecast context:
{forecast_str}

Current market context:
{market_str}

Produce a cross-country synthesis:
1. Top anomalies — rank by market impact potential and evidence quality
2. Cross-country money flow patterns — capital moving between countries in anomalous ways
3. Narrative gaps — where official stories diverge most from documented reality
4. Anomaly level — overall assessment of how anomalous current conditions are
5. Forecaster blind spots — what does the consensus forecast miss that CT analysis sees?
6. Recommended follow-up — what additional data or time period analysis would sharpen the picture?

Return JSON:
{{
  "top_anomalies": [
    {{
      "description": str,
      "countries_involved": [str],
      "financial_mechanism": str,
      "market_impact_potential": str,
      "evidence_quality": str,
      "confidence": float,
      "reference_anchors": [str]
    }}
  ],
  "cross_country_flows": [str],
  "narrative_gaps_count": int,
  "narrative_gaps_summary": str,
  "anomaly_level": "low|moderate|elevated|high",
  "forecaster_blind_spots": [str],
  "recommended_follow_up": [str],
  "confidence": float
}}"""

        raw = self.call_llm(system_prompt, user_msg, include_kb=True, model_tier="reasoning")
        parsed = self.validate_json_output(raw, required_keys=[
            "top_anomalies", "anomaly_level", "confidence",
        ])
        if parsed is None:
            return {
                "top_anomalies": [],
                "cross_country_flows": [],
                "narrative_gaps_count": 0,
                "narrative_gaps_summary": "Synthesis failed",
                "anomaly_level": "unknown",
                "forecaster_blind_spots": [],
                "recommended_follow_up": [],
                "confidence": 0.0,
            }

        # Enrich top anomalies with economist + political scientist analysis
        for anomaly in parsed.get("top_anomalies", [])[:3]:
            if anomaly.get("confidence", 0) >= 0.60:
                try:
                    econ = self.economist.analyze_anomaly(
                        anomaly.get("description", ""),
                        ",".join(anomaly.get("countries_involved", [])),
                        year_month,
                        market_context=market_str,
                    )
                    anomaly["economist_analysis"] = econ.get("mechanism", "")
                    anomaly["sectors_affected"] = econ.get("market_implications", {}).get("sectors_affected", [])
                except Exception:
                    pass

                try:
                    polsci = self.political_scientist.analyze_power_structure(
                        ",".join(anomaly.get("countries_involved", [])),
                        year_month,
                        anomaly.get("description", ""),
                        anomaly.get("financial_mechanism", ""),
                    )
                    anomaly["political_analysis"] = polsci.get("institutional_mechanism", "")
                except Exception:
                    pass

        return parsed

    def _cross_reference_forecast(
        self,
        forecast_output: dict[str, Any] | None,
        synthesis: dict[str, Any],
        market_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Cross-reference the consensus forecast with CT findings.

        Returns a structured assessment of where the forecast may be blind
        to documented anomalies.
        """
        if not forecast_output:
            return {"assessment": "No forecast to cross-reference", "adjustments": []}

        top_scenario = ""
        if forecast_output.get("scenarios"):
            top_scenario = max(
                forecast_output["scenarios"].items(), key=lambda x: x[1]
            )[0] if isinstance(forecast_output.get("scenarios"), dict) else str(forecast_output.get("scenarios", ""))

        blind_spots = synthesis.get("forecaster_blind_spots", [])
        top_anomalies = synthesis.get("top_anomalies", [])

        system_prompt = """You are the Conspiracy Theorist reviewing the consensus forecast.
Your job is to find what the consensus is missing — not to contradict without evidence,
but to flag documented anomalies that should adjust probability weights.

Be rigorous: only flag things supported by evidence. Speculat is not your method.
Documented reality is your anchor.

Return ONLY valid JSON."""

        user_msg = f"""Consensus top scenario: {top_scenario}
Forecast confidence: {forecast_output.get('confidence', 'unknown')}

CT-identified blind spots: {json.dumps(blind_spots)}
Top CT anomalies: {json.dumps([a.get('description', '') for a in top_anomalies[:5]])}

Cross-reference and produce:
1. Does any CT anomaly change the probability distribution of scenarios?
2. Are there scenario dependencies the forecast misses due to official narrative acceptance?
3. What is the CT-adjusted probability shift (if any) and the evidence basis?

Return JSON:
{{
  "verdict": "confirms|partially_confirms|flags_blind_spots|contradicts",
  "adjustments": [
    {{"scenario": str, "direction": "up|down", "basis": str, "magnitude": "small|moderate|significant"}}
  ],
  "blind_spot_summary": str,
  "ct_confidence_in_adjustment": float,
  "key_evidence": [str]
}}"""

        raw = self.call_llm(system_prompt, user_msg, include_kb=False, model_tier="reasoning")
        parsed = self.validate_json_output(raw, required_keys=["verdict", "adjustments"])
        return parsed if parsed else {
            "verdict": "unknown",
            "adjustments": [],
            "blind_spot_summary": "Cross-reference failed",
            "ct_confidence_in_adjustment": 0.0,
            "key_evidence": [],
        }

    def _generate_ct_portfolio_theory(
        self,
        synthesis: dict[str, Any],
        forecast_cross_ref: dict[str, Any],
        geo_political_output: dict[str, Any] | None,
        geo_economic_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Generate CT's secondary portfolio theory — the alternative hypothesis.

        This is stored in the live_portfolio_recommendations table with
        data_source='ct_secondary' and tracked alongside the main forecast.
        """
        system_prompt = """You are the Conspiracy Theorist generating an alternative portfolio theory.
Your thesis: documented financial anomalies and money trails that the consensus misses
create exploitable market inefficiencies. Follow the money to where it's actually going,
not where official narratives say it should go.

Use only documented patterns and evidence. This is a research hypothesis, not investment advice.

Return ONLY valid JSON."""

        geo_pol_str = json.dumps(geo_political_output, indent=2)[:400] if geo_political_output else "none"
        geo_econ_str = json.dumps(geo_economic_output, indent=2)[:400] if geo_economic_output else "none"

        user_msg = f"""CT synthesis findings:
{json.dumps(synthesis, indent=2)[:800]}

Forecast cross-reference:
{json.dumps(forecast_cross_ref, indent=2)[:400]}

Geopolitical context: {geo_pol_str}
Geoeconomic context: {geo_econ_str}

Generate CT's secondary portfolio thesis. Based on DOCUMENTED money trails and anomalies,
where is capital ACTUALLY flowing vs. where official narratives say it should flow?

Return JSON:
{{
  "thesis_summary": str,
  "portfolio_allocation": {{
    "sectors": [{{"sector": str, "allocation_pct": float, "rationale": str}}],
    "geographic_tilts": [str],
    "avoid_list": [str]
  }},
  "key_money_trail_signals": [str],
  "divergence_from_consensus": str,
  "time_horizon": str,
  "evidence_basis": [str],
  "confidence": float,
  "disclaimer": "This is a research hypothesis based on documented financial patterns, not investment advice."
}}"""

        raw = self.call_llm(system_prompt, user_msg, include_kb=True, model_tier="reasoning")
        parsed = self.validate_json_output(raw, required_keys=["thesis_summary", "portfolio_allocation", "confidence"])
        return parsed if parsed else {
            "thesis_summary": "CT portfolio theory generation failed",
            "portfolio_allocation": {},
            "confidence": 0.0,
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist_findings(self, year_month: str, synthesis: dict[str, Any]) -> None:
        """Persist top anomalies to the conspiracy_findings table."""
        top_anomalies = synthesis.get("top_anomalies", [])
        if not top_anomalies:
            return
        try:
            with get_session() as session:
                for anomaly in top_anomalies[:5]:
                    session.add(ConspiracyFinding(
                        finding_type="financial_anomaly",
                        year_month=year_month,
                        countries_involved=anomaly.get("countries_involved"),
                        source_documents=anomaly.get("reference_anchors"),
                        anomaly_description=anomaly.get("description", ""),
                        market_event_within_6mo=anomaly.get("market_impact_potential"),
                        financial_flow_description=anomaly.get("financial_mechanism"),
                        economist_analysis=anomaly.get("economist_analysis"),
                        political_analysis=anomaly.get("political_analysis"),
                        confidence=anomaly.get("confidence", 0.5),
                        status="flagged",
                    ))
        except Exception as exc:
            self.log_audit("persist_error", year_month, "", str(exc))
