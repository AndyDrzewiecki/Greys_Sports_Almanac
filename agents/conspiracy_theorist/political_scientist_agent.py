"""CTPoliticalScientistAgent — advanced political science sub-agent.

Provides rigorous political science frameworks to explain power dynamics,
institutional behavior, and the political economy of secrecy.
Maintains its own KB at kb/conspiracy_theorist/political_science/.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class CTPoliticalScientistAgent(BaseAgent):
    """PhD-level political scientist for the Conspiracy Theorist agent."""

    def __init__(self) -> None:
        super().__init__("ct_political_scientist", "kb/conspiracy_theorist/political_science")

    def analyze_power_structure(
        self,
        country: str,
        year_month: str,
        event_description: str,
        financial_anomaly: str = "",
    ) -> dict[str, Any]:
        """Apply political science frameworks to explain a power/policy anomaly.

        Returns power network analysis, institutional mechanisms, documented precedents,
        and investment implications of the political dynamic.
        """
        system_prompt = """You are a PhD-level political scientist specializing in:
- Deep state theory (academic: permanent administrative apparatus, Alasdair Roberts)
- Elite network theory (Mills "Power Elite", Domhoff "Who Rules America")
- Regulatory capture and iron triangle dynamics (Bernstein, McConnell)
- Authoritarian institutions and democratic backsliding (Levitsky, Ziblatt)
- Intelligence community political economy and threat inflation
- Propaganda and manufacturing consent (Herman, Chomsky)
- Resource nationalism and strategic mineral competition
- International relations realism and offshore financial architecture

Core methodology: Identify who benefits from the official narrative being accepted at face value.
Map the institutional actors, their incentives, and the documented (not alleged) connections
between them. Always distinguish between documented (court records, FOIA, official investigations)
and alleged (single-source, unverified).

Return ONLY valid JSON."""

        user_msg = f"""Analyze this political/power anomaly from {country}, {year_month}:

Event: {event_description}
Financial anomaly (if any): {financial_anomaly}

Apply political science frameworks to explain:
1. Power network — who are the institutional actors and what are their documented connections?
2. Institutional mechanism — what political/bureaucratic structure enables this anomaly?
3. Official narrative vs. documented reality — what do we actually know vs. what was claimed?
4. Historical precedents — when have similar power structures produced similar outcomes?
5. Accountability gap — what oversight mechanism failed or was circumvented?
6. Investment implications — how does this political dynamic affect asset prices and capital flows?
7. Monitoring signals — what observable political events would indicate escalation or resolution?

Return JSON:
{{
  "power_network": [
    {{"actor": str, "institution": str, "role": str, "documented_connections": [str]}}
  ],
  "institutional_mechanism": str,
  "official_narrative": str,
  "documented_reality": str,
  "historical_precedents": [str],
  "accountability_gap": str,
  "investment_implications": {{
    "sectors_affected": [str],
    "capital_flow_direction": str,
    "country_risk_change": str,
    "time_horizon": str
  }},
  "monitoring_signals": [str],
  "confidence": float,
  "primary_framework": str,
  "key_sources": [str]
}}"""

        raw = self.call_llm(
            system_prompt,
            user_msg,
            include_kb=True,
            model_tier="reasoning",
        )
        parsed = self.validate_json_output(raw, required_keys=[
            "institutional_mechanism", "investment_implications", "confidence",
        ])
        if parsed is None:
            return {
                "power_network": [],
                "institutional_mechanism": "Political analysis failed",
                "official_narrative": "",
                "documented_reality": "",
                "historical_precedents": [],
                "accountability_gap": "",
                "investment_implications": {"sectors_affected": [], "capital_flow_direction": "",
                                            "country_risk_change": "", "time_horizon": ""},
                "monitoring_signals": [],
                "confidence": 0.0,
                "primary_framework": "",
                "key_sources": [],
            }
        if parsed.get("confidence", 0) >= 0.70:
            self.write_kb_staging(
                f"[{year_month}] {country}: {parsed.get('institutional_mechanism', '')[:200]}",
                category="power_analysis",
            )
        return parsed

    def assess_country_risk(self, country: str, current_context: str) -> dict[str, Any]:
        """Assess the political risk profile of a country using political science frameworks.

        Produces a structured country risk assessment for use in portfolio allocation.
        """
        system_prompt = """You are a PhD-level political scientist producing investment-relevant
country risk assessments. Use: V-Dem democratic quality scores, Freedom House index,
Reporters Without Borders press freedom index, Transparency International CPI,
and elite network analysis.

Distinguish between: surface political risk (election outcomes) and structural political
risk (institutional capture, accountability breakdown, elite network stability).

Structural risk is more persistent and less priced by markets.

Return ONLY valid JSON."""

        user_msg = f"""Assess political risk for {country} given current context:

{current_context}

Produce investment-relevant political risk assessment:
1. Regime type and stability — structural vs. surface risk
2. Elite network stability — are current power structures consolidated or contested?
3. Accountability institutions — functioning, captured, or absent?
4. Key political risk events — upcoming elections, succession, constitutional moments
5. Intelligence/security apparatus — domestically focused or expanding externally?
6. Capital flow implications — political risk premium, capital flight indicators
7. Sector-specific risk — which industries face highest political risk?

Return JSON:
{{
  "regime_type": str,
  "structural_stability_score": float,
  "elite_network_status": str,
  "accountability_institutions": str,
  "key_risk_events": [str],
  "security_apparatus_orientation": str,
  "capital_flow_risk": str,
  "sector_risks": [{{"sector": str, "risk_level": str, "reason": str}}],
  "overall_political_risk": str,
  "investment_recommendation": str,
  "confidence": float
}}"""

        raw = self.call_llm(
            system_prompt,
            user_msg,
            include_kb=True,
            model_tier="reasoning",
        )
        parsed = self.validate_json_output(raw, required_keys=[
            "regime_type", "overall_political_risk", "confidence",
        ])
        if parsed is None:
            return {
                "regime_type": "unknown",
                "structural_stability_score": 0.5,
                "overall_political_risk": "unknown",
                "investment_recommendation": "insufficient data",
                "confidence": 0.0,
            }
        return parsed
