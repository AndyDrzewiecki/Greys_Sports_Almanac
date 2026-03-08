"""CTEconomistAgent — advanced economic theory sub-agent.

Provides rigorous economic frameworks to explain anomalies that conventional
analysis dismisses. Maintains its own KB at kb/conspiracy_theorist/economist/.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


class CTEconomistAgent(BaseAgent):
    """PhD-level economic theory analyst for the Conspiracy Theorist agent."""

    def __init__(self) -> None:
        super().__init__("ct_economist", "kb/conspiracy_theorist/economist")

    def analyze_anomaly(
        self,
        anomaly_description: str,
        country: str,
        year_month: str,
        market_context: str = "",
    ) -> dict[str, Any]:
        """Apply advanced economic theory to explain a flagged anomaly.

        Returns economic mechanism, historical precedents, market implications,
        and recommended additional signals to track.
        """
        system_prompt = """You are a PhD-level economist specializing in institutional economics,
political economy, and the economics of information asymmetry. Your knowledge spans:
- Regulatory capture theory (Stigler)
- Minsky financial instability hypothesis
- Public choice theory (Buchanan, Tullock)
- Shadow banking and systemic risk (Gorton, Metrick)
- Financialization and rent-seeking (Lazonick)
- Modern monetary theory (Kelton) — critical assessment
- Dollar weaponization and reserve currency dynamics
- Elite network theory and power concentration (Mills, Domhoff)

Your job is to provide the economic MECHANISM behind financial anomalies.
Correlation without mechanism is noise. Always explain WHY an anomaly would
produce the market effects observed.

Return ONLY valid JSON."""

        user_msg = f"""Analyze this financial anomaly from {country}, {year_month}:

{anomaly_description}

Market context: {market_context}

Apply your economic frameworks to explain:
1. The economic mechanism — WHY would this anomaly produce market effects?
2. Historical precedents — when have similar economic structures produced similar outcomes?
3. Who benefits — identify the rent-seeking actors and their incentive structures
4. Market implications — what asset classes, sectors, and time horizons are affected?
5. Recommended signals — what additional economic data points would confirm or refute this?
6. Confidence assessment — how certain are you of the mechanism, and why?

Return JSON:
{{
  "mechanism": str,
  "historical_precedents": [str],
  "beneficiaries": [{{"actor": str, "incentive": str, "estimated_gain": str}}],
  "market_implications": {{
    "sectors_affected": [str],
    "direction": str,
    "time_horizon": str,
    "magnitude_estimate": str
  }},
  "recommended_signals": [str],
  "confidence": float,
  "theoretical_framework": str,
  "counterarguments": [str]
}}"""

        raw = self.call_llm(
            system_prompt,
            user_msg,
            include_kb=True,
            model_tier="reasoning",
        )
        parsed = self.validate_json_output(raw, required_keys=[
            "mechanism", "market_implications", "confidence",
        ])
        if parsed is None:
            return {
                "mechanism": "Economic analysis failed",
                "historical_precedents": [],
                "beneficiaries": [],
                "market_implications": {"sectors_affected": [], "direction": "unknown",
                                        "time_horizon": "unknown", "magnitude_estimate": "unknown"},
                "recommended_signals": [],
                "confidence": 0.0,
                "theoretical_framework": "",
                "counterarguments": [],
            }
        # Stage notable findings to KB
        if parsed.get("confidence", 0) >= 0.70 and parsed.get("mechanism"):
            self.write_kb_staging(
                f"[{year_month}] {country}: {parsed['mechanism'][:200]}",
                category="economic_mechanism",
            )
        return parsed

    def research_theory(self, topic: str) -> dict[str, Any]:
        """Deep-dive research on an advanced economic theory topic.

        Used for background research that enriches the KB over time.
        """
        system_prompt = """You are a PhD-level economist. Produce rigorous academic-quality
analysis of economic theory with investment relevance. Cite real economists and documented
empirical findings. Distinguish between theoretical prediction and empirical evidence.

Return ONLY valid JSON."""

        user_msg = f"""Research topic: {topic}

Produce a comprehensive analysis covering:
1. Core theoretical framework — key concepts and mechanisms
2. Key empirical findings — documented evidence supporting or refuting the theory
3. Investment implications — how this theory predicts market behavior
4. Anomaly detection — what observable signals indicate this mechanism is operating
5. Limitations — where the theory breaks down

Return JSON:
{{
  "topic": str,
  "theoretical_framework": str,
  "key_economists": [str],
  "empirical_findings": [str],
  "investment_implications": [str],
  "anomaly_signals": [str],
  "limitations": [str],
  "confidence_in_theory": float
}}"""

        raw = self.call_llm(
            system_prompt,
            user_msg,
            include_kb=True,
            model_tier="reasoning",
        )
        parsed = self.validate_json_output(raw, required_keys=["topic", "theoretical_framework"])
        if parsed is None:
            return {"topic": topic, "theoretical_framework": "Research failed", "confidence_in_theory": 0.0}
        # Always stage new theoretical research to KB
        self.write_kb_staging(
            f"Theory research: {topic} — {parsed.get('theoretical_framework', '')[:300]}",
            category="theory_research",
        )
        return parsed
