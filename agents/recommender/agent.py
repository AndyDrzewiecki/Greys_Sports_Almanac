"""Recommender agent for macro, ETF, and stock positioning ideas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from agents.base_agent import BaseAgent
from storage.database import Recommendation


class Recommender(BaseAgent):
    """Converts validated forecast context into research recommendations."""

    SCENARIO_TO_ETFS = {
        "credit_accident": ["SH", "TLT", "GLD", "XLU"],
        "stagflation_drift": ["PDBC", "XLE", "TIPS", "GLD"],
        "soft_landing": ["QQQ", "VGT", "IWM", "HYG"],
        "energy_shock": ["XLE", "XOP", "TAN", "GLD"],
        "dollar_down_rotation": ["EEM", "GLD", "PDBC", "IAU"],
        "null_hypothesis": ["SPY", "VTI", "BND"],
        "treasury_plumbing_crisis": ["SH", "GLD", "TLT", "SHY"],
    }

    POSITIONING_MAP = {
        "credit_accident": {
            "posture": "defensive",
            "duration": "long",
            "geography": ["US"],
            "factors": ["quality"],
        },
        "stagflation_drift": {
            "posture": "defensive",
            "duration": "short",
            "geography": ["US", "International"],
            "factors": ["value", "quality"],
        },
        "soft_landing": {
            "posture": "offensive",
            "duration": "neutral",
            "geography": ["US", "International"],
            "factors": ["growth", "momentum"],
        },
        "energy_shock": {
            "posture": "neutral",
            "duration": "short",
            "geography": ["US", "International"],
            "factors": ["value", "quality"],
        },
        "treasury_plumbing_crisis": {
            "posture": "defensive",
            "duration": "long",
            "geography": ["US"],
            "factors": ["quality"],
        },
        "dollar_down_rotation": {
            "posture": "offensive",
            "duration": "neutral",
            "geography": ["International", "EM"],
            "factors": ["value", "momentum"],
        },
        "null_hypothesis": {
            "posture": "neutral",
            "duration": "neutral",
            "geography": ["US"],
            "factors": ["quality", "broad_market"],
        },
    }

    def __init__(self) -> None:
        """Initialize the recommender KB and shared services."""

        super().__init__("recommender", "kb/recommender")

    def run(
        self,
        forecast_output: dict[str, Any],
        signal_output: dict[str, Any],
        market_output: dict[str, Any],
        skeptic_output: dict[str, Any],
        geo_political_output: dict[str, Any] | None = None,
        geo_economic_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate recommendations if the skeptic allows the forecast through.

        Args:
            forecast_output: Scenario probabilities from Forecaster.
            signal_output: Macro regime from SignalWatcher.
            market_output: Risk posture from MarketWatcher.
            skeptic_output: Skeptic verdict and challenges.
            geo_political_output: Optional output from GeopoliticalHistorian.
            geo_economic_output: Optional output from GeoEconomicHistorian.
        """

        verdict = skeptic_output.get("skeptic_verdict", "approve")
        if verdict == "reject":
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "no_recommendations",
                "reason": skeptic_output.get("corrections_required", "Skeptic rejected forecast."),
            }

        timestamp = datetime.now(timezone.utc)
        top_scenario = forecast_output.get("top_scenario", "null_hypothesis")
        if top_scenario not in self.POSITIONING_MAP:
            top_scenario = "null_hypothesis"
        confidence_modifier = 0.75 if verdict == "caution" else 1.0
        base_positioning = self.POSITIONING_MAP[top_scenario]
        etf_suggestions = self._build_etf_suggestions(top_scenario, forecast_output, confidence_modifier)
        stock_ideas = self._generate_stock_ideas(
            top_scenario, market_output, confidence_modifier,
            geo_political_output, geo_economic_output,
        )
        do_nothing_assessment = self._do_nothing_assessment(forecast_output, skeptic_output)

        output = {
            "timestamp": timestamp.isoformat(),
            "skeptic_verdict_applied": verdict,
            "macro_positioning": base_positioning,
            "etf_suggestions": etf_suggestions,
            "stock_ideas": stock_ideas,
            "do_nothing_assessment": do_nothing_assessment,
            "geopolitical_risks": (geo_political_output or {}).get("top_risks", []),
            "economic_risks": (geo_economic_output or {}).get("top_risks", []),
            "disclaimer": "These are model-generated ideas for research purposes only, not financial advice.",
        }
        self._save_recommendations(output, forecast_output)
        self._append_recommendation_history(output, forecast_output)
        self.write_kb_staging(
            note=f"Recommendation batch staged for {top_scenario} with skeptic verdict {verdict}.",
            category="recommendation_history",
        )
        self.log_audit(
            event_type="recommender_run_success",
            input_summary=json.dumps({"scenario": top_scenario, "verdict": verdict}),
            output_summary=json.dumps(
                {
                    "macro_posture": base_positioning["posture"],
                    "etf_count": len(etf_suggestions),
                    "stock_count": len(stock_ideas),
                }
            ),
        )
        return output

    def _build_etf_suggestions(
        self,
        top_scenario: str,
        forecast_output: dict[str, Any],
        confidence_modifier: float,
    ) -> list[dict[str, Any]]:
        """Build deterministic ETF suggestions from the top scenario."""

        scenario_probability = forecast_output["scenarios"][top_scenario]
        tickers = self.SCENARIO_TO_ETFS[top_scenario]
        suggestions = []
        for ticker in tickers[:4]:
            suggestions.append(
                {
                    "ticker": ticker,
                    "thesis": f"{ticker} aligns with the {top_scenario} scenario and the current regime context.",
                    "scenario": top_scenario,
                    "confidence": round(min(1.0, scenario_probability * confidence_modifier), 3),
                }
            )
        return suggestions

    def _generate_stock_ideas(
        self,
        top_scenario: str,
        market_output: dict[str, Any],
        confidence_modifier: float,
        geo_political_output: dict[str, Any] | None = None,
        geo_economic_output: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Use Claude to generate stock ideas informed by scenario, leadership, and geo context."""

        payload = {
            "top_scenario": top_scenario,
            "risk_label": market_output.get("risk_label"),
            "sector_leadership": market_output.get("sector_leadership", []),
            "divergences": market_output.get("divergences", []),
            "geopolitical_top_risks": (geo_political_output or {}).get("top_risks", []),
            "economic_top_risks": (geo_economic_output or {}).get("top_risks", []),
            "geopolitical_regime": ((geo_political_output or {}).get("synthesis") or {}).get("geopolitical_regime"),
        }
        raw = self.call_claude(
            system_prompt=(
                "You are the Recommender agent. Generate 3 to 5 individual stock ideas for research only. "
                "Return JSON only with a top-level key stock_ideas containing objects with keys: ticker, thesis, time_horizon, key_risk, scenario_dependency. "
                "Each thesis should be exactly 2 sentences and clearly tied to the provided scenario."
            ),
            user_message=json.dumps(payload, default=str),
            include_kb=True,
        )
        parsed = self.validate_json_output(raw, required_keys=["stock_ideas"])
        ideas = []
        for item in parsed["stock_ideas"][:5]:
            ideas.append(
                {
                    "ticker": item["ticker"],
                    "thesis": item["thesis"],
                    "time_horizon": item["time_horizon"],
                    "key_risk": item["key_risk"],
                    "scenario_dependency": item["scenario_dependency"],
                    "confidence": round(confidence_modifier, 3),
                }
            )
        return ideas

    def _do_nothing_assessment(self, forecast_output: dict[str, Any], skeptic_output: dict[str, Any]) -> str:
        """Return an explicit assessment of whether standing pat is preferable."""

        confidence = float(forecast_output.get("confidence", 0.5))
        if skeptic_output.get("skeptic_verdict") == "caution" or confidence < 0.45:
            return "Doing nothing is a strong option because signal strength is mixed and the skeptic flagged meaningful uncertainty."
        return "Doing nothing is not the base case because the top scenario has enough support to justify research-sized positioning changes."

    def _save_recommendations(self, output: dict[str, Any], forecast_output: dict[str, Any]) -> None:
        """Persist recommendations to the database."""

        timestamp = datetime.fromisoformat(output["timestamp"])
        with self.session_factory() as session:
            macro = output["macro_positioning"]
            session.add(
                Recommendation(
                    timestamp=timestamp,
                    rec_type="macro",
                    ticker_or_theme=macro["posture"],
                    rationale=json.dumps(macro),
                    scenario_context=forecast_output["top_scenario"],
                    confidence=max(0.0, min(1.0, forecast_output["confidence"])),
                    skeptic_approved=output["skeptic_verdict_applied"] in {"approve", "caution"},
                )
            )
            for item in output["etf_suggestions"]:
                session.add(
                    Recommendation(
                        timestamp=timestamp,
                        rec_type="etf",
                        ticker_or_theme=item["ticker"],
                        rationale=item["thesis"],
                        scenario_context=item["scenario"],
                        confidence=item["confidence"],
                        skeptic_approved=True,
                    )
                )
            for item in output["stock_ideas"]:
                session.add(
                    Recommendation(
                        timestamp=timestamp,
                        rec_type="stock",
                        ticker_or_theme=item["ticker"],
                        rationale=item["thesis"],
                        scenario_context=item["scenario_dependency"],
                        confidence=item["confidence"],
                        skeptic_approved=True,
                    )
                )

    def _append_recommendation_history(self, output: dict[str, Any], forecast_output: dict[str, Any]) -> None:
        """Append ETF and stock recommendations to the KB history table."""

        kb = self.read_kb()
        marker = "|------|------|--------------|-----------|------------|------------|-------|"
        rows = []
        for item in output["etf_suggestions"]:
            rows.append(
                f"| {output['timestamp'][:10]} | ETF | {item['ticker']} | {item['thesis']} | pending | pending | pending |"
            )
        for item in output["stock_ideas"]:
            rows.append(
                f"| {output['timestamp'][:10]} | Stock | {item['ticker']} | {item['thesis']} | pending | pending | pending |"
            )
        rows.append(
            f"| {output['timestamp'][:10]} | Macro | {forecast_output['top_scenario']} | {output['do_nothing_assessment']} | pending | pending | pending |"
        )
        if marker in kb:
            kb = kb.replace(marker, f"{marker}\n" + "\n".join(rows), 1)
            self.kb_file.write_text(kb, encoding="utf-8")
            self.memory = kb
