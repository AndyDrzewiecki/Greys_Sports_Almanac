"""Forecaster agent for scenario probability generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import config
from agents.base_agent import BaseAgent
from storage.database import Forecast


class Forecaster(BaseAgent):
    """Generates probabilistic market scenarios from upstream context."""

    def __init__(self) -> None:
        """Initialize the forecaster KB and base agent services."""

        super().__init__("forecaster", "kb/forecaster")

    def run(
        self,
        signal_output: dict[str, Any],
        market_output: dict[str, Any],
        historian_output: dict[str, Any],
        simulation_mode: bool = False,
    ) -> dict[str, Any]:
        """Generate scenario probabilities and optionally persist the forecast.

        Args:
            signal_output: Output from SignalWatcher.
            market_output: Output from MarketWatcher.
            historian_output: Output from Historian (may include geo context).
            simulation_mode: When True (BacktestEngine calls), skip all DB writes
                             and KB staging so production data is not polluted.
        """

        timestamp = datetime.now(timezone.utc)
        context = {
            "signal_output": signal_output,
            "market_output": market_output,
            "historian_output": historian_output,
        }
        raw = self.call_claude(
            system_prompt=(
                "You are the Forecaster agent for a financial research system.\n"
                "Use exactly these scenarios: null_hypothesis, soft_landing, stagflation_drift, "
                "credit_accident, energy_shock, treasury_plumbing_crisis, dollar_down_rotation.\n"
                "Return JSON only with keys: scenarios, top_drivers, confidence, invalidations, horizon, notes.\n"
                "Rules: probabilities must sum to 1.0, null_hypothesis must be at least 0.15, "
                "top_drivers must cite specific indicator names from the inputs, invalidations must cover the top 2 scenarios."
            ),
            user_message=json.dumps(context, default=str),
            include_kb=True,
        )
        parsed = self.validate_json_output(
            raw,
            required_keys=["scenarios", "top_drivers", "confidence", "invalidations", "horizon", "notes"],
        )
        scenarios = self._normalize_scenarios(parsed["scenarios"])
        top_scenario = max(scenarios, key=scenarios.get)

        output = {
            "timestamp": timestamp.isoformat(),
            "scenarios": scenarios,
            "top_scenario": top_scenario,
            "top_drivers": parsed["top_drivers"],
            "confidence": float(parsed["confidence"]),
            "invalidations": parsed["invalidations"],
            "horizon": parsed["horizon"],
            "notes": parsed["notes"],
            "simulation_mode": simulation_mode,
        }

        if not simulation_mode:
            self._save_forecast(output)
            self._append_forecast_history(output)
            self.write_kb_staging(
                note=f"Forecast staged with top scenario {top_scenario} at confidence {output['confidence']:.2f}.",
                category="forecast_history",
            )

        self.log_audit(
            event_type="forecaster_run_success",
            input_summary=json.dumps({"regime": signal_output.get("regime"), "risk": market_output.get("risk_label")}),
            output_summary=json.dumps({"top_scenario": top_scenario, "confidence": output["confidence"]}),
        )
        return output

    def _normalize_scenarios(self, scenarios: dict[str, Any]) -> dict[str, float]:
        """Validate and normalize scenario probabilities."""

        missing = [name for name in config.FORECAST_SCENARIOS if name not in scenarios]
        if missing:
            raise ValueError(f"Missing required scenarios: {missing}")

        normalized = {name: max(0.0, float(scenarios[name])) for name in config.FORECAST_SCENARIOS}
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("Scenario probabilities must sum to a positive value.")

        normalized = {name: value / total for name, value in normalized.items()}
        if normalized["null_hypothesis"] < 0.15:
            deficit = 0.15 - normalized["null_hypothesis"]
            adjustable = [name for name in normalized if name != "null_hypothesis" and normalized[name] > 0]
            adjustable_total = sum(normalized[name] for name in adjustable)
            if adjustable_total <= 0:
                raise ValueError("Cannot enforce null_hypothesis minimum without positive alternate scenarios.")
            for name in adjustable:
                normalized[name] -= deficit * (normalized[name] / adjustable_total)
            normalized["null_hypothesis"] = 0.15

        final_total = sum(normalized.values())
        normalized = {name: round(value / final_total, 6) for name, value in normalized.items()}
        rounding_diff = round(1.0 - sum(normalized.values()), 6)
        normalized["null_hypothesis"] = round(normalized["null_hypothesis"] + rounding_diff, 6)
        return normalized

    def _save_forecast(self, output: dict[str, Any]) -> None:
        """Persist the forecast to the database."""

        with self.session_factory() as session:
            session.add(
                Forecast(
                    timestamp=datetime.fromisoformat(output["timestamp"]),
                    agent_version="forecaster_v1",
                    scenarios_json=json.dumps(output["scenarios"]),
                    top_drivers_json=json.dumps(output["top_drivers"]),
                    confidence=output["confidence"],
                    invalidations_json=json.dumps(output["invalidations"]),
                    notes=output["notes"],
                )
            )

    def _append_forecast_history(self, output: dict[str, Any]) -> None:
        """Append an unscored forecast row to the KB history table."""

        top_scenario = output["top_scenario"]
        probability = output["scenarios"][top_scenario]
        row = f"| {output['timestamp'][:10]} | {top_scenario} | {probability:.3f} | pending | pending | {output['notes']} |"
        kb = self.read_kb()
        marker = "|------|----------|-------------|-------------|-------------|-------|"
        if marker in kb:
            kb = kb.replace(marker, f"{marker}\n{row}", 1)
            self.kb_file.write_text(kb, encoding="utf-8")
            self.memory = kb
