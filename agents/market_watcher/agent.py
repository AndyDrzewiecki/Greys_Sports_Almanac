"""Market watcher agent for cross-asset risk analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from agents.base_agent import BaseAgent
from data.connectors.yfinance_connector import YFinanceConnector
from storage.database import MarketReading


class MarketWatcher(BaseAgent):
    """Tracks cross-asset prices, leadership, and market divergences."""

    TICKERS = {
        "equity": ["SPY", "QQQ", "IWM", "DIA"],
        "fixed_income": ["TLT", "SHY", "HYG", "LQD"],
        "sectors": ["XLE", "XLF", "XLU", "XLV"],
        "commodities": ["GLD", "USO"],
        "international": ["EFA", "EWJ", "VEA"],
        "volatility": ["^VIX", "^VVIX"],
    }

    def __init__(self) -> None:
        """Initialize connector dependencies."""

        super().__init__("market_watcher", "kb/market_watcher")
        self.yfinance = YFinanceConnector()

    def run(self) -> dict[str, Any]:
        """Fetch market data, compute composite risk score, and persist results."""

        timestamp = datetime.now(timezone.utc)
        metrics: dict[str, dict[str, Any]] = {}

        for ticker in self._all_tickers():
            try:
                df = self.yfinance.fetch(ticker, period="1y")
                if df.empty or not self.yfinance.validate(df):
                    continue
                metrics[ticker] = self._compute_ticker_metrics(df)
            except Exception as exc:
                self.log_audit("fetch_error", ticker, "", error=str(exc), data_quality_flags="true")

        risk_components = self._risk_components(metrics)
        risk_on_score = sum(risk_components.values()) / len(risk_components) if risk_components else 0.0
        risk_label = "risk-on" if risk_on_score > 0.2 else "risk-off" if risk_on_score < -0.2 else "neutral"
        divergences = self._detect_divergences(metrics)
        sector_leadership = self._sector_leadership(metrics)
        cross_asset_summary = self._generate_summary(risk_on_score, risk_label, divergences, sector_leadership, metrics)
        self._save_readings(timestamp, metrics, risk_on_score, risk_components)

        output = {
            "timestamp": timestamp.isoformat(),
            "risk_on_score": round(risk_on_score, 4),
            "risk_label": risk_label,
            "prices": {
                ticker: {
                    "price": details["price"],
                    "1d": details["1d"],
                    "5d": details["5d"],
                    "20d": details["20d"],
                    "vs_52w_high_pct": details["vs_52w_high_pct"],
                }
                for ticker, details in metrics.items()
            },
            "sector_leadership": sector_leadership,
            "divergences": divergences,
            "cross_asset_summary": cross_asset_summary,
        }
        self.log_audit(
            event_type="market_watcher_run_success",
            input_summary=f"{len(metrics)} tickers processed",
            output_summary=json.dumps({"risk_on_score": risk_on_score, "divergences": divergences[:3]}),
        )
        return output

    def _all_tickers(self) -> list[str]:
        """Return the flattened ticker universe."""

        tickers: list[str] = []
        for values in self.TICKERS.values():
            tickers.extend(values)
        return tickers

    def _compute_ticker_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute returns and range position for a ticker."""

        closes = pd.to_numeric(df["close"], errors="coerce").dropna().reset_index(drop=True)
        current = float(closes.iloc[-1])
        one_day = self._pct_change(closes, 1)
        five_day = self._pct_change(closes, 5)
        twenty_day = self._pct_change(closes, 20)
        window = closes.tail(min(len(closes), 252))
        high_52w = float(window.max())
        low_52w = float(window.min())
        range_pct = 50.0 if high_52w == low_52w else ((current - low_52w) / (high_52w - low_52w)) * 100.0
        return {
            "price": current,
            "1d": one_day,
            "5d": five_day,
            "20d": twenty_day,
            "vs_52w_high_pct": round(range_pct, 2),
        }

    def _risk_components(self, metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Create normalized composite inputs for the risk-on score."""

        def bounded(value: float | None, scale: float) -> float:
            if value is None:
                return 0.0
            return max(-1.0, min(1.0, value / scale))

        spy_return = bounded(metrics.get("SPY", {}).get("5d"), 5.0)
        hyg_lqd = bounded(
            (metrics.get("HYG", {}).get("5d") or 0.0) - (metrics.get("LQD", {}).get("5d") or 0.0),
            3.0,
        )
        tlt_shy = bounded(
            -((metrics.get("TLT", {}).get("5d") or 0.0) - (metrics.get("SHY", {}).get("5d") or 0.0)),
            3.0,
        )
        xlf_xlu = bounded(
            (metrics.get("XLF", {}).get("5d") or 0.0) - (metrics.get("XLU", {}).get("5d") or 0.0),
            4.0,
        )
        vix = metrics.get("^VIX") or {}
        vix_component = 0.0
        if isinstance(vix, dict) and vix:
            level = float(vix.get("price") or 20.0)
            direction = float(vix.get("5d") or 0.0)
            vix_component = max(-1.0, min(1.0, ((20.0 - level) / 15.0) + (-direction / 10.0)))

        return {
            "spy_5d": spy_return,
            "hyg_vs_lqd": hyg_lqd,
            "tlt_vs_shy": tlt_shy,
            "xlf_vs_xlu": xlf_xlu,
            "vix": vix_component,
        }

    def _detect_divergences(self, metrics: dict[str, dict[str, Any]]) -> list[str]:
        """Detect simple cross-asset divergence conditions."""

        divergences: list[str] = []
        spy_5d = metrics.get("SPY", {}).get("5d")
        hyg_5d = metrics.get("HYG", {}).get("5d")
        vix_5d = metrics.get("^VIX", {}).get("5d")
        vvix_5d = metrics.get("^VVIX", {}).get("5d")
        xlu_5d = metrics.get("XLU", {}).get("5d")
        xlf_5d = metrics.get("XLF", {}).get("5d")

        if spy_5d is not None and hyg_5d is not None and spy_5d > 0 and hyg_5d < 0:
            divergences.append("SPY is rising while HYG is falling, suggesting a credit/equity divergence.")

        if vix_5d is not None and vvix_5d is not None and vix_5d < 0 and vvix_5d > 0:
            divergences.append("VIX is falling while VVIX is rising, indicating volatility-of-volatility stress.")

        equity_avg = pd.Series(
            [metrics.get(ticker, {}).get("5d") for ticker in self.TICKERS["equity"] if metrics.get(ticker)]
        ).dropna()
        if not equity_avg.empty and xlu_5d is not None and xlf_5d is not None:
            # Use explicit float() to avoid 'truth value of a Series is ambiguous'
            # when pandas returns np.float64 from mean() in certain versions.
            if float(equity_avg.mean()) > 0 and float(xlu_5d) > float(xlf_5d):
                divergences.append("Equities are up, but defensive utilities are leading cyclicals.")

        return divergences

    def _sector_leadership(self, metrics: dict[str, dict[str, Any]]) -> list[str]:
        """Rank sectors by 5-day return from best to worst."""

        sectors = []
        for ticker in self.TICKERS["sectors"]:
            if ticker in metrics and metrics[ticker]["5d"] is not None:
                sectors.append((ticker, metrics[ticker]["5d"]))
        return [ticker for ticker, _ in sorted(sectors, key=lambda item: item[1], reverse=True)]

    def _generate_summary(
        self,
        risk_on_score: float,
        risk_label: str,
        divergences: list[str],
        sector_leadership: list[str],
        metrics: dict[str, dict[str, Any]],
    ) -> str:
        """Use Claude to produce a concise cross-asset summary."""

        payload = {
            "risk_on_score": risk_on_score,
            "risk_label": risk_label,
            "divergences": divergences,
            "sector_leadership": sector_leadership,
            "selected_metrics": {ticker: metrics[ticker] for ticker in ["SPY", "HYG", "LQD", "TLT", "SHY", "^VIX"] if ticker in metrics},
        }
        return self.call_claude(
            system_prompt=(
                "You are the Market Watcher for a multi-agent market monitoring system. "
                "Write exactly 3 concise sentences summarizing cross-asset positioning, leadership, and notable divergences."
            ),
            user_message=json.dumps(payload, default=str),
            include_kb=True,
        )

    def _save_readings(
        self,
        timestamp: datetime,
        metrics: dict[str, dict[str, Any]],
        risk_on_score: float,
        risk_components: dict[str, float],
    ) -> None:
        """Persist ticker snapshots to the database."""

        with self.session_factory() as session:
            for ticker, details in metrics.items():
                session.add(
                    MarketReading(
                        timestamp=timestamp,
                        ticker=ticker,
                        price=details["price"],
                        pct_change_1d=details["1d"],
                        pct_change_5d=details["5d"],
                        risk_on_score=risk_on_score,
                        regime_contribution=json.dumps(risk_components),
                        source=self.yfinance.source_name,
                    )
                )

    @staticmethod
    def _pct_change(series: pd.Series, lookback: int) -> float | None:
        """Compute percentage change over a lookback window."""

        if len(series) <= lookback:
            return None
        previous = float(series.iloc[-lookback - 1])
        current = float(series.iloc[-1])
        if previous == 0:
            return None
        return ((current - previous) / abs(previous)) * 100.0


if __name__ == "__main__":
    agent = MarketWatcher()
    print(json.dumps(agent.run(), indent=2))
