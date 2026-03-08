"""BacktestEngine — trains the Forecaster on 660 months of history (1965-2020).

Design principles:
  - One month at a time, with checkpointing so the run can be interrupted and resumed.
  - GDELT only available 2015+; 1979+ has event data; pre-1979 uses FRED/World Bank only.
  - Scoring is deferred: for a month M, we score the call against M+10yr data once
    10 years of actual returns exist.
  - Lessons are written to the backtest_lessons table and promoted to KB files
    after 3+ validating examples.

Performance note: 660 months × ~5 min LLM time = ~55 hours.  Run overnight in batches.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, date, timezone
from typing import Any

import pandas as pd

import config
from agents.base_agent import BaseAgent
from agents.forecaster.agent import Forecaster
from agents.geopolitical_historian.agent import GeopoliticalHistorian
from agents.geo_economic_historian.agent import GeoEconomicHistorian
from data.connectors.fred_connector import FREDConnector
from data.connectors.yfinance_connector import YFinanceConnector
from storage.database import BacktestLesson, BacktestRun, BacktestScore, get_session

_BACKTEST_DELAY_SEC = 2  # pause between months to avoid rate limits

# FRED series to pull for each historical snapshot
BACKTEST_FRED_SERIES: dict[str, str] = {
    "FEDFUNDS": "fed_funds_rate",
    "CPIAUCSL": "cpi",
    "UNRATE": "unemployment",
    "GS10": "treasury_10yr",
    "GS2": "treasury_2yr",
    "DCOILWTICO": "wti_oil",
    "GOLDAMGBD228NLBM": "gold",
    "VIXCLS": "vix",
}


class BacktestEngine(BaseAgent):
    """Runs the Forecaster over historical months and scores the results."""

    def __init__(self) -> None:
        super().__init__("backtester", "kb/forecaster")
        self.fred = FREDConnector()
        self.yf = YFinanceConnector()
        self.geo_pol = GeopoliticalHistorian()
        self.geo_econ = GeoEconomicHistorian()

    def run_month(self, year_month: str, force_refresh: bool = False) -> dict[str, Any]:
        """Run the full forecasting pipeline for one historical month.

        This simulates what the model would have said with only information
        available at that time (data isolation is approximate — we use FRED
        vintage data where available, and note coverage limitations).

        Args:
            year_month: "YYYY-MM" format.
            force_refresh: Re-run even if a BacktestRun already exists.

        Returns:
            Dict with the backtest result including checkpoint info.
        """

        if not force_refresh:
            existing = self._load_checkpoint(year_month)
            if existing:
                return {"status": "skipped_cached", "year_month": year_month, "run_id": existing}

        fred_snapshot = self._build_fred_snapshot(year_month)
        political_summary = self._get_political_context(year_month)
        economic_summary = self._get_economic_context(year_month)
        coverage = self._determine_coverage(year_month)

        # Synthesize a mock signal_output and market_output from historical data
        mock_signal = self._build_mock_signal(year_month, fred_snapshot)
        mock_market = self._build_mock_market(year_month, fred_snapshot)
        mock_historian = self._build_mock_historian(year_month, political_summary, economic_summary)

        try:
            forecaster = Forecaster()
            # simulation_mode=True: skips all DB/KB writes so production data stays clean
            forecast = forecaster.run(mock_signal, mock_market, mock_historian, simulation_mode=True)
            portfolio_rec = self._generate_portfolio_call(year_month, forecast, political_summary, economic_summary)
        except Exception as exc:
            self.log_audit("backtest_run_error", year_month, str(exc), error=str(exc))
            portfolio_rec = {"error": str(exc), "year_month": year_month}
            forecast = {"confidence": 0.0}

        run_id = self._save_run(
            year_month=year_month,
            fred_snapshot=fred_snapshot,
            political_summary=political_summary,
            economic_summary=economic_summary,
            portfolio_rec=portfolio_rec,
            confidence=float(forecast.get("confidence", 0.0)),
            coverage=coverage,
        )

        self.log_audit(
            "backtest_run_complete",
            f"month={year_month}",
            f"run_id={run_id} coverage={coverage}",
        )

        time.sleep(_BACKTEST_DELAY_SEC)
        return {
            "status": "completed",
            "year_month": year_month,
            "run_id": run_id,
            "coverage": coverage,
            "portfolio_rec": portfolio_rec,
        }

    def run_batch(
        self,
        start_year: int = 1965,
        end_year: int = 2020,
        force_refresh: bool = False,
        batch_size: int = 12,
    ) -> dict[str, Any]:
        """Run backtest across a multi-year range with checkpointing.

        Processes months oldest-first.  Completed months are skipped automatically
        unless force_refresh=True.  Set batch_size to limit how many months run
        per invocation (useful for scheduling).

        Args:
            start_year: First year of the backtest range.
            end_year: Last year of the backtest range (inclusive).
            force_refresh: Recompute already-completed months.
            batch_size: Max months to process in this invocation.

        Returns:
            Dict with: processed, skipped, errors, next_month.
        """

        months = self._generate_month_range(start_year, end_year)
        processed, skipped, errors = 0, 0, 0
        last_month = None

        for ym in months:
            if processed >= batch_size:
                break
            try:
                result = self.run_month(ym, force_refresh=force_refresh)
                if result["status"] == "skipped_cached":
                    skipped += 1
                else:
                    processed += 1
                last_month = ym
            except Exception as exc:
                errors += 1
                self.log_audit("backtest_batch_error", ym, str(exc), error=str(exc))

        remaining = len(months) - processed - skipped
        return {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "last_month": last_month,
            "remaining_estimate": max(0, remaining),
        }

    def score_run(self, run_id: int) -> dict[str, Any] | None:
        """Score a backtest run against actual data if 10 years have passed.

        Loads the original portfolio recommendation, fetches 10-year actual
        returns, computes metrics, writes a BacktestScore row, and extracts
        lessons.

        Args:
            run_id: Primary key of the BacktestRun row.

        Returns:
            Dict with scoring results, or None if data not yet available.
        """

        with get_session() as session:
            run = session.query(BacktestRun).filter_by(id=run_id).first()
        if not run:
            return None

        year = int(run.year_month[:4])
        month = int(run.year_month[5:7])
        ten_yr_year = year + 10
        if ten_yr_year > date.today().year:
            return None

        actual_data = self._fetch_actual_10yr_returns(run.year_month)
        score_metrics = self._compute_score_metrics(run.portfolio_recommendation, actual_data)
        root_cause = self._analyze_root_cause(run, score_metrics, actual_data)
        lessons = self._extract_lessons(run, score_metrics, root_cause)

        with get_session() as session:
            bt_score = BacktestScore(
                run_id=run_id,
                score_date=datetime.now(timezone.utc),
                actual_10yr_data=actual_data,
                return_vs_spy=score_metrics.get("return_vs_spy"),
                brier_score=score_metrics.get("brier_score"),
                wins=score_metrics.get("wins"),
                losses=score_metrics.get("losses"),
                root_cause_analysis=root_cause,
                missing_signals=score_metrics.get("missing_signals"),
            )
            session.add(bt_score)

        for lesson in lessons:
            self._save_lesson(lesson)

        return score_metrics

    def score_all_eligible(self) -> dict[str, int]:
        """Score all backtest runs that now have 10 years of actual data available."""

        with get_session() as session:
            runs = session.query(BacktestRun).all()

        scored, skipped = 0, 0
        for run in runs:
            year = int(run.year_month[:4])
            if year + 10 <= date.today().year:
                result = self.score_run(run.id)
                if result:
                    scored += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        return {"scored": scored, "skipped": skipped}

    def get_win_rate(self, era: str | None = None) -> dict[str, Any]:
        """Return win-rate statistics from scored backtest runs.

        Args:
            era: Optional era filter, e.g. "Cold War", "Post-Soviet", "Post-2008".

        Returns:
            Dict with win_rate, sample_size, avg_return_vs_spy, by_era.
        """

        with get_session() as session:
            scores = session.query(BacktestScore).all()
            lessons = session.query(BacktestLesson).all()

        if not scores:
            return {"win_rate": None, "sample_size": 0, "message": "No scored runs yet."}

        wins = sum(1 for s in scores if s.return_vs_spy is not None and s.return_vs_spy > 0)
        returns = [s.return_vs_spy for s in scores if s.return_vs_spy is not None]
        avg_return = sum(returns) / len(returns) if returns else None

        by_era: dict[str, dict] = {}
        for lesson in lessons:
            era_key = lesson.era or "unknown"
            if era and era_key != era:
                continue
            if era_key not in by_era:
                by_era[era_key] = {"wins": 0, "total": 0}
            by_era[era_key]["total"] += 1
            if lesson.lesson_type == "win":
                by_era[era_key]["wins"] += 1

        return {
            "win_rate": round(wins / len(scores), 3) if scores else None,
            "sample_size": len(scores),
            "avg_return_vs_spy": round(avg_return, 3) if avg_return is not None else None,
            "by_era": by_era,
        }

    def promote_lessons_to_kb(self, min_validation_count: int = 3) -> int:
        """Promote validated lessons to the Forecaster KB file.

        Lessons with validation_count >= min_validation_count are appended
        to kb/forecaster/memory.md under a "## Backtested Lessons" section.

        Args:
            min_validation_count: Minimum number of validating examples.

        Returns:
            Number of lessons promoted.
        """

        from pathlib import Path

        with get_session() as session:
            lessons = (
                session.query(BacktestLesson)
                .filter(BacktestLesson.validation_count >= min_validation_count)
                .all()
            )

        if not lessons:
            return 0

        kb_path = Path("kb/forecaster/memory.md")
        kb_content = kb_path.read_text(encoding="utf-8") if kb_path.exists() else ""
        section_header = "\n## Backtested Lessons\n"

        if section_header not in kb_content:
            kb_content += section_header
            kb_content += "| Era | Type | Signal Combination | Outcome | Validated |\n"
            kb_content += "|-----|------|--------------------|---------|----------|\n"

        promoted = 0
        for lesson in lessons:
            signals = json.dumps(lesson.signal_combination or {})
            row = f"| {lesson.era} | {lesson.lesson_type} | {signals[:60]} | {lesson.outcome[:80]} | {lesson.validation_count} |\n"
            if row not in kb_content:
                kb_content += row
                promoted += 1

        kb_path.write_text(kb_content, encoding="utf-8")
        return promoted

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_fred_snapshot(self, year_month: str) -> dict[str, Any]:
        """Fetch historical FRED data AS OF the given month and build a snapshot dict.

        Uses end_date to limit observations so the snapshot contains only data
        that would have been available at that point in time.  This is critical for
        a valid backtest — fetching today's data would look-ahead.
        """

        year = int(year_month[:4])
        month = int(year_month[5:7])
        # Last day of the month (conservative: always use 28 to avoid weekend issues)
        end_date = f"{year}-{month:02d}-28"
        # Pull 5 years of history so we have enough for pct_change(12) and trend analysis
        start_date = f"{year - 5}-{month:02d}-01"

        snapshot: dict[str, Any] = {"year_month": year_month}
        for series_id, name in BACKTEST_FRED_SERIES.items():
            try:
                df = self.fred.fetch(series_id, start_date=start_date, end_date=end_date, periods=60)
                if not df.empty:
                    snapshot[name] = float(df["value"].iloc[-1])
                    snapshot[f"{name}_1yr_change"] = (
                        float(df["value"].pct_change(12).iloc[-1])
                        if len(df) >= 13
                        else None
                    )
                else:
                    snapshot[name] = None
                    snapshot[f"{name}_1yr_change"] = None
            except Exception:
                snapshot[name] = None
                snapshot[f"{name}_1yr_change"] = None
        return snapshot

    def _get_political_context(self, year_month: str) -> str:
        """Get cached or freshly generated political context for a month."""

        try:
            return self.geo_pol.get_monthly_summary(year_month)
        except Exception:
            return f"Political context unavailable for {year_month}."

    def _get_economic_context(self, year_month: str) -> str:
        """Get cached or freshly generated economic context for a month."""

        try:
            return self.geo_econ.get_monthly_summary(year_month)
        except Exception:
            return f"Economic context unavailable for {year_month}."

    def _determine_coverage(self, year_month: str) -> str:
        """Determine data coverage tier for a month."""

        year = int(year_month[:4])
        if year >= 2015:
            return "full"
        if year >= 1979:
            return "partial"
        return "sparse"

    def _build_mock_signal(self, year_month: str, fred_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Build a mock signal_output dict from historical FRED data."""

        return {
            "timestamp": f"{year_month}-15T00:00:00+00:00",
            "regime": self._classify_regime(fred_snapshot),
            "regime_summary": f"Historical reconstruction for {year_month}",
            "indicators": fred_snapshot,
            "top_anomalies": [],
            "data_quality_warnings": [f"Historical reconstruction — coverage: {self._determine_coverage(year_month)}"],
        }

    def _build_mock_market(self, year_month: str, fred_snapshot: dict[str, Any]) -> dict[str, Any]:
        """Build a mock market_output dict from historical FRED data."""

        vix = fred_snapshot.get("vix", 20.0) or 20.0
        oil = fred_snapshot.get("wti_oil", 40.0) or 40.0
        return {
            "timestamp": f"{year_month}-15T00:00:00+00:00",
            "risk_score": min(1.0, max(-1.0, (20.0 - float(vix)) / 15.0)),
            "risk_label": "risk_off" if float(vix) > 25 else "neutral" if float(vix) > 18 else "risk_on",
            "divergences": [],
            "metrics": {"vix": {"price": vix}, "oil": {"price": oil}},
        }

    def _build_mock_historian(
        self, year_month: str, political_summary: str, economic_summary: str
    ) -> dict[str, Any]:
        """Build a mock historian_output dict from geo context summaries."""

        return {
            "timestamp": f"{year_month}-15T00:00:00+00:00",
            "top_analogs": [],
            "regime_theme": "historical_reconstruction",
            "geopolitical_context": political_summary,
            "economic_context": economic_summary,
            "year_month": year_month,
        }

    def _generate_portfolio_call(
        self,
        year_month: str,
        forecast: dict[str, Any],
        political_summary: str,
        economic_summary: str,
    ) -> dict[str, Any]:
        """Generate a portfolio recommendation from the forecast output."""

        user_message = json.dumps(
            {
                "year_month": year_month,
                "forecast": forecast,
                "political_context": political_summary[:500],
                "economic_context": economic_summary[:500],
                "instruction": (
                    "Given the forecast and context for this historical month, "
                    "what would have been the optimal portfolio call? "
                    "Return JSON with keys: equities (overweight/neutral/underweight), "
                    "bonds (overweight/neutral/underweight), gold (overweight/neutral/underweight), "
                    "oil (overweight/neutral/underweight), cash (0-100 pct), "
                    "top_3_positions (list of str), rationale (str), horizon_months (int)."
                ),
            },
            default=str,
        )

        raw = self.call_llm(
            system_prompt=(
                "You are a portfolio manager with deep macro expertise. "
                "Generate a historically grounded asset allocation call. "
                "Return valid JSON only."
            ),
            user_message=user_message,
            include_kb=True,
            model_tier="reasoning",
        )

        return self.validate_json_output(
            raw,
            required_keys=["equities", "bonds", "rationale"],
        )

    def _fetch_actual_10yr_returns(self, year_month: str) -> dict[str, Any]:
        """Fetch actual 10-year market returns starting from year_month.

        Uses yfinance for broad market indices.  Returns approximate returns.
        """

        year = int(year_month[:4])
        month = int(year_month[5:7])
        target_year = year + 10
        target_ym = f"{target_year}-{month:02d}"

        tickers = {"^GSPC": "spy", "GC=F": "gold", "CL=F": "oil", "^TYX": "bonds_30yr"}
        actuals: dict[str, Any] = {"start_month": year_month, "end_month": target_ym}

        for ticker, name in tickers.items():
            try:
                df = self.yf.fetch(ticker, period="10y")
                if not df.empty and len(df) > 1:
                    start_price = float(df["close"].iloc[0])
                    end_price = float(df["close"].iloc[-1])
                    total_return = (end_price - start_price) / start_price
                    actuals[f"{name}_10yr_return"] = round(total_return, 4)
                else:
                    actuals[f"{name}_10yr_return"] = None
            except Exception:
                actuals[f"{name}_10yr_return"] = None

        return actuals

    def _compute_score_metrics(
        self, portfolio_rec: dict[str, Any], actual_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute quantitative scoring metrics comparing recommendation to actual."""

        spy_return = actual_data.get("spy_10yr_return")
        gold_return = actual_data.get("gold_10yr_return")

        equity_stance = (portfolio_rec.get("equities") or "neutral").lower()
        gold_stance = (portfolio_rec.get("gold") or "neutral").lower()

        equity_contribution = 0.0
        if spy_return is not None:
            if equity_stance == "overweight":
                equity_contribution = spy_return * 0.6
            elif equity_stance == "neutral":
                equity_contribution = spy_return * 0.4
            else:
                equity_contribution = spy_return * 0.1

        gold_contribution = 0.0
        if gold_return is not None:
            if gold_stance == "overweight":
                gold_contribution = gold_return * 0.2
            elif gold_stance == "neutral":
                gold_contribution = gold_return * 0.1

        estimated_return = equity_contribution + gold_contribution
        spy_benchmark = (spy_return or 0.0) * 0.6

        wins = []
        losses = []
        if spy_return is not None:
            if estimated_return > spy_benchmark:
                wins.append(f"Outperformed SPY benchmark ({estimated_return:.1%} vs {spy_benchmark:.1%})")
            else:
                losses.append(f"Underperformed SPY benchmark ({estimated_return:.1%} vs {spy_benchmark:.1%})")

        return {
            "estimated_10yr_return": round(estimated_return, 4),
            "return_vs_spy": round(estimated_return - spy_benchmark, 4),
            "brier_score": None,
            "wins": wins,
            "losses": losses,
            "missing_signals": None,
        }

    def _analyze_root_cause(
        self, run: Any, score_metrics: dict[str, Any], actual_data: dict[str, Any]
    ) -> str:
        """Use LLM to identify root causes of wins and losses."""

        user_message = json.dumps(
            {
                "year_month": run.year_month,
                "original_recommendation": run.portfolio_recommendation,
                "actual_10yr_outcomes": actual_data,
                "score_metrics": score_metrics,
                "political_context": (run.political_context_summary or "")[:300],
                "economic_context": (run.economic_context_summary or "")[:300],
                "question": (
                    "What signals were present in the data that would have improved this forecast? "
                    "What signals were misleading?  What historical analogies apply?"
                ),
            },
            default=str,
        )

        return self.call_llm(
            system_prompt=(
                "You are a macro portfolio post-mortem analyst. "
                "Given a historical forecast and what actually happened, identify "
                "root causes of errors and overlooked signals. "
                "Be specific and actionable."
            ),
            user_message=user_message,
            include_kb=True,
            model_tier="reasoning",
        )

    def _extract_lessons(
        self, run: Any, score_metrics: dict[str, Any], root_cause: str
    ) -> list[dict[str, Any]]:
        """Extract structured lessons from the root cause analysis."""

        lesson_type = "win" if (score_metrics.get("return_vs_spy") or 0) > 0 else "loss"
        era = self._classify_era(run.year_month)

        return [
            {
                "lesson_type": lesson_type,
                "era": era,
                "signal_combination": run.fred_snapshot,
                "outcome": str(score_metrics.get("wins") or score_metrics.get("losses") or ""),
                "root_cause": root_cause[:500],
                "applicable_scenarios": list(config.FORECAST_SCENARIOS),
                "confidence": "provisional",
                "validation_count": 1,
            }
        ]

    def _save_run(
        self,
        year_month: str,
        fred_snapshot: dict[str, Any],
        political_summary: str,
        economic_summary: str,
        portfolio_rec: dict[str, Any],
        confidence: float,
        coverage: str,
    ) -> int:
        """Insert or update a BacktestRun row and return its id."""

        with get_session() as session:
            existing = session.query(BacktestRun).filter_by(year_month=year_month).first()
            if existing:
                existing.fred_snapshot = fred_snapshot
                existing.political_context_summary = political_summary[:2000]
                existing.economic_context_summary = economic_summary[:2000]
                existing.portfolio_recommendation = portfolio_rec
                existing.confidence = confidence
                existing.context_coverage = coverage
                return existing.id
            row = BacktestRun(
                year_month=year_month,
                fred_snapshot=fred_snapshot,
                political_context_summary=political_summary[:2000],
                economic_context_summary=economic_summary[:2000],
                portfolio_recommendation=portfolio_rec,
                confidence=confidence,
                context_coverage=coverage,
            )
            session.add(row)
            session.flush()
            return row.id

    def _save_lesson(self, lesson: dict[str, Any]) -> None:
        """Upsert a lesson into backtest_lessons table."""

        with get_session() as session:
            existing = (
                session.query(BacktestLesson)
                .filter_by(
                    lesson_type=lesson["lesson_type"],
                    era=lesson["era"],
                    outcome=lesson["outcome"][:255],
                )
                .first()
            )
            if existing:
                existing.validation_count += 1
                existing.confidence = "established" if existing.validation_count >= 3 else "provisional"
            else:
                session.add(BacktestLesson(
                    lesson_type=lesson["lesson_type"],
                    era=lesson["era"],
                    signal_combination=lesson.get("signal_combination"),
                    outcome=lesson["outcome"][:500],
                    root_cause=lesson["root_cause"][:500],
                    applicable_scenarios=lesson.get("applicable_scenarios"),
                    confidence=lesson.get("confidence", "provisional"),
                    validation_count=lesson.get("validation_count", 1),
                ))

    def _load_checkpoint(self, year_month: str) -> int | None:
        """Return the run_id if this month has already been processed."""

        with get_session() as session:
            row = session.query(BacktestRun).filter_by(year_month=year_month).first()
        return row.id if row else None

    @staticmethod
    def _classify_regime(fred_snapshot: dict[str, Any]) -> str:
        """Classify the macro regime from a FRED snapshot."""

        ff = fred_snapshot.get("fed_funds_rate") or 0.0
        cpi = fred_snapshot.get("cpi_1yr_change") or 0.0
        unrate = fred_snapshot.get("unemployment") or 5.0

        if float(cpi) > 0.06 and float(ff) > 0.04:
            return "stagflation"
        if float(ff) > 0.05:
            return "tightening"
        if float(ff) < 0.01:
            return "zirp"
        if float(unrate) > 0.08:
            return "recession"
        return "expansion"

    @staticmethod
    def _classify_era(year_month: str) -> str:
        """Classify a year_month into a geopolitical era."""

        year = int(year_month[:4])
        if year < 1973:
            return "Bretton Woods"
        if year < 1980:
            return "Stagflation"
        if year < 1991:
            return "Cold War Late"
        if year < 2001:
            return "Post-Soviet Expansion"
        if year < 2009:
            return "9/11 and Housing Boom"
        if year < 2017:
            return "Post-2008 Recovery"
        if year < 2021:
            return "Trade War and Covid"
        return "Post-Covid"

    @staticmethod
    def _generate_month_range(start_year: int, end_year: int) -> list[str]:
        """Generate a sorted list of YYYY-MM strings from start to end year."""

        months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                months.append(f"{year}-{month:02d}")
        return months
