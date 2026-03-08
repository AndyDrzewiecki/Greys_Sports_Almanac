"""CLI entry point for the BacktestEngine.

Usage examples:
    # Run a single historical month
    python run_backtest.py --month 1972-03

    # Run a full-year batch (processes up to 12 months, then checkpoint)
    python run_backtest.py --start-year 1965 --end-year 2020 --batch-size 12

    # Score all eligible runs (those with >= 10 years of actual data)
    python run_backtest.py --score

    # Promote validated lessons (3+ examples) to the Forecaster KB
    python run_backtest.py --promote-lessons

    # Show current win-rate statistics
    python run_backtest.py --stats

Tip: wrap in a loop script or cron job to work through the 660-month backlog overnight.
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.backtester.engine import BacktestEngine
from storage.database import health_check, init_db


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BacktestEngine CLI — train the Macro Market Portfolio Alignment Tool on history."
    )
    parser.add_argument("--month", type=str, help="Run a single month, e.g. 1972-03.")
    parser.add_argument("--start-year", type=int, default=1965, help="Start year for batch mode.")
    parser.add_argument("--end-year", type=int, default=2020, help="End year for batch mode.")
    parser.add_argument("--batch-size", type=int, default=12, help="Max months to process per run.")
    parser.add_argument("--force", action="store_true", help="Re-run already-completed months.")
    parser.add_argument("--score", action="store_true", help="Score all eligible backtest runs.")
    parser.add_argument("--promote-lessons", action="store_true", help="Promote validated lessons to KB.")
    parser.add_argument("--stats", action="store_true", help="Print win-rate statistics.")
    parser.add_argument("--health", action="store_true", help="Print database health check.")
    args = parser.parse_args()

    init_db()

    if args.health:
        _print_json(health_check())
        return

    engine = BacktestEngine()

    if args.stats:
        _print_json(engine.get_win_rate())
        return

    if args.promote_lessons:
        count = engine.promote_lessons_to_kb(min_validation_count=3)
        print(f"Promoted {count} lessons to kb/forecaster/memory.md")
        return

    if args.score:
        results = engine.score_all_eligible()
        _print_json(results)
        return

    if args.month:
        result = engine.run_month(args.month, force_refresh=args.force)
        _print_json(result)
        return

    # Default: batch mode
    results = engine.run_batch(
        start_year=args.start_year,
        end_year=args.end_year,
        force_refresh=args.force,
        batch_size=args.batch_size,
    )
    _print_json(results)


if __name__ == "__main__":
    main()
