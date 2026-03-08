# Market Watch Agents

A multi-agent financial research system that monitors macro stress signals, generates scenario forecasts, and produces stock, ETF, and macro research ideas with adversarial review before anything reaches the dashboard.

The system is designed as a daily pipeline. It pulls macro and market data from free sources, classifies the current regime, compares the present setup with historical analogs, generates scenario probabilities, runs a Skeptic pass, and then creates research-oriented recommendations only if the forecast survives review.

Each major agent maintains its own markdown knowledge base and writes structured events to SQLite so the system can score itself over time. Weekly meta-evaluation updates calibration notes, tracks hit rates, and stages improvement suggestions for human review before those notes are promoted into active KB context.

## Setup
1. Clone the repo.
2. Run `pip install -r requirements.txt`.
3. Keep your existing `.env` file in place and fill in your API keys if needed.
4. Get a free FRED API key at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).
5. Get an Anthropic API key at [console.anthropic.com](https://console.anthropic.com).
6. Run `python run.py` to execute the pipeline once.
7. Run `streamlit run dashboard/app.py` to open the dashboard.

## Agent Architecture
- `SignalWatcher` fetches macro and stress indicators, validates them, classifies each reading, and determines the overall regime.
- `MarketWatcher` tracks cross-asset leadership, relative performance, and divergences to estimate a composite risk-on or risk-off score.
- `Historian` matches the current fingerprint to curated historical episodes and warns against weak analogs.
- `Forecaster` turns upstream context into scenario probabilities with explicit drivers, invalidations, and confidence.
- `Skeptic` attacks the forecast, checks for overconfidence, and can approve, caution, or reject the output.
- `Recommender` converts approved scenarios into macro posture, ETF ideas, and stock idea generation for research.
- `MetaEvaluator` scores old forecasts and recommendations, measures agent reliability, and stages KB improvements.

## Knowledge Base System
Each agent has a `kb/<agent_name>/memory.md` file that stores durable notes, historical tables, and calibration guidance. New self-improvement notes are written to staging first and also recorded in the `kb_updates` database table.

The Streamlit dashboard exposes staged KB notes in the **KB Health** section. Use **Promote to Active** only after reviewing whether the note reflects a real recurring pattern rather than noise or one-off market behavior.

## Important Disclaimers
This system is for research and educational purposes only.
Nothing produced by this system constitutes financial advice.
Past model performance does not predict future results.
Always consult a licensed financial professional before making investment decisions.

## API Keys Required
- Anthropic (paid): [console.anthropic.com](https://console.anthropic.com)
- FRED (free): [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
- yfinance: no key needed

## Data Sources
- Anthropic API: [anthropic.com](https://www.anthropic.com)
- FRED API: [fred.stlouisfed.org](https://fred.stlouisfed.org)
- Yahoo Finance via `yfinance`: [finance.yahoo.com](https://finance.yahoo.com)
- RSS parsing via `feedparser`: [feedparser.readthedocs.io](https://feedparser.readthedocs.io/)
