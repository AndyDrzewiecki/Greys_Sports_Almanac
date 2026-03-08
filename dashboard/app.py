from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Streamlit dashboard — Macro Market Portfolio Alignment Tool."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(
    page_title="Macro Market Portfolio Alignment Tool",
    page_icon="📊",
    layout="wide",
)


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_latest_run() -> dict:
    """Load the most recent combined pipeline run JSON."""

    path = Path(config.LATEST_RUN_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=300)
def load_table(query: str) -> pd.DataFrame:
    """Load a SQL query result. Works with SQLite; for PostgreSQL, configure DB_URL."""

    try:
        from sqlalchemy import text
        from storage.database import ENGINE
        with ENGINE.connect() as conn:
            return pd.read_sql_query(text(query), conn)
    except Exception:
        return pd.DataFrame()


def badge(label: str, color: str) -> str:
    """Render a compact colored badge."""

    return (
        f"<div style='padding:0.75rem;border-radius:0.5rem;background:{color};"
        f"color:white;font-weight:600;text-align:center'>{label}</div>"
    )


def risk_color(level: str) -> str:
    """Map a risk/health level string to a display color."""

    palette = {
        "low": "#2e7d32",
        "stable": "#2e7d32",
        "strong": "#2e7d32",
        "medium": "#f9a825",
        "neutral": "#546e7a",
        "stressed": "#ef6c00",
        "high": "#c62828",
        "critical": "#7b1fa2",
        "crisis": "#7b1fa2",
        "dislocated": "#c62828",
        "tense": "#ef6c00",
    }
    return palette.get((level or "").lower(), "#546e7a")


# ── Load run data ─────────────────────────────────────────────────────────────

latest_run = load_latest_run() or {}
signal_output    = latest_run.get("signal_output")    or {}
market_output    = latest_run.get("market_output")    or {}
forecast_output  = latest_run.get("forecast_output")  or {}
skeptic_output   = latest_run.get("skeptic_output")   or {}
recommendations  = latest_run.get("recommendations")  or {}
geo_pol_output   = latest_run.get("geo_political_output") or {}
geo_econ_output  = latest_run.get("geo_economic_output")  or {}
daily_brief      = latest_run.get("daily_brief")      or ""
kb_synthesis     = latest_run.get("kb_synthesis")     or ""

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Macro Market Portfolio Alignment Tool")
ts = latest_run.get("timestamp", "No run available")
st.caption(f"Last updated: {ts}")

if not latest_run or not any([signal_output, market_output, forecast_output]):
    st.warning("No pipeline data yet. Click **Run Now** to start the first pipeline pass.")

col_run, col_health, _ = st.columns([2, 2, 8])
with col_run:
    if st.button("▶ Run Now", type="primary"):
        subprocess.run(["python", "run.py"], check=False)
        st.cache_data.clear()
        st.rerun()

with col_health:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── Top KPI row ───────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5 = st.columns(5)
regime = signal_output.get("regime", "—")
risk_label = market_output.get("risk_label", "—")
top_scenario = forecast_output.get("top_scenario", "—")
geo_regime = (geo_pol_output.get("synthesis") or {}).get("geopolitical_regime", "—")
econ_regime = (geo_econ_output.get("synthesis") or {}).get("global_economic_regime", "—")

with k1:
    st.markdown(badge(f"Macro: {regime}", risk_color(regime)), unsafe_allow_html=True)
with k2:
    st.markdown(badge(f"Risk: {risk_label}", risk_color(risk_label)), unsafe_allow_html=True)
with k3:
    top_p = (forecast_output.get("scenarios") or {}).get(top_scenario, 0)
    st.markdown(badge(f"Top: {top_scenario} {top_p:.0%}", "#1565c0"), unsafe_allow_html=True)
with k4:
    st.markdown(badge(f"Geo: {geo_regime}", risk_color(geo_regime)), unsafe_allow_html=True)
with k5:
    st.markdown(badge(f"Econ: {econ_regime}", risk_color(econ_regime)), unsafe_allow_html=True)

st.divider()

# ── Main tabs ─────────────────────────────────────────────────────────────────

(
    tab_brief, tab_macro, tab_geo_pol, tab_geo_econ,
    tab_backtest, tab_kb, tab_audit
) = st.tabs([
    "📋 Daily Brief", "📈 Macro Dashboard",
    "🌍 Political Map", "💹 Economic Flows",
    "📚 Backtest History", "🧠 KB Health", "🔎 Audit Log",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Daily Brief
# ══════════════════════════════════════════════════════════════════════════════
with tab_brief:
    st.subheader("Today's Investment Brief")
    if daily_brief:
        st.markdown(daily_brief)
    else:
        st.info("Daily brief will appear here after the first pipeline run.")

    if kb_synthesis:
        with st.expander("Cross-Agent Intelligence Synthesis"):
            st.markdown(kb_synthesis)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Macro Dashboard (existing content)
# ══════════════════════════════════════════════════════════════════════════════
with tab_macro:
    st.subheader("Scenario Probabilities")
    scenarios = forecast_output.get("scenarios") or {}
    if scenarios:
        scenario_df = pd.DataFrame(
            {"scenario": list(scenarios.keys()), "probability": list(scenarios.values())}
        ).sort_values("probability", ascending=True)
        st.bar_chart(data=scenario_df, x="scenario", y="probability", horizontal=True)
        st.caption(f"Confidence: {forecast_output.get('confidence', 0.0):.2f}  |  Horizon: {forecast_output.get('horizon', '—')}")
    else:
        st.info("No forecast data available yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Skeptic Verdict")
        verdict = skeptic_output.get("skeptic_verdict", "—") if skeptic_output else "—"
        st.markdown(badge(verdict.upper(), risk_color(verdict)), unsafe_allow_html=True)
        st.write(skeptic_output.get("why_forecast_could_be_wrong", "No skeptic memo.") if skeptic_output else "No skeptic memo.")
    with right:
        st.subheader("Top Anomalies")
        anomaly_table = []
        for name in (signal_output.get("top_anomalies") or [])[:6]:
            details = (signal_output.get("indicators") or {}).get(name, {})
            anomaly_table.append({
                "indicator": name,
                "value": details.get("value"),
                "5d Δ": details.get("pct_5d"),
                "status": details.get("status"),
            })
        if anomaly_table:
            st.table(pd.DataFrame(anomaly_table))
        else:
            st.info("No anomaly data.")

    st.subheader("Recommendations")
    macro_tab_inner, etf_tab, stock_tab = st.tabs(["Macro", "ETFs", "Stocks"])
    with macro_tab_inner:
        if recommendations.get("macro_positioning"):
            st.json(recommendations["macro_positioning"])
        st.write(recommendations.get("do_nothing_assessment", "") or "")
        st.caption("For research purposes only — not financial advice.")
    with etf_tab:
        etf_df = pd.DataFrame(recommendations.get("etf_suggestions") or [])
        if not etf_df.empty:
            st.dataframe(etf_df, use_container_width=True)
        else:
            st.info("No ETF suggestions yet.")
        st.caption("For research purposes only — not financial advice.")
    with stock_tab:
        stock_df = pd.DataFrame(recommendations.get("stock_ideas") or [])
        if not stock_df.empty:
            st.dataframe(stock_df, use_container_width=True)
        else:
            st.info("No stock ideas yet.")
        st.caption("For research purposes only — not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Political Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_geo_pol:
    st.subheader("Geopolitical Risk by Country")

    country_analyses = geo_pol_output.get("country_analyses") or {}
    synthesis = geo_pol_output.get("synthesis") or {}

    if country_analyses:
        # Build a display dataframe
        pol_rows = []
        for code, data in country_analyses.items():
            pol_rows.append({
                "country_code": code,
                "risk_level": data.get("risk_level", "unknown"),
                "regime_stability": data.get("regime_stability", "—"),
                "key_themes": ", ".join(data.get("key_themes") or [])[:80],
                "coverage": data.get("context_coverage", "—"),
            })
        pol_df = pd.DataFrame(pol_rows)

        # Plotly choropleth map
        try:
            import plotly.express as px
            risk_map = {"low": 1, "medium": 2, "high": 3, "critical": 4, "unknown": 0}
            pol_df["risk_num"] = pol_df["risk_level"].map(risk_map).fillna(0)
            fig = px.choropleth(
                pol_df,
                locations="country_code",
                color="risk_num",
                hover_name="country_code",
                hover_data={"risk_level": True, "key_themes": True, "risk_num": False},
                color_continuous_scale=["#2e7d32", "#f9a825", "#ef6c00", "#c62828", "#7b1fa2"],
                range_color=[0, 4],
                title=f"Geopolitical Risk — {geo_pol_output.get('year_month', '')}",
            )
            fig.update_layout(coloraxis_showscale=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Install plotly for the choropleth map: pip install plotly")

        st.dataframe(pol_df.drop(columns=["risk_num"], errors="ignore"), use_container_width=True)

        if synthesis:
            with st.expander("Synthesis"):
                st.write(synthesis.get("summary", ""))
                hot = synthesis.get("hotspots") or []
                if hot:
                    st.write("**Hotspots:**", ", ".join(hot))
                sfh = synthesis.get("safe_haven_drivers") or []
                if sfh:
                    st.write("**Safe-Haven Drivers:**", ", ".join(sfh))
    else:
        st.info("Geopolitical data will appear here after the first pipeline run with the new historians.")

    st.subheader("Top Geopolitical Risks")
    for risk in (geo_pol_output.get("top_risks") or []):
        st.warning(risk)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Economic Flows
# ══════════════════════════════════════════════════════════════════════════════
with tab_geo_econ:
    st.subheader("Country Economic Health")

    econ_analyses = geo_econ_output.get("country_analyses") or {}
    econ_synthesis = geo_econ_output.get("synthesis") or {}

    if econ_analyses:
        econ_rows = []
        for code, data in econ_analyses.items():
            wb = data.get("wb_snapshot") or {}
            econ_rows.append({
                "country": code,
                "health": data.get("economic_health", "—"),
                "currency_stress": data.get("currency_stress", "—"),
                "debt_level": data.get("debt_level", "—"),
                "gdp_growth %": f"{wb.get('gdp_growth_pct', '—'):.1f}" if isinstance(wb.get("gdp_growth_pct"), float) else "—",
                "inflation %": f"{wb.get('inflation_cpi_pct', '—'):.1f}" if isinstance(wb.get("inflation_cpi_pct"), float) else "—",
                "coverage": data.get("context_coverage", "—"),
            })
        econ_df = pd.DataFrame(econ_rows)
        st.dataframe(econ_df, use_container_width=True)

        # Debt stress / currency stress bar chart
        try:
            import plotly.express as px
            stress_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            econ_df["currency_stress_num"] = econ_df["currency_stress"].map(stress_map).fillna(0)
            fig2 = px.bar(
                econ_df,
                x="country",
                y="currency_stress_num",
                color="currency_stress",
                title="Currency Stress by Country",
                color_discrete_map={"low": "#2e7d32", "medium": "#f9a825", "high": "#c62828", "critical": "#7b1fa2"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            pass

        if econ_synthesis:
            with st.expander("Economic Synthesis"):
                st.write(econ_synthesis.get("summary", ""))
                debt_countries = econ_synthesis.get("debt_stress_countries") or []
                if debt_countries:
                    st.write("**Debt stress countries:**", ", ".join(debt_countries))
                currency_countries = econ_synthesis.get("currency_stress_countries") or []
                if currency_countries:
                    st.write("**Currency stress countries:**", ", ".join(currency_countries))

        st.subheader("Trade Dependencies")
        trade_deps = geo_econ_output.get("trade_dependencies") or []
        if trade_deps:
            dep_rows = [
                {
                    "A → B": f"{d.get('country_a')} → {d.get('country_b')}",
                    "avg exports A (% GDP)": d.get("avg_exports_a_pct_gdp"),
                    "avg imports A (% GDP)": d.get("avg_imports_a_pct_gdp"),
                }
                for d in trade_deps
                if not d.get("error")
            ]
            if dep_rows:
                st.dataframe(pd.DataFrame(dep_rows), use_container_width=True)
    else:
        st.info("Economic flow data will appear here after the first pipeline run with the new historians.")

    st.subheader("Top Economic Risks")
    for risk in (geo_econ_output.get("top_risks") or []):
        st.warning(risk)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Backtest History
# ══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    st.subheader("Backtest Engine — Historical Training Progress")

    bt_runs = load_table(
        "SELECT year_month, confidence, context_coverage, run_at FROM backtest_runs ORDER BY year_month DESC LIMIT 50"
    )
    bt_scores = load_table(
        "SELECT run_id, return_vs_spy, brier_score, scored_at FROM backtest_scores ORDER BY scored_at DESC LIMIT 50"
    )
    bt_lessons = load_table(
        "SELECT era, lesson_type, outcome, root_cause, validation_count, confidence FROM backtest_lessons ORDER BY validation_count DESC LIMIT 30"
    )

    if not bt_runs.empty:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Months Processed", len(bt_runs))
        with col_b:
            eligible_scored = len(bt_scores) if not bt_scores.empty else 0
            st.metric("Scored Runs", eligible_scored)
        with col_c:
            if not bt_scores.empty and "return_vs_spy" in bt_scores.columns:
                wins = (bt_scores["return_vs_spy"] > 0).sum()
                win_rate = wins / len(bt_scores) if len(bt_scores) > 0 else 0
                st.metric("Win Rate vs SPY", f"{win_rate:.1%}")
            else:
                st.metric("Win Rate vs SPY", "—")

        st.write("**Recent Backtest Runs**")
        st.dataframe(bt_runs, use_container_width=True)

        if not bt_scores.empty:
            st.write("**Scored Runs — Return vs SPY**")
            try:
                import plotly.express as px
                fig3 = px.bar(
                    bt_scores,
                    x="run_id",
                    y="return_vs_spy",
                    color=bt_scores["return_vs_spy"].apply(lambda v: "Win" if (v or 0) > 0 else "Loss"),
                    color_discrete_map={"Win": "#2e7d32", "Loss": "#c62828"},
                    title="10-Year Return vs SPY by Backtest Run",
                )
                st.plotly_chart(fig3, use_container_width=True)
            except ImportError:
                st.dataframe(bt_scores, use_container_width=True)
    else:
        st.info(
            "No backtest runs yet.  Start the engine:\n\n"
            "```\npython run_backtest.py --start-year 1965 --end-year 2020 --batch-size 12\n```"
        )

    if not bt_lessons.empty:
        st.subheader("Accumulated Lessons")
        st.dataframe(bt_lessons, use_container_width=True)

        with st.expander("Promote Validated Lessons to KB"):
            if st.button("Promote lessons with ≥ 3 validations to Forecaster KB"):
                from agents.backtester.engine import BacktestEngine
                count = BacktestEngine().promote_lessons_to_kb(min_validation_count=3)
                st.success(f"Promoted {count} lessons to kb/forecaster/memory.md")
    else:
        st.caption("Lessons will accumulate as backtest runs are scored.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: KB Health
# ══════════════════════════════════════════════════════════════════════════════
with tab_kb:
    st.subheader("Knowledge Base Health")

    staged_updates = load_table(
        "SELECT id, timestamp, agent_name, update_type, content, status FROM kb_updates "
        "WHERE status = 'staging' ORDER BY timestamp DESC"
    )
    st.write("**Staged notes awaiting review**")
    if staged_updates.empty:
        st.info("No staged KB updates.")
    else:
        for row in staged_updates.to_dict(orient="records"):
            from agents.base_agent import BaseAgent as _BA
            cols = st.columns([2, 6, 2])
            cols[0].write(f"{row['agent_name']} #{row['id']}")
            cols[1].write(row["content"])
            if cols[2].button("Promote", key=f"promote-{row['id']}"):
                _BA(row["agent_name"], f"kb/{row['agent_name']}").promote_kb_note(int(row["id"]))
                st.cache_data.clear()
                st.success(f"Promoted note {row['id']}.")
                st.rerun()

    forecaster_trend = load_table(
        "SELECT timestamp, brier_score FROM forecasts WHERE brier_score IS NOT NULL ORDER BY timestamp DESC LIMIT 20"
    )
    if not forecaster_trend.empty:
        st.subheader("Forecaster Accuracy Trend (Brier Score)")
        trend = forecaster_trend.sort_values("timestamp")
        st.line_chart(trend.set_index("timestamp")["brier_score"])
    else:
        st.caption("Brier scores will appear after forecasts have been scored against outcomes.")

    # KB file sizes as a proxy for content richness
    st.subheader("KB File Inventory")
    kb_root = Path("kb")
    kb_files = []
    for md_file in sorted(kb_root.rglob("memory.md")):
        try:
            lines = len(md_file.read_text(encoding="utf-8").splitlines())
            kb_files.append({"agent": md_file.parent.name, "file": str(md_file), "lines": lines})
        except OSError:
            pass
    if kb_files:
        st.dataframe(pd.DataFrame(kb_files), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: Audit Log
# ══════════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("Agent Audit Trail")
    history = load_table(
        "SELECT timestamp, agent_name, event_type, error FROM audit_log ORDER BY timestamp DESC LIMIT 50"
    )
    if not history.empty:
        error_mask = history["error"].notna() & (history["error"] != "")
        st.write(f"Last 50 events — {error_mask.sum()} with errors")
        st.dataframe(
            history.style.apply(
                lambda row: ["background-color: #ffebee" if row["error"] else "" for _ in row],
                axis=1,
            ),
            use_container_width=True,
        )
    else:
        st.info("No audit history available yet.")

    st.caption(f"Database: {config.DB_PATH} (SQLite) or DB_URL env var (PostgreSQL)")
