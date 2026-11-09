"""Streamlit Portfolio Analytics Dashboard powered by portfolio_database.db and PortfolioAgent."""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from agent import PortfolioAgent
from exposure_tool import calculate_sector_exposure
from risk_analytics import (
    compute_all_portfolios_statistical_summary,
    create_risk_return_matplotlib_figure,
    create_sector_and_correlation_seaborn_figure,
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "portfolio_database.db")

st.set_page_config(
    page_title="Portfolio Analytics Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_agent() -> PortfolioAgent:
    """Create a cached PortfolioAgent instance connected in read-only mode."""
    return PortfolioAgent(db_path=DB_PATH)


def query_df(agent: PortfolioAgent, sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query through the agent's secure execute_sql method and return a DataFrame."""
    res = agent.execute_sql(sql)
    if "error" in res:
        st.error(res["user_message"])
        return pd.DataFrame()
    return pd.DataFrame(res.get("rows", []), columns=res.get("columns", []))


def compute_concentration_table(agent: PortfolioAgent) -> pd.DataFrame:
    """
    Compute factual concentration and diversification indicators for all 13 portfolios
    using existing database columns (holding weights, asset types, and sector mappings).
    Flags high single-holding concentration (>25% weight) or high single-sector equity concentration (>60% weight).
    """
    portfolios_df = query_df(
        agent,
        """
        SELECT p.portfolio_id, p.portfolio_name, p.target_risk_level, p.total_aum, p.status
        FROM portfolios p
        ORDER BY p.portfolio_id ASC;
        """,
    )
    holdings_df = query_df(
        agent,
        """
        SELECT h.portfolio_id, s.symbol, s.company_name, s.asset_type,
               COALESCE(sec.sector_name, 'Fixed Income / Bond') AS sector_name,
               h.current_weight, h.cost_basis, (h.quantity * s.current_price) AS market_value
        FROM holdings h
        JOIN securities s ON h.security_id = s.security_id
        LEFT JOIN sectors sec ON s.sector_id = sec.sector_id;
        """,
    )
    if portfolios_df.empty or holdings_df.empty:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []
    for _, p_row in portfolios_df.iterrows():
        pid = p_row["portfolio_id"]
        pname = p_row["portfolio_name"]
        p_holdings = holdings_df[holdings_df["portfolio_id"] == pid]

        total_holdings = len(p_holdings)
        stock_holdings = p_holdings[p_holdings["asset_type"] == "Stock"]
        distinct_stock_sectors = int(stock_holdings["sector_name"].nunique()) if not stock_holdings.empty else 0
        div_ratio = round(distinct_stock_sectors / total_holdings, 3) if total_holdings > 0 else 0.0

        # Largest holding by portfolio weight & NumPy HHI concentration
        if not p_holdings.empty:
            weights_np = p_holdings["current_weight"].to_numpy(dtype=np.float64)
            w_sum = float(np.sum(weights_np))
            norm_w = weights_np / w_sum if w_sum > 0 else weights_np
            hhi_val = round(float(np.sum(np.square(norm_w))), 4)
            top_h = p_holdings.sort_values(by="current_weight", ascending=False).iloc[0]
            top_holding_label = f"{top_h['symbol']} ({top_h['current_weight'] * 100:.1f}%)"
            max_holding_pct = round(float(np.max(weights_np)) * 100, 2)
        else:
            hhi_val = 0.0
            top_holding_label = "None"
            max_holding_pct = 0.0

        # Largest equity sector exposure
        exp_res = calculate_sector_exposure(agent.db_connection, pname)
        exposures = exp_res.get("sector_exposures", {})
        if exposures:
            top_sector, top_sector_pct = next(iter(exposures.items()))
            top_sector_label = f"{top_sector} ({top_sector_pct:.1f}%)"
        else:
            top_sector_pct = 0.0
            top_sector_label = "Bonds Only (0% Equities)"

        flags: List[str] = []
        if top_sector_pct >= 60.0:
            flags.append(f"High Sector Concentration ({top_sector}: {top_sector_pct:.0f}%)")
        if max_holding_pct >= 25.0:
            flags.append(f"High Single-Holding Weight ({max_holding_pct:.0f}%)")
        if distinct_stock_sectors in (1, 2) and not stock_holdings.empty:
            flags.append(f"Low Sector Breadth ({distinct_stock_sectors} sector{'s' if distinct_stock_sectors > 1 else ''})")

        records.append(
            {
                "Portfolio": pname,
                "Status": p_row["status"],
                "Target Risk": p_row["target_risk_level"],
                "Total AUM ($)": float(p_row["total_aum"]),
                "Total Holdings": total_holdings,
                "Equity Sectors": distinct_stock_sectors,
                "Diversification Ratio": div_ratio,
                "HHI (NumPy)": hhi_val,
                "Largest Holding": top_holding_label,
                "Top Equity Sector": top_sector_label,
                "Concentration Alert": "; ".join(flags) if flags else "Well Diversified",
            }
        )

    return pd.DataFrame(records)


def format_answer_text(question: str, res: Dict[str, Any]) -> str:
    """Ensure natural, domain-specific wording for agent answers displayed in the UI."""
    answer = str(res.get("answer") or "")
    q_lower = question.strip().lower()
    raw = res.get("raw_result")

    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], (tuple, list)) and len(raw[0]) == 1:
        val = raw[0][0]
        if ("how many" in q_lower or "number of" in q_lower or "count" in q_lower) and "portfolio" in q_lower:
            if "active" in q_lower:
                return f"There are {val} active portfolios in total."
            return f"There are {val} portfolios in total."

    if "matching records in total" in answer and "portfolio" in q_lower:
        answer = answer.replace("matching records in total", "portfolios in total")

    return answer


def main() -> None:
    agent = get_agent()

    st.title("Portfolio Analytics & Conversational AI Agent")
    st.caption("Connected to read-only SQLite database (`portfolio_database.db`) with Explainable AI routing")

    # Top-level KPI summary metrics
    kpi_df = query_df(
        agent,
        """
        SELECT COUNT(*) AS total_portfolios,
               SUM(CASE WHEN status = 'Active' THEN 1 ELSE 0 END) AS active_portfolios,
               SUM(total_aum) AS total_aum,
               SUM(CASE WHEN target_risk_level = 'High' THEN total_aum ELSE 0 END) AS high_risk_aum
        FROM portfolios;
        """,
    )
    if not kpi_df.empty:
        row = kpi_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Portfolios", int(row["total_portfolios"]))
        c2.metric("Active Portfolios", int(row["active_portfolios"]))
        c3.metric("Total Managed AUM", f"${float(row['total_aum']):,.0f}")
        c4.metric("High-Risk Portfolio AUM", f"${float(row['high_risk_aum']):,.0f}")

    tab_overview, tab_sector, tab_risk, tab_compare, tab_chat = st.tabs(
        [
            "Active Portfolio Overview",
            "Sector Allocation",
            "Risk & Concentration Analysis",
            "Portfolio Comparison",
            "Conversational AI Agent (Explainable AI)",
        ]
    )

    # TAB 1: Active Portfolio Overview
    with tab_overview:
        st.subheader("Portfolios Overview")
        status_filter = st.radio(
            "Filter by Status:",
            options=["Active Only", "All Portfolios (Active & Passive)"],
            horizontal=True,
        )
        where_clause = "WHERE p.status = 'Active'" if status_filter == "Active Only" else ""
        overview_df = query_df(
            agent,
            f"""
            SELECT p.portfolio_id AS ID,
                   p.portfolio_name AS Portfolio,
                   p.status AS Status,
                   p.target_risk_level AS Risk_Level,
                   p.strategy_type AS Strategy,
                   p.benchmark_index AS Benchmark,
                   p.total_aum AS Total_AUM,
                   COUNT(h.holding_id) AS Holdings_Count,
                   ROUND(COALESCE(SUM(h.quantity * s.current_price), 0.0), 2) AS Holdings_Market_Value
            FROM portfolios p
            LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id
            LEFT JOIN securities s ON h.security_id = s.security_id
            {where_clause}
            GROUP BY p.portfolio_id, p.portfolio_name, p.status, p.target_risk_level,
                     p.strategy_type, p.benchmark_index, p.total_aum
            ORDER BY p.total_aum DESC;
            """,
        )
        st.dataframe(overview_df, width="stretch", hide_index=True)

        if not overview_df.empty:
            st.subheader("Assets Under Management (AUM) by Portfolio")
            chart_df = overview_df.set_index("Portfolio")[["Total_AUM"]]
            st.bar_chart(chart_df)

    # TAB 2: Sector Allocation Charts
    with tab_sector:
        st.subheader("Equity Sector Exposure Calculator (Excluding Bonds)")
        portfolios_list_df = query_df(agent, "SELECT portfolio_name FROM portfolios ORDER BY portfolio_name ASC;")
        portfolio_names = portfolios_list_df["portfolio_name"].tolist() if not portfolios_list_df.empty else []

        selected_portfolio = st.selectbox(
            "Select Portfolio for Sector Exposure Breakdown:",
            options=portfolio_names,
            index=portfolio_names.index("International Equity Fund") if "International Equity Fund" in portfolio_names else 0,
        )
        if selected_portfolio:
            exp_data = calculate_sector_exposure(agent.db_connection, selected_portfolio)
            exposures = exp_data.get("sector_exposures", {})
            col_left, col_right = st.columns([1, 1])
            with col_left:
                if exposures:
                    exp_df = pd.DataFrame(
                        [{"Sector": k, "Exposure (%)": v} for k, v in exposures.items()]
                    )
                    st.dataframe(exp_df, width="stretch", hide_index=True)
                else:
                    st.info(f"{selected_portfolio} holds only fixed-income bonds (0% equity sector exposure).")
            with col_right:
                if exposures:
                    st.bar_chart(exp_df.set_index("Sector")[["Exposure (%)"]])

            st.subheader(f"All Underlying Holdings in {selected_portfolio}")
            safe_p = selected_portfolio.replace("'", "''")
            p_holdings_df = query_df(
                agent,
                f"""
                SELECT h.holding_id AS Holding_ID,
                       s.symbol AS Symbol,
                       s.company_name AS Security_Name,
                       s.asset_type AS Asset_Type,
                       COALESCE(sec.sector_name, 'Bond (Excluded from Equity Exposure)') AS Sector,
                       h.quantity AS Quantity,
                       ROUND(h.current_weight * 100, 2) AS Weight_Pct,
                       h.cost_basis AS Cost_Basis,
                       ROUND(h.quantity * s.current_price, 2) AS Current_Market_Value
                FROM holdings h
                JOIN portfolios p ON h.portfolio_id = p.portfolio_id
                JOIN securities s ON h.security_id = s.security_id
                LEFT JOIN sectors sec ON s.sector_id = sec.sector_id
                WHERE p.portfolio_name = '{safe_p}'
                ORDER BY h.cost_basis DESC;
                """,
            )
            st.dataframe(p_holdings_df, width="stretch", hide_index=True)

    # TAB 3: Risk & Concentration Analysis
    with tab_risk:
        st.subheader("Portfolio Concentration & Diversification Indicators")
        st.markdown(
            "Flags portfolios with **High Sector Concentration** (≥60% in a single equity sector), "
            "**High Single-Holding Weight** (≥25% in one security), or **Low Sector Breadth** (≤2 equity sectors)."
        )
        conc_df = compute_concentration_table(agent)
        if not conc_df.empty:
            st.dataframe(conc_df, width="stretch", hide_index=True)

        st.subheader("Largest Holdings Across All Portfolios (by Cost Basis)")
        top_holdings_df = query_df(
            agent,
            """
            SELECT p.portfolio_name AS Portfolio,
                   s.symbol AS Symbol,
                   s.company_name AS Company,
                   s.asset_type AS Asset_Type,
                   COALESCE(sec.sector_name, 'Bond') AS Sector,
                   h.quantity AS Quantity,
                   ROUND(h.current_weight * 100, 2) AS Portfolio_Weight_Pct,
                   h.cost_basis AS Cost_Basis,
                   ROUND(h.quantity * s.current_price, 2) AS Market_Value
            FROM holdings h
            JOIN portfolios p ON h.portfolio_id = p.portfolio_id
            JOIN securities s ON h.security_id = s.security_id
            LEFT JOIN sectors sec ON s.sector_id = sec.sector_id
            ORDER BY h.cost_basis DESC
            LIMIT 10;
            """,
        )
        st.dataframe(top_holdings_df, width="stretch", hide_index=True)

        st.subheader("Latest Recorded Risk Metrics & Performance (From Database)")
        risk_perf_df = query_df(
            agent,
            """
            SELECT p.portfolio_name AS Portfolio,
                   r.calculation_date AS Date,
                   perf.nav AS NAV,
                   perf.total_return_1y AS Return_1Y_Pct,
                   perf.volatility AS Volatility,
                   perf.sharpe_ratio AS Sharpe_Ratio,
                   perf.max_drawdown AS Max_Drawdown,
                   r.var_95 AS VaR_95,
                   r.cvar_95 AS CVaR_95,
                   r.beta AS Beta,
                   r.sortino_ratio AS Sortino_Ratio
            FROM risk_metrics r
            JOIN portfolios p ON r.portfolio_id = p.portfolio_id
            LEFT JOIN portfolio_performance perf
              ON r.portfolio_id = perf.portfolio_id AND r.calculation_date = perf.performance_date
            WHERE r.calculation_date = (SELECT MAX(calculation_date) FROM risk_metrics)
            ORDER BY p.portfolio_id ASC;
            """,
        )
        st.dataframe(risk_perf_df, width="stretch", hide_index=True)

        st.subheader("Calculated Statistical Risk Metrics vs. Recorded Database Metrics (NumPy & SciPy)")
        st.caption(
            "**Calculated metrics** (`HHI`, `Effective N`, `Mean Daily Return %`, `Daily Volatility %`, "
            "`Skewness`, `Kurtosis`, and `Parametric 95% Daily VaR %`) are computed via NumPy and SciPy "
            "from `holdings` and `historical_prices`. **Recorded DB metrics** come directly from `portfolio_performance` "
            "(available for the 4 portfolios with recorded performance table history; `None` for the remaining 9)."
        )
        stat_summary_df = compute_all_portfolios_statistical_summary(agent)
        if not stat_summary_df.empty:
            st.dataframe(stat_summary_df, width="stretch", hide_index=True)

        st.subheader("Portfolio Return & NAV Visualizations (Matplotlib)")
        fig_mpl = create_risk_return_matplotlib_figure(agent)
        st.pyplot(fig_mpl)
        plt.close(fig_mpl)

        st.subheader("Sector Exposure & Return Correlation Heatmaps (Seaborn & SciPy)")
        fig_sns = create_sector_and_correlation_seaborn_figure(agent)
        st.pyplot(fig_sns)
        plt.close(fig_sns)

    # TAB 4: Portfolio Comparison
    with tab_compare:
        st.subheader("Side-by-Side Portfolio Comparison")
        selected_compare = st.multiselect(
            "Select Portfolios to Compare:",
            options=portfolio_names,
            default=[p for p in ["Growth Equity Fund", "Tech Innovation Fund", "International Equity Fund"] if p in portfolio_names],
        )
        if selected_compare:
            quoted = ", ".join(f"'{p.replace(chr(39), chr(39)*2)}'" for p in selected_compare)
            comp_df = query_df(
                agent,
                f"""
                SELECT p.portfolio_name AS Portfolio,
                       p.target_risk_level AS Risk_Level,
                       p.strategy_type AS Strategy,
                       p.status AS Status,
                       p.benchmark_index AS Benchmark,
                       p.total_aum AS Total_AUM,
                       COUNT(h.holding_id) AS Total_Holdings,
                       ROUND(COALESCE(SUM(h.quantity * s.current_price), 0.0), 2) AS Holdings_Market_Value,
                       ROUND(COALESCE(SUM(h.cost_basis), 0.0), 2) AS Total_Cost_Basis
                FROM portfolios p
                LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id
                LEFT JOIN securities s ON h.security_id = s.security_id
                WHERE p.portfolio_name IN ({quoted})
                GROUP BY p.portfolio_id, p.portfolio_name, p.target_risk_level,
                         p.strategy_type, p.status, p.benchmark_index, p.total_aum
                ORDER BY p.total_aum DESC;
                """,
            )
            st.dataframe(comp_df, width="stretch", hide_index=True)

            # Sector exposure comparison matrix
            sector_rows = []
            for p_name in selected_compare:
                exp = calculate_sector_exposure(agent.db_connection, p_name).get("sector_exposures", {})
                for sec_name, pct in exp.items():
                    sector_rows.append({"Portfolio": p_name, "Sector": sec_name, "Exposure (%)": pct})
            if sector_rows:
                st.subheader("Equity Sector Exposure Comparison (%)")
                pivot_df = pd.DataFrame(sector_rows).pivot(
                    index="Sector", columns="Portfolio", values="Exposure (%)"
                ).fillna(0.0)
                st.dataframe(pivot_df, width="stretch")
                st.bar_chart(pivot_df)

    # TAB 5: Conversational AI Agent with Explainable AI
    with tab_chat:
        st.subheader("Ask the Portfolio Analytics Agent")
        sample_queries = [
            "Custom Question...",
            "How many portfolios do we have in total?",
            "What are the names of all active portfolios?",
            "Which securities are in the Technology sector?",
            "What is the total Assets Under Management (AUM) for portfolios with high target risk level?",
            "Show me the top 5 holdings by cost basis in the Growth Equity Fund",
            "What is the average current price of securities in each sector?",
            "For each portfolio, show the total value of Technology sector holdings and what percentage it represents of the total portfolio value",
            "Find portfolios that have holdings in more than 5 different sectors and show their diversification metrics",
            "What are the sector exposures for the Tech Innovation Fund?",
            "Calculate the sector exposure breakdown for international equity",
            "Show the total AUM and sector exposures for Tech Innovation Fund",
            "Compare Growth Equity Fund and Tech Innovation Fund",
        ]
        chosen_sample = st.selectbox("Select a sample question or type your own below:", options=sample_queries)
        default_text = "" if chosen_sample == "Custom Question..." else chosen_sample
        user_question = st.text_input("Your Question:", value=default_text)

        if st.button("Run Query", type="primary"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            else:
                res = agent.answer_question(user_question)
                fallback_used = bool(res.get("fallback_used", False))
                exec_mode_raw = res.get("execution_mode", "unknown")

                if fallback_used:
                    exec_mode_label = f"Deterministic Fallback (`{exec_mode_raw}`)"
                elif exec_mode_raw == "gemini_live":
                    exec_mode_label = f"Gemini AI (`{exec_mode_raw}`)"
                else:
                    exec_mode_label = f"`{exec_mode_raw}`"

                if res.get("error"):
                    st.error(res.get("answer", res["error"]))
                else:
                    if fallback_used:
                        st.info(
                            "Answered via Deterministic Fallback — Gemini API was not used for this response; "
                            "verified local SQL/exposure routing was executed."
                        )
                    else:
                        st.success("Answered via Gemini AI — Live natural-language tool routing and query generation.")

                    formatted_answer = format_answer_text(user_question, res)
                    st.markdown(f"**Agent Response:**\n\n{formatted_answer}")

                with st.expander("How This Answer Was Generated", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.markdown(f"**Selected Tool:** `{res.get('tool_used')}`")
                    col_b.markdown(f"**Execution Mode:** {exec_mode_label}")
                    col_c.markdown(f"**Deterministic Fallback Used:** `{fallback_used}`")

                    if fallback_used and res.get("fallback_reason"):
                        st.warning(f"**Fallback Reason:** {res.get('fallback_reason')}")

                    st.markdown(f"**Reasoning / Explanation:** {res.get('explanation', 'N/A')}")

                    if res.get("sql_query"):
                        st.markdown("**Generated Read-Only SQL Query:**")
                        st.code(res["sql_query"], language="sql")

                    st.markdown("**Execution Details:**")
                    raw_res = res.get("raw_result")
                    if isinstance(raw_res, list):
                        result_summary = f"{len(raw_res)} row(s) returned"
                    elif isinstance(raw_res, dict):
                        result_summary = f"{len(raw_res)} key/sector entry(s) returned"
                    else:
                        result_summary = "No tabular result rows"

                    exec_details: Dict[str, Any] = {
                        "answer_source": (
                            "Deterministic Local Fallback (No Gemini generation)"
                            if fallback_used
                            else ("Gemini AI Live Generation" if exec_mode_raw == "gemini_live" else exec_mode_raw)
                        ),
                        "selected_tool": res.get("tool_used"),
                        "execution_mode": exec_mode_raw,
                        "deterministic_fallback_used": fallback_used,
                        "database_security_mode": "SQLite URI mode=ro + PRAGMA query_only=ON + Authorizer",
                        "result_summary": result_summary,
                    }
                    if fallback_used and res.get("fallback_reason"):
                        exec_details["fallback_reason"] = res.get("fallback_reason")
                    if res.get("tools_used"):
                        exec_details["tools_used"] = res.get("tools_used")
                    if res.get("portfolio_identified"):
                        exec_details["portfolio_identified"] = res.get("portfolio_identified")
                    if res.get("aum_result") is not None:
                        exec_details["aum_result"] = res.get("aum_result")
                    if res.get("sector_exposure_result") is not None:
                        exec_details["sector_exposure_result"] = res.get("sector_exposure_result")
                    if res.get("execution_status"):
                        exec_details["execution_status"] = res.get("execution_status")
                    if res.get("columns"):
                        exec_details["columns"] = res.get("columns")
                    if res.get("parameters"):
                        exec_details["parameters"] = res.get("parameters")

                    st.json(exec_details)


if __name__ == "__main__":
    main()
