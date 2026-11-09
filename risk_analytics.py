"""
Portfolio Risk & Statistical Analytics Module using NumPy, SciPy, Pandas, Matplotlib, and Seaborn.
All calculations use only actual rows from portfolio_database.db (holdings, historical_prices,
portfolio_performance, risk_metrics, portfolios, securities, and sectors) via read-only queries.
"""

from typing import Dict, Any, Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from agent import PortfolioAgent
from exposure_tool import calculate_sector_exposure, resolve_portfolio_name


def compute_portfolio_risk_analytics(agent: PortfolioAgent, portfolio_name: str) -> Dict[str, Any]:
    """
    Compute comprehensive portfolio analytics for a single portfolio using NumPy and SciPy,
    clearly separating:
      1. Recorded database values (from portfolios, portfolio_performance, risk_metrics)
      2. Calculated metrics (computed via NumPy and SciPy from holdings and historical_prices / NAVs)
    """
    if agent.db_connection is None:
        return {"error": "Database connection is not available."}

    resolution = resolve_portfolio_name(agent.db_connection, portfolio_name)
    if "error" in resolution:
        return resolution

    resolved_name = resolution["portfolio_name"]
    safe_name = resolved_name.replace("'", "''")

    # 1. Fetch portfolio metadata from database
    p_res = agent.execute_sql(
        f"""
        SELECT portfolio_id, portfolio_name, target_risk_level, total_aum,
               strategy_type, benchmark_index, status
        FROM portfolios
        WHERE portfolio_name = '{safe_name}';
        """
    )
    if "error" in p_res or not p_res.get("rows"):
        return {"error": f"Portfolio '{resolved_name}' metadata could not be retrieved."}

    p_row = p_res["rows"][0]
    portfolio_id = int(p_row[0])
    total_aum = float(p_row[3])

    # 2. Fetch holdings and compute NumPy weight/concentration statistics
    h_res = agent.execute_sql(
        f"""
        SELECT h.security_id, s.symbol, s.company_name, s.asset_type,
               h.quantity, h.current_weight, h.cost_basis,
               (h.quantity * s.current_price) AS market_value
        FROM holdings h
        JOIN securities s ON h.security_id = s.security_id
        WHERE h.portfolio_id = {portfolio_id}
        ORDER BY h.current_weight DESC;
        """
    )
    h_rows = h_res.get("rows", [])
    weights_arr = np.array([float(r[5]) for r in h_rows], dtype=np.float64) if h_rows else np.array([], dtype=np.float64)
    market_vals_arr = np.array([float(r[7]) for r in h_rows], dtype=np.float64) if h_rows else np.array([], dtype=np.float64)

    if weights_arr.size > 0:
        # Normalize weights for HHI calculation using NumPy
        weight_sum = float(np.sum(weights_arr))
        norm_weights = weights_arr / weight_sum if weight_sum > 0 else weights_arr
        hhi_index = float(np.sum(np.square(norm_weights)))
        effective_holdings = float(1.0 / hhi_index) if hhi_index > 0 else 0.0
        mean_weight_pct = float(np.mean(weights_arr) * 100.0)
        std_weight_pct = float(np.std(weights_arr, ddof=0) * 100.0)
        max_weight_pct = float(np.max(weights_arr) * 100.0)
        total_holdings_market_value = float(np.sum(market_vals_arr))
    else:
        hhi_index = 0.0
        effective_holdings = 0.0
        mean_weight_pct = 0.0
        std_weight_pct = 0.0
        max_weight_pct = 0.0
        total_holdings_market_value = 0.0

    # 3. Compute weighted daily return series from historical_prices using NumPy & SciPy
    hp_res = agent.execute_sql(
        f"""
        SELECT hp.security_id, hp.price_date, hp.close_price, h.current_weight
        FROM historical_prices hp
        JOIN holdings h ON hp.security_id = h.security_id
        WHERE h.portfolio_id = {portfolio_id}
        ORDER BY hp.price_date ASC, hp.security_id ASC;
        """
    )
    hp_rows = hp_res.get("rows", [])
    calculated_return_stats: Dict[str, Any] = {
        "data_source": "historical_prices + holdings (weighted daily close returns)",
        "observation_days": 0,
        "mean_daily_return_pct": None,
        "std_daily_return_pct": None,
        "min_daily_return_pct": None,
        "max_daily_return_pct": None,
        "median_daily_return_pct": None,
        "skewness_scipy": None,
        "kurtosis_scipy": None,
        "parametric_var_95_daily_pct_scipy": None,
        "shapiro_normality_p_value_scipy": None,
    }

    if hp_rows:
        hp_df = pd.DataFrame(hp_rows, columns=["security_id", "price_date", "close_price", "current_weight"])
        pivot_prices = hp_df.pivot(index="price_date", columns="security_id", values="close_price").sort_index()
        sec_weights_series = hp_df.groupby("security_id")["current_weight"].first()

        if len(pivot_prices) >= 2:
            sec_returns = pivot_prices.pct_change().dropna(how="all")
            aligned_weights = np.array(
                [float(sec_weights_series.get(col, 0.0)) for col in sec_returns.columns],
                dtype=np.float64,
            )
            w_total = float(np.sum(aligned_weights))
            if w_total > 0:
                aligned_weights = aligned_weights / w_total

            # NumPy matrix multiplication for daily portfolio percentage returns
            daily_port_returns_pct = np.dot(sec_returns.fillna(0.0).to_numpy(dtype=np.float64), aligned_weights) * 100.0
            n_obs = int(daily_port_returns_pct.size)
            if n_obs >= 2:
                mean_ret = float(np.mean(daily_port_returns_pct))
                std_ret = float(np.std(daily_port_returns_pct, ddof=1))
                min_ret = float(np.min(daily_port_returns_pct))
                max_ret = float(np.max(daily_port_returns_pct))
                median_ret = float(np.percentile(daily_port_returns_pct, 50))

                # SciPy statistical moments and parametric VaR
                skew_val = float(stats.skew(daily_port_returns_pct, bias=False)) if n_obs >= 3 else 0.0
                kurt_val = float(stats.kurtosis(daily_port_returns_pct, bias=False)) if n_obs >= 4 else 0.0
                param_var_95 = float(stats.norm.ppf(0.05, loc=mean_ret, scale=std_ret)) if std_ret > 0 else mean_ret
                shapiro_p = (
                    float(stats.shapiro(daily_port_returns_pct).pvalue)
                    if n_obs >= 3 and std_ret > 0
                    else None
                )

                calculated_return_stats = {
                    "data_source": "historical_prices + holdings (weighted daily close returns)",
                    "observation_days": n_obs,
                    "mean_daily_return_pct": round(mean_ret, 4),
                    "std_daily_return_pct": round(std_ret, 4),
                    "min_daily_return_pct": round(min_ret, 4),
                    "max_daily_return_pct": round(max_ret, 4),
                    "median_daily_return_pct": round(median_ret, 4),
                    "skewness_scipy": round(skew_val, 4),
                    "kurtosis_scipy": round(kurt_val, 4),
                    "parametric_var_95_daily_pct_scipy": round(param_var_95, 4),
                    "shapiro_normality_p_value_scipy": round(shapiro_p, 4) if shapiro_p is not None else None,
                }

    # 4. Fetch recorded database performance & risk_metrics rows (available for portfolios 1-4)
    perf_res = agent.execute_sql(
        f"""
        SELECT performance_date, nav, total_return_1m, total_return_3m,
               total_return_6m, total_return_1y, volatility, sharpe_ratio, max_drawdown
        FROM portfolio_performance
        WHERE portfolio_id = {portfolio_id}
        ORDER BY performance_date ASC;
        """
    )
    risk_res = agent.execute_sql(
        f"""
        SELECT calculation_date, var_95, var_99, cvar_95, beta,
               correlation_sp500, tracking_error, information_ratio, sortino_ratio
        FROM risk_metrics
        WHERE portfolio_id = {portfolio_id}
        ORDER BY calculation_date DESC
        LIMIT 1;
        """
    )

    perf_rows = perf_res.get("rows", [])
    risk_rows = risk_res.get("rows", [])

    recorded_db_metrics: Dict[str, Any] = {
        "has_recorded_performance_history": bool(perf_rows),
        "latest_performance_date": None,
        "latest_nav": None,
        "total_return_1m_pct": None,
        "total_return_3m_pct": None,
        "total_return_6m_pct": None,
        "total_return_1y_pct": None,
        "recorded_volatility_pct": None,
        "recorded_sharpe_ratio": None,
        "recorded_max_drawdown_pct": None,
        "recorded_var_95_pct": None,
        "recorded_var_99_pct": None,
        "recorded_cvar_95_pct": None,
        "recorded_beta": None,
        "recorded_correlation_sp500": None,
        "recorded_tracking_error": None,
        "recorded_information_ratio": None,
        "recorded_sortino_ratio": None,
    }

    if perf_rows:
        latest_p = perf_rows[-1]
        recorded_db_metrics.update(
            {
                "latest_performance_date": latest_p[0],
                "latest_nav": float(latest_p[1]),
                "total_return_1m_pct": float(latest_p[2]),
                "total_return_3m_pct": float(latest_p[3]),
                "total_return_6m_pct": float(latest_p[4]),
                "total_return_1y_pct": float(latest_p[5]),
                "recorded_volatility_pct": float(latest_p[6]),
                "recorded_sharpe_ratio": float(latest_p[7]),
                "recorded_max_drawdown_pct": float(latest_p[8]),
            }
        )

    if risk_rows:
        latest_r = risk_rows[0]
        recorded_db_metrics.update(
            {
                "recorded_var_95_pct": float(latest_r[1]),
                "recorded_var_99_pct": float(latest_r[2]),
                "recorded_cvar_95_pct": float(latest_r[3]),
                "recorded_beta": float(latest_r[4]),
                "recorded_correlation_sp500": float(latest_r[5]),
                "recorded_tracking_error": float(latest_r[6]),
                "recorded_information_ratio": float(latest_r[7]),
                "recorded_sortino_ratio": float(latest_r[8]),
            }
        )

    # 5. Reuse existing calculate_sector_exposure
    exp_res = calculate_sector_exposure(agent.db_connection, resolved_name)
    sector_exposures = exp_res.get("sector_exposures", {})

    return {
        "portfolio_id": portfolio_id,
        "portfolio_name": resolved_name,
        "target_risk_level": p_row[2],
        "total_aum": total_aum,
        "strategy_type": p_row[4],
        "benchmark_index": p_row[5],
        "status": p_row[6],
        "sector_exposures": sector_exposures,
        "calculated_concentration_metrics_numpy": {
            "holdings_count": len(h_rows),
            "total_holdings_market_value": round(total_holdings_market_value, 2),
            "hhi_concentration_index": round(hhi_index, 4),
            "effective_number_of_holdings": round(effective_holdings, 2),
            "mean_holding_weight_pct": round(mean_weight_pct, 2),
            "std_holding_weight_pct": round(std_weight_pct, 2),
            "max_single_holding_weight_pct": round(max_weight_pct, 2),
        },
        "calculated_return_statistics_numpy_scipy": calculated_return_stats,
        "recorded_database_metrics": recorded_db_metrics,
    }


def compute_all_portfolios_statistical_summary(agent: PortfolioAgent) -> pd.DataFrame:
    """
    Build a summary DataFrame across all 13 portfolios combining NumPy concentration metrics
    and SciPy daily return statistics from historical_prices + holdings.
    """
    p_res = agent.execute_sql("SELECT portfolio_name FROM portfolios ORDER BY portfolio_id ASC;")
    names = [r[0] for r in p_res.get("rows", [])]
    rows: List[Dict[str, Any]] = []

    for name in names:
        data = compute_portfolio_risk_analytics(agent, name)
        if "error" in data:
            continue
        conc = data["calculated_concentration_metrics_numpy"]
        ret = data["calculated_return_statistics_numpy_scipy"]
        rec = data["recorded_database_metrics"]
        rows.append(
            {
                "Portfolio": data["portfolio_name"],
                "Risk Level": data["target_risk_level"],
                "Total AUM ($)": data["total_aum"],
                "Holdings": conc["holdings_count"],
                "HHI (NumPy)": conc["hhi_concentration_index"],
                "Effective N (NumPy)": conc["effective_number_of_holdings"],
                "Mean Daily Return % (NumPy)": ret["mean_daily_return_pct"],
                "Daily Volatility % (NumPy)": ret["std_daily_return_pct"],
                "Return Skewness (SciPy)": ret["skewness_scipy"],
                "Return Kurtosis (SciPy)": ret["kurtosis_scipy"],
                "Parametric 95% Daily VaR % (SciPy)": ret["parametric_var_95_daily_pct_scipy"],
                "Recorded 1Y Return % (DB)": rec["total_return_1y_pct"],
                "Recorded Sharpe (DB)": rec["recorded_sharpe_ratio"],
            }
        )
    return pd.DataFrame(rows)


def create_risk_return_matplotlib_figure(agent: PortfolioAgent) -> plt.Figure:
    """
    Create a two-panel Matplotlib figure:
      Panel 1: Calculated Daily Return vs. Daily Volatility (NumPy/SciPy) across all 13 portfolios
      Panel 2: Recorded NAV Time-Series Trajectory for portfolios with historical performance records in DB
    """
    summary_df = compute_all_portfolios_statistical_summary(agent)
    perf_res = agent.execute_sql(
        """
        SELECT p.portfolio_name, perf.performance_date, perf.nav
        FROM portfolio_performance perf
        JOIN portfolios p ON perf.portfolio_id = p.portfolio_id
        ORDER BY perf.performance_date ASC;
        """
    )
    perf_df = pd.DataFrame(
        perf_res.get("rows", []),
        columns=["Portfolio", "Date", "NAV"],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Risk-Return scatter plot (Calculated via NumPy from historical_prices)
    if not summary_df.empty:
        valid_df = summary_df.dropna(subset=["Daily Volatility % (NumPy)", "Mean Daily Return % (NumPy)"])
        risk_colors = {"High": "#d9534f", "Medium": "#f0ad4e", "Low": "#5cb85c"}
        for risk_level, group in valid_df.groupby("Risk Level"):
            ax1.scatter(
                group["Daily Volatility % (NumPy)"],
                group["Mean Daily Return % (NumPy)"],
                label=f"{risk_level} Risk",
                color=risk_colors.get(str(risk_level), "#0275d8"),
                s=80,
                alpha=0.85,
                edgecolors="black",
            )
        ax1.set_title("Calculated Daily Mean Return vs. Volatility (NumPy)")
        ax1.set_xlabel("Calculated Daily Return Std Dev (%)")
        ax1.set_ylabel("Calculated Mean Daily Return (%)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="best", title="Target Risk")

    # Panel 2: Recorded NAV trajectories from portfolio_performance table
    if not perf_df.empty:
        for pname, group in perf_df.groupby("Portfolio"):
            ax2.plot(group["Date"], group["NAV"], marker="o", linewidth=1.8, label=str(pname))
        ax2.set_title("Recorded NAV Trajectories (DB portfolio_performance)")
        ax2.set_xlabel("Performance Date")
        ax2.set_ylabel("Net Asset Value (NAV)")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="best", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No recorded NAV history available", ha="center", va="center")

    fig.tight_layout()
    return fig


def create_sector_and_correlation_seaborn_figure(agent: PortfolioAgent) -> plt.Figure:
    """
    Create a two-panel Seaborn figure:
      Panel 1: Seaborn Heatmap of Equity Sector Exposures (%) across portfolios
      Panel 2: Seaborn Heatmap of SciPy Pairwise Pearson Correlations of Recorded Portfolio NAVs
    """
    p_res = agent.execute_sql("SELECT portfolio_name FROM portfolios ORDER BY portfolio_id ASC;")
    portfolio_names = [r[0] for r in p_res.get("rows", [])]

    sector_records: List[Dict[str, Any]] = []
    for pname in portfolio_names:
        exp = calculate_sector_exposure(agent.db_connection, pname).get("sector_exposures", {})
        for sec, pct in exp.items():
            sector_records.append({"Portfolio": pname, "Sector": sec, "Exposure": float(pct)})

    perf_res = agent.execute_sql(
        """
        SELECT p.portfolio_name, perf.performance_date, perf.nav
        FROM portfolio_performance perf
        JOIN portfolios p ON perf.portfolio_id = p.portfolio_id
        ORDER BY perf.performance_date ASC;
        """
    )
    perf_df = pd.DataFrame(perf_res.get("rows", []), columns=["Portfolio", "Date", "NAV"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Sector Exposure Heatmap via Seaborn
    if sector_records:
        sec_pivot = (
            pd.DataFrame(sector_records)
            .pivot(index="Portfolio", columns="Sector", values="Exposure")
            .fillna(0.0)
        )
        sns.heatmap(
            sec_pivot,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            cbar_kws={"label": "Equity Exposure (%)"},
            ax=ax1,
        )
        ax1.set_title("Equity Sector Exposure Heatmap (%) — Seaborn")
        ax1.set_xlabel("Sector")
        ax1.set_ylabel("Portfolio")
    else:
        ax1.text(0.5, 0.5, "No sector exposure data", ha="center", va="center")

    # Panel 2: Pairwise SciPy Pearson Correlation Matrix of Recorded Portfolio NAV Returns via Seaborn
    if not perf_df.empty:
        nav_pivot = perf_df.pivot(index="Date", columns="Portfolio", values="NAV").sort_index()
        nav_returns = nav_pivot.pct_change().dropna(how="all")
        cols = list(nav_returns.columns)
        n_cols = len(cols)
        corr_matrix = np.eye(n_cols, dtype=np.float64)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                r_val, _ = stats.pearsonr(
                    nav_returns[cols[i]].to_numpy(dtype=np.float64),
                    nav_returns[cols[j]].to_numpy(dtype=np.float64),
                )
                corr_matrix[i, j] = float(r_val)
                corr_matrix[j, i] = float(r_val)
        corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            cbar_kws={"label": "Pearson Correlation (r)"},
            ax=ax2,
        )
        ax2.set_title("Recorded NAV Return Correlation (SciPy & Seaborn)")
    else:
        ax2.text(0.5, 0.5, "Insufficient NAV series for correlation", ha="center", va="center")

    fig.tight_layout()
    return fig
