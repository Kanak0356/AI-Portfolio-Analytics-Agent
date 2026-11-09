"""Pytest suite for the Portfolio Analytics Agent (executes locally with 0 live Gemini API calls)."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from agent import PortfolioAgent
from exposure_tool import calculate_sector_exposure
from evaluator import run_evaluation, INDEPENDENT_EXPECTED_EXPOSURES
from app import compute_concentration_table

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "portfolio_database.db")


@pytest.fixture
def agent():
    ag = PortfolioAgent(db_path=DB_PATH, api_key="mock_key_no_network")
    yield ag
    ag.close()


def test_ground_truth_evaluation_suite(tmp_path):
    """Verify all 10 ground-truth questions pass against independent expected values."""
    report_file = tmp_path / "test_eval_report.json"
    report = run_evaluation(db_path=DB_PATH, report_path=str(report_file), mock_http_429=True)
    assert report["summary"]["total_questions"] == 10
    assert report["summary"]["passed_questions"] == 10
    assert report["summary"]["failed_questions"] == 0
    assert report["summary"]["overall_accuracy_percent"] == 100.0
    assert report["summary"]["live_gemini_tested"] is False


def test_independent_exposure_values(agent):
    """Verify exposure calculator matches independent fixed expectations for Q9 and Q10."""
    res_q9 = calculate_sector_exposure(agent.db_connection, "Tech Innovation Fund")
    assert res_q9["sector_exposures"] == INDEPENDENT_EXPECTED_EXPOSURES["Tech Innovation Fund"]

    res_q10 = calculate_sector_exposure(agent.db_connection, "international equity")
    assert res_q10["portfolio_name"] == "International Equity Fund"
    assert res_q10["sector_exposures"] == INDEPENDENT_EXPECTED_EXPOSURES["International Equity Fund"]


def test_mocked_successful_gemini_sql_and_explainability(agent):
    """Verify Gemini planning path sets fallback_used=False and includes explainable AI fields."""
    mock_plan = json.dumps(
        {
            "tool": "text2sql",
            "sql_query": "SELECT COUNT(*) FROM portfolios;",
            "portfolio_name": None,
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_plan, None)):
        res = agent.answer_question("How many portfolios exist?")
        assert res["tool_used"] == "text2sql"
        assert res["fallback_used"] is False
        assert res["execution_mode"] == "gemini_live"
        assert res["raw_result"] == [(13,)]
        assert "explanation" in res


def test_hybrid_execution_both_tools(agent):
    """Verify hybrid questions execute both SQL and exposure calculator across all phrasings and edge cases."""
    with patch.object(
        agent,
        "_call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        # Example 1: Growth Equity Fund
        r1 = agent.answer_question("Show me the AUM and sector exposure of Growth Equity Fund.")
        assert r1["tool_used"] == "hybrid"
        assert r1["tools_used"] == ["SQL Query Tool", "Exposure Calculator"]
        assert r1["portfolio_identified"] == "Growth Equity Fund"
        assert r1["aum_result"] == 25000000.0
        assert r1["sector_exposure_result"] == {
            "Technology": 93.0,
            "Consumer Discretionary": 7.0,
        }
        assert r1["execution_status"] == "success"
        assert r1["fallback_used"] is True
        assert "Portfolio: Growth Equity Fund" in r1["answer"]
        assert "Total AUM: $25,000,000.00" in r1["answer"]
        assert "Sector Exposure:" in r1["answer"]
        assert "- Technology: 93.0%" in r1["answer"]
        assert "- Consumer Discretionary: 7.0%" in r1["answer"]
        assert "Explanation:" in r1["answer"]

        # Example 2: International Equity Fund
        r2 = agent.answer_question("Compare the AUM and sector allocation of International Equity Fund.")
        assert r2["tool_used"] == "hybrid"
        assert r2["portfolio_identified"] == "International Equity Fund"
        assert r2["aum_result"] == 28000000.0
        assert r2["sector_exposure_result"] == {
            "Energy": 38.0,
            "Technology": 36.0,
            "Automotive": 15.0,
            "Consumer Staples": 11.0,
        }
        assert r2["execution_status"] == "success"

        # Example 3: Tech Innovation Fund
        r3 = agent.answer_question("Give me the total assets and sector breakdown for Tech Innovation Fund.")
        assert r3["tool_used"] == "hybrid"
        assert r3["portfolio_identified"] == "Tech Innovation Fund"
        assert r3["aum_result"] == 32000000.0
        assert r3["sector_exposure_result"] == {"Technology": 100.0}
        assert r3["raw_result"]["sql_rows"][0][0] == "Tech Innovation Fund"

        # Additional phrasings from Part A.4
        r4 = agent.answer_question("Show the portfolio value and industry breakdown for Growth Equity Fund")
        assert r4["tool_used"] == "hybrid"
        assert r4["aum_result"] == 25000000.0

        r5 = agent.answer_question("What are the assets under management and sector distribution of Tech Innovation Fund?")
        assert r5["tool_used"] == "hybrid"
        assert r5["aum_result"] == 32000000.0

        # Ambiguous portfolio in hybrid mode asks user to clarify before executing tools
        r_ambig = agent.answer_question("Show me the AUM and sector exposure of Equity")
        assert r_ambig["tool_used"] == "hybrid"
        assert r_ambig["execution_status"] == "clarification_required"
        assert r_ambig["matching_portfolios"] == ["Growth Equity Fund", "International Equity Fund"]
        assert "Multiple portfolios match 'Equity'" in r_ambig["answer"]

        # Nonexistent portfolio in hybrid mode returns clear error
        r_missing = agent.answer_question("Show me the AUM and sector exposure of Nonexistent Alpha Fund")
        assert r_missing["tool_used"] == "hybrid"
        assert r_missing["execution_status"] == "error"
        assert "not found" in r_missing["answer"].lower()


def test_cli_interface(capsys):
    """Verify the CLI supports both single-question execution and interactive REPL loop without live API calls."""
    from agent import run_cli

    with patch(
        "agent.PortfolioAgent._call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        # 1. Single-question CLI mode with --details
        exit_code = run_cli(["Show me the AUM and sector exposure of Growth Equity Fund.", "--details"])
        assert exit_code == 0
        out1 = capsys.readouterr().out
        assert "Portfolio: Growth Equity Fund" in out1
        assert "Total AUM: $25,000,000.00" in out1
        assert "SQL Query Tool, Exposure Calculator" in out1

        # 2. Interactive REPL loop mode (help, details toggle, query, exit)
        inputs = iter(["", "help", "details", "How many portfolios do we have in total?", "exit"])
        with patch("builtins.input", side_effect=lambda _: next(inputs)):
            exit_code_repl = run_cli([])
            assert exit_code_repl == 0
            out2 = capsys.readouterr().out
            assert "There are 13 portfolios in total." in out2
            assert "Thank you for using Portfolio Analytics Agent" in out2


def test_portfolio_comparison_question(agent):
    """Verify conversational portfolio comparison works accurately."""
    with patch.object(
        agent,
        "_call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        res = agent.answer_question("Compare Growth Equity Fund and Tech Innovation Fund")
        assert res["tool_used"] == "text2sql"
        assert len(res["raw_result"]) == 2


def test_sql_security_enforcement(agent):
    """Verify write, DDL, PRAGMA, and multi-statement queries are blocked."""
    for unsafe_sql in [
        "DELETE FROM portfolios WHERE portfolio_id = 1;",
        "UPDATE portfolios SET total_aum = 0;",
        "INSERT INTO sectors VALUES (99, 'Test', 'Desc', 'Group');",
        "DROP TABLE holdings;",
        "CREATE TABLE test_hack (id INT);",
        "ATTACH DATABASE ':memory:' AS aux;",
        "PRAGMA query_only = OFF;",
        "SELECT 1; SELECT 2;",
    ]:
        res = agent.execute_sql(unsafe_sql)
        assert "error" in res
        assert "Security error" in res["error"]


def test_dashboard_concentration_analysis(agent):
    """Verify risk & concentration table computes across all 13 portfolios without error."""
    df = compute_concentration_table(agent)
    assert len(df) == 13
    assert "Concentration Alert" in df.columns
    assert "Diversification Ratio" in df.columns


def test_streamlit_explainable_ai_expander_and_wording():
    """Verify Streamlit Explainable AI expander is named 'How This Answer Was Generated', collapsed by default, and distinguishes fallback vs Gemini."""
    from streamlit.testing.v1 import AppTest

    # 1. Test Deterministic Fallback UI state (0 live API calls)
    with patch(
        "agent.PortfolioAgent._call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at.run()
        at.text_input[0].set_value("How many portfolios do we have in total?").run()
        at.button[0].click().run()

        # Verify no success banner claiming Gemini succeeded when fallback was used
        assert len(at.success) == 0
        assert any("Deterministic Fallback" in info.value for info in at.info)

        # Verify improved answer wording
        assert any("There are 13 portfolios in total." in md.value for md in at.markdown)

        # Verify expander title and collapsed-by-default state
        expanders = [exp for exp in at.expander if exp.label == "How This Answer Was Generated"]
        assert len(expanders) == 1
        exp = expanders[0]
        assert exp.proto.expanded is False

        # Verify contents inside the expander
        exp_markdowns = "\n".join(m.value for m in exp.markdown)
        assert "**Selected Tool:** `text2sql`" in exp_markdowns
        assert "**Execution Mode:** Deterministic Fallback (`deterministic_fallback`)" in exp_markdowns
        assert "**Deterministic Fallback Used:** `True`" in exp_markdowns
        assert "**Reasoning / Explanation:**" in exp_markdowns
        assert "**Generated Read-Only SQL Query:**" in exp_markdowns
        assert "**Execution Details:**" in exp_markdowns
        assert any("Fallback Reason:" in w.value for w in exp.warning)
        assert len(exp.code) == 1
        assert "SELECT COUNT(*) FROM portfolios;" in exp.code[0].value
        assert len(exp.json) == 1

    # 2. Test Mocked Gemini Live UI state (0 live API calls)
    mock_plan = json.dumps(
        {
            "tool": "text2sql",
            "sql_query": "SELECT COUNT(*) FROM portfolios;",
            "portfolio_name": None,
            "reason": None,
        }
    )
    with patch("agent.PortfolioAgent._call_gemini", return_value=(mock_plan, None)):
        at_gemini = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
        at_gemini.run()
        at_gemini.text_input[0].set_value("How many portfolios do we have in total?").run()
        at_gemini.button[0].click().run()

        assert len(at_gemini.success) == 1
        assert "Gemini AI" in at_gemini.success[0].value
        exp_g = [exp for exp in at_gemini.expander if exp.label == "How This Answer Was Generated"][0]
        assert exp_g.proto.expanded is False
        exp_g_md = "\n".join(m.value for m in exp_g.markdown)
        assert "**Execution Mode:** Gemini AI (`gemini_live`)" in exp_g_md
        assert "**Deterministic Fallback Used:** `False`" in exp_g_md


def test_alternative_phrasings_and_unsupported_questions(agent):
    """Verify alternative natural-language phrasings, portfolio matching, and unsupported questions."""
    with patch.object(
        agent,
        "_call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        # Alternative phrasing for Q1
        r1 = agent.answer_question("What is the total number of portfolios?")
        assert r1["raw_result"] == [(13,)]
        assert "There are 13 portfolios in total." in r1["answer"]

        # Alternative phrasing for Q3
        r3 = agent.answer_question("Which stocks are in the Technology sector?")
        assert len(r3["raw_result"]) == 8

        # Alternative phrasing for Q6
        r6 = agent.answer_question("What is the average price of securities by sector?")
        assert len(r6["raw_result"]) == 7

        # Portfolio holdings lookup (Tech Innovation Fund has 8 holdings)
        rh = agent.answer_question("Show the holdings in Tech Innovation Fund")
        assert rh["tool_used"] == "text2sql"
        assert len(rh["raw_result"]) == 8

        # Portfolio details lookup
        rp = agent.answer_question("Tell me about Growth Equity Fund")
        assert rp["tool_used"] == "text2sql"
        assert rp["raw_result"][0][0] == "Growth Equity Fund"

        # Unsupported out-of-domain question
        ru = agent.answer_question("What is the weather in Tokyo tomorrow?")
        assert ru["tool_used"] is None
        assert "error" in ru


def test_database_integrity_and_unmodified_source_files(agent):
    """Verify database integrity, row counts (603 rows across 9 tables), and SHA-256 match against original source files."""
    import hashlib
    import sqlite3

    # Verify that agent.db_connection authorizer blocks direct PRAGMA execution at the C-engine level
    with pytest.raises(sqlite3.DatabaseError):
        agent.db_connection.cursor().execute("PRAGMA integrity_check;")

    # Open a read-only inspection connection without the PRAGMA-blocking authorizer to run PRAGMA integrity checks
    ro_uri = f"{Path(DB_PATH).resolve().as_uri()}?mode=ro"
    check_conn = sqlite3.connect(ro_uri, uri=True)
    cursor = check_conn.cursor()
    cursor.execute("PRAGMA integrity_check;")
    assert cursor.fetchone()[0] == "ok"

    cursor.execute("PRAGMA foreign_key_check;")
    assert cursor.fetchall() == []

    expected_counts = {
        "sectors": 10,
        "securities": 30,
        "benchmarks": 13,
        "portfolios": 13,
        "holdings": 95,
        "transactions": 24,
        "historical_prices": 330,
        "portfolio_performance": 44,
        "risk_metrics": 44,
    }
    total_rows = 0
    for table, exp_count in expected_counts.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        cnt = cursor.fetchone()[0]
        assert cnt == exp_count
        total_rows += cnt
    assert total_rows == 603
    check_conn.close()

    orig_root = BASE_DIR / "Exercise_PL_AI_Engineer" / "Exercise_PL_AI_Engineer"
    if orig_root.exists():
        for rel_path in [
            "database_schema.sql",
            "ground_truth_dataset.json",
            "requirements.txt",
            "data/sectors.csv",
            "data/securities.csv",
            "data/benchmarks.csv",
            "data/portfolios.csv",
            "data/holdings.csv",
            "data/transactions.csv",
            "data/historical_prices.csv",
            "data/portfolio_performance.csv",
            "data/risk_metrics.csv",
        ]:
            h1 = hashlib.sha256((BASE_DIR / rel_path).read_bytes()).hexdigest()
            h2 = hashlib.sha256((orig_root / rel_path).read_bytes()).hexdigest()
            assert h1 == h2, f"Mismatch in {rel_path}"


def test_numpy_scipy_matplotlib_seaborn_risk_analytics(agent):
    """Verify NumPy, SciPy, Matplotlib, and Seaborn analytics functions execute accurately on real DB data."""
    import matplotlib.pyplot as plt
    from risk_analytics import (
        compute_portfolio_risk_analytics,
        compute_all_portfolios_statistical_summary,
        create_risk_return_matplotlib_figure,
        create_sector_and_correlation_seaborn_figure,
    )

    # 1. Portfolio WITH recorded performance & risk_metrics history (Growth Equity Fund)
    res_growth = compute_portfolio_risk_analytics(agent, "Growth Equity Fund")
    assert res_growth["portfolio_name"] == "Growth Equity Fund"
    assert res_growth["total_aum"] == 25000000.0
    assert res_growth["sector_exposures"] == {"Technology": 93.0, "Consumer Discretionary": 7.0}

    conc = res_growth["calculated_concentration_metrics_numpy"]
    assert conc["holdings_count"] == 8
    assert conc["hhi_concentration_index"] > 0.0
    assert conc["effective_number_of_holdings"] > 1.0

    ret = res_growth["calculated_return_statistics_numpy_scipy"]
    assert ret["observation_days"] == 10
    assert ret["mean_daily_return_pct"] is not None
    assert ret["std_daily_return_pct"] is not None
    assert ret["skewness_scipy"] is not None
    assert ret["kurtosis_scipy"] is not None
    assert ret["parametric_var_95_daily_pct_scipy"] is not None

    rec = res_growth["recorded_database_metrics"]
    assert rec["has_recorded_performance_history"] is True
    assert rec["latest_nav"] == 127.89
    assert rec["recorded_sharpe_ratio"] == 1.40
    assert rec["recorded_var_95_pct"] == -3.0

    # 2. Portfolio WITHOUT recorded performance_history rows (International Equity Fund) handled gracefully
    res_intl = compute_portfolio_risk_analytics(agent, "International Equity Fund")
    assert res_intl["portfolio_name"] == "International Equity Fund"
    assert res_intl["recorded_database_metrics"]["has_recorded_performance_history"] is False
    assert res_intl["recorded_database_metrics"]["latest_nav"] is None
    assert res_intl["calculated_return_statistics_numpy_scipy"]["observation_days"] == 10

    # 3. Summary table across all 13 portfolios
    summary_df = compute_all_portfolios_statistical_summary(agent)
    assert len(summary_df) == 13
    assert "HHI (NumPy)" in summary_df.columns
    assert "Return Skewness (SciPy)" in summary_df.columns

    # 4. Matplotlib and Seaborn figure generation
    fig1 = create_risk_return_matplotlib_figure(agent)
    assert len(fig1.axes) == 2
    plt.close(fig1)

    fig2 = create_sector_and_correlation_seaborn_figure(agent)
    assert len(fig2.axes) >= 2
    plt.close(fig2)


def test_fastapi_rest_endpoints_with_httpx():
    """Verify FastAPI REST endpoints (/health, /portfolios, details, exposure, analytics, /agent/query) via HTTPX."""
    from fastapi.testclient import TestClient
    import httpx
    from api import app

    assert hasattr(httpx, "Client")
    with TestClient(app) as client:
        # 1. GET /health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "ok"
        assert health_data["database_connected"] is True
        assert health_data["total_portfolios"] == 13

        # 2. GET /portfolios
        r_list = client.get("/portfolios")
        assert r_list.status_code == 200
        list_data = r_list.json()
        assert list_data["count"] == 13
        assert len(list_data["portfolios"]) == 13

        # 3. GET /portfolios/{portfolio_name}
        r_detail = client.get("/portfolios/Growth Equity Fund")
        assert r_detail.status_code == 200
        detail_data = r_detail.json()
        assert detail_data["portfolio_name"] == "Growth Equity Fund"
        assert detail_data["total_aum"] == 25000000.0
        assert detail_data["holdings_count"] == 8

        # 4. GET /portfolios/{portfolio_name}/exposure (using partial name 'international equity')
        r_exp = client.get("/portfolios/international equity/exposure")
        assert r_exp.status_code == 200
        exp_data = r_exp.json()
        assert exp_data["portfolio_name"] == "International Equity Fund"
        assert exp_data["sector_exposures"] == {
            "Energy": 38.0,
            "Technology": 36.0,
            "Automotive": 15.0,
            "Consumer Staples": 11.0,
        }

        # 5. GET /portfolios/{portfolio_name}/analytics
        r_ana = client.get("/portfolios/Tech Innovation Fund/analytics")
        assert r_ana.status_code == 200
        ana_data = r_ana.json()
        assert ana_data["portfolio_name"] == "Tech Innovation Fund"
        assert ana_data["total_aum"] == 32000000.0
        assert ana_data["sector_exposures"] == {"Technology": 100.0}
        assert ana_data["calculated_concentration_metrics_numpy"]["holdings_count"] == 8
        assert ana_data["calculated_return_statistics_numpy_scipy"]["observation_days"] == 10
        assert ana_data["recorded_database_metrics"]["has_recorded_performance_history"] is True

        # 6. POST /agent/query (with mocked Gemini to avoid live API calls)
        with patch(
            "agent.PortfolioAgent._call_gemini",
            return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
        ):
            r_query = client.post(
                "/agent/query",
                json={"question": "Show me the AUM and sector exposure of Growth Equity Fund."},
            )
            assert r_query.status_code == 200
            q_data = r_query.json()
            assert q_data["tool_used"] == "hybrid"
            assert q_data["portfolio_identified"] == "Growth Equity Fund"
            assert q_data["aum_result"] == 25000000.0
            assert q_data["sector_exposure_result"] == {"Technology": 93.0, "Consumer Discretionary": 7.0}


def test_fastapi_validation_and_error_handling_with_httpx():
    """Verify HTTP 404, 400 (ambiguous portfolio), 422 (malformed request), and 503 error handling via HTTPX."""
    from fastapi.testclient import TestClient
    import api

    with TestClient(api.app) as client:
        # 1. Invalid / nonexistent portfolio -> 404 Not Found
        r_404 = client.get("/portfolios/Nonexistent Alpha Fund")
        assert r_404.status_code == 404

        r_404_exp = client.get("/portfolios/Nonexistent Alpha Fund/exposure")
        assert r_404_exp.status_code == 404

        # 2. Ambiguous portfolio name ('Equity') -> 400 Bad Request with matching_portfolios
        r_400 = client.get("/portfolios/Equity/exposure")
        assert r_400.status_code == 400
        assert "matching_portfolios" in r_400.json()["detail"]

        # 3. Missing or whitespace-only question -> 422 Unprocessable Entity (Pydantic validation)
        r_empty = client.post("/agent/query", json={"question": "   "})
        assert r_empty.status_code == 422

        r_missing_field = client.post("/agent/query", json={})
        assert r_missing_field.status_code == 422

        # 4. API 503 error handling when database connection is unavailable
        ag = api.get_api_agent()
        with patch.object(ag, "db_connection", None), patch("api.get_api_agent", return_value=ag):
            r_503 = client.get("/portfolios/Growth Equity Fund")
            assert r_503.status_code == 503
            r_q_503 = client.post("/agent/query", json={"question": "How many portfolios?"})
            assert r_q_503.status_code == 503
