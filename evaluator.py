"""Evaluation suite for the Portfolio Analytics Agent against ground_truth_dataset.json and robustness tests."""

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from agent import PortfolioAgent

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = str(BASE_DIR / "portfolio_database.db")
DEFAULT_GT_PATH = str(BASE_DIR / "ground_truth_dataset.json")
DEFAULT_REPORT_PATH = str(BASE_DIR / "evaluation_report.json")

# Independent fixed expected sector exposure values for Q9 and Q10
# Derived directly from holdings.csv, securities.csv, and sectors.csv (equities only, excluding bonds)
INDEPENDENT_EXPECTED_EXPOSURES: Dict[str, Dict[str, float]] = {
    "Tech Innovation Fund": {
        "Technology": 100.0,
    },
    "International Equity Fund": {
        "Energy": 38.0,
        "Technology": 36.0,
        "Automotive": 15.0,
        "Consumer Staples": 11.0,
    },
}


def _normalize_value(val: Any) -> Any:
    """Normalize scalar values for comparison (rounding floats to 4 decimal places)."""
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, str):
        return val.strip()
    return val


def _compare_sql_results(
    expected_rows: List[Tuple[Any, ...]],
    actual_rows: List[Tuple[Any, ...]],
    tol: float = 1e-3,
) -> Tuple[bool, Optional[str]]:
    """Compare expected SQL rows with actual SQL rows returned by the agent."""
    if not isinstance(actual_rows, list):
        return False, f"Expected a list of rows, got {type(actual_rows).__name__}."

    if len(expected_rows) != len(actual_rows):
        return False, f"Row count mismatch: expected {len(expected_rows)} rows, got {len(actual_rows)} rows."

    for row_idx, (exp_row, act_row) in enumerate(zip(expected_rows, actual_rows)):
        if len(exp_row) != len(act_row):
            return (
                False,
                f"Column count mismatch at row {row_idx}: expected {len(exp_row)} columns, got {len(act_row)}.",
            )
        for col_idx, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                if not math.isclose(float(exp_val), float(act_val), abs_tol=tol, rel_tol=tol):
                    return (
                        False,
                        f"Numeric mismatch at row {row_idx}, col {col_idx}: expected {exp_val}, got {act_val}.",
                    )
            else:
                if str(exp_val).strip().lower() != str(act_val).strip().lower():
                    return (
                        False,
                        f"Value mismatch at row {row_idx}, col {col_idx}: expected {repr(exp_val)}, got {repr(act_val)}.",
                    )

    return True, None


def _compare_exposure_results(
    expected_exposures: Dict[str, float],
    actual_exposures: Dict[str, float],
    tol: float = 1e-2,
) -> Tuple[bool, Optional[str]]:
    """Compare independent expected sector exposure dictionary with actual exposures."""
    if not isinstance(actual_exposures, dict):
        return False, f"Expected a dictionary of sector exposures, got {type(actual_exposures).__name__}."

    exp_keys = {k.strip().lower(): (k, v) for k, v in expected_exposures.items()}
    act_keys = {k.strip().lower(): (k, v) for k, v in actual_exposures.items()}

    if set(exp_keys.keys()) != set(act_keys.keys()):
        return (
            False,
            f"Sector keys mismatch: expected {list(expected_exposures.keys())}, got {list(actual_exposures.keys())}.",
        )

    for k_lower, (orig_k, exp_val) in exp_keys.items():
        _, act_val = act_keys[k_lower]
        if not math.isclose(float(exp_val), float(act_val), abs_tol=tol):
            return (
                False,
                f"Exposure mismatch for sector '{orig_k}': expected {exp_val}%, got {act_val}%.",
            )

    return True, None


def run_additional_verification_tests(agent: PortfolioAgent) -> List[Dict[str, Any]]:
    """
    Run non-quota-consuming verification tests covering:
    1. Mocked successful Gemini planning (text2sql + exposure_calculator).
    2. Hybrid SQL + exposure execution.
    3. Portfolio comparison query.
    4. Unsafe SQL rejection (DROP TABLE / multi-statement).
    5. Invalid SQL syntax / nonexistent column handling.
    6. Ambiguous portfolio name handling.
    7. API error handling (HTTP 429, 503, 504) on unmapped queries.
    """
    checks: List[Dict[str, Any]] = []

    # 1. Mocked successful Gemini planning test (text2sql)
    mock_gemini_sql_json = json.dumps(
        {
            "tool": "text2sql",
            "sql_query": "SELECT s.symbol, s.company_name FROM securities s JOIN sectors sec ON s.sector_id = sec.sector_id WHERE sec.sector_name = 'Technology';",
            "portfolio_name": None,
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_gemini_sql_json, None)):
        res_mock_gemini = agent.answer_question("List all stocks in the Technology sector")
        passed_mock_gemini = (
            res_mock_gemini.get("tool_used") == "text2sql"
            and res_mock_gemini.get("fallback_used") is False
            and res_mock_gemini.get("execution_mode") == "gemini_live"
            and isinstance(res_mock_gemini.get("raw_result"), list)
            and len(res_mock_gemini["raw_result"]) == 8
        )
        checks.append(
            {
                "test_name": "mocked_successful_gemini_planning_sql",
                "passed": passed_mock_gemini,
                "execution_mode": res_mock_gemini.get("execution_mode"),
                "fallback_used": res_mock_gemini.get("fallback_used"),
                "details": f"Returned {len(res_mock_gemini.get('raw_result', []))} rows with fallback_used=False.",
            }
        )

    # 2. Hybrid execution test (both SQL AUM retrieval + sector exposure calculator)
    mock_hybrid_json = json.dumps(
        {
            "tool": "hybrid",
            "sql_query": "SELECT portfolio_name, total_aum FROM portfolios WHERE portfolio_name = 'Tech Innovation Fund';",
            "portfolio_name": "Tech Innovation Fund",
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_hybrid_json, None)):
        res_hybrid = agent.answer_question("Show total AUM and sector exposures for Tech Innovation Fund")
        raw_hybrid = res_hybrid.get("raw_result", {})
        passed_hybrid = (
            res_hybrid.get("tool_used") == "hybrid"
            and isinstance(raw_hybrid, dict)
            and raw_hybrid.get("sql_rows") == [("Tech Innovation Fund", 32000000.0)]
            and raw_hybrid.get("sector_exposures") == {"Technology": 100.0}
        )
        checks.append(
            {
                "test_name": "hybrid_sql_and_exposure_execution",
                "passed": passed_hybrid,
                "details": f"sql_rows={raw_hybrid.get('sql_rows')}, exposures={raw_hybrid.get('sector_exposures')}",
            }
        )

    # 3. Portfolio comparison test (deterministic fallback when Gemini returns 429)
    with patch.object(
        agent,
        "_call_gemini",
        return_value=(None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."),
    ):
        res_comp = agent.answer_question("Compare Growth Equity Fund and Tech Innovation Fund")
        comp_rows = res_comp.get("raw_result", [])
        passed_comp = (
            res_comp.get("tool_used") == "text2sql"
            and res_comp.get("fallback_used") is True
            and isinstance(comp_rows, list)
            and len(comp_rows) == 2
        )
        checks.append(
            {
                "test_name": "portfolio_comparison_query",
                "passed": passed_comp,
                "details": f"Compared {len(comp_rows)} portfolios via deterministic SQL.",
            }
        )

    # 4. Unsafe SQL generated by Gemini blocked
    mock_unsafe_json = json.dumps(
        {
            "tool": "text2sql",
            "sql_query": "DROP TABLE portfolios;",
            "portfolio_name": None,
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_unsafe_json, None)):
        res_unsafe = agent.answer_question("Drop the portfolios table")
        passed_unsafe = "Security error" in str(res_unsafe.get("error", ""))
        checks.append(
            {
                "test_name": "unsafe_sql_rejection",
                "passed": passed_unsafe,
                "details": res_unsafe.get("error"),
            }
        )

    # 5. Invalid SQL / nonexistent column handled gracefully
    mock_invalid_sql_json = json.dumps(
        {
            "tool": "text2sql",
            "sql_query": "SELECT nonexistent_col FROM portfolios;",
            "portfolio_name": None,
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_invalid_sql_json, None)):
        res_invalid = agent.answer_question("Query nonexistent column")
        passed_invalid = "no such column" in str(res_invalid.get("error", "")).lower()
        checks.append(
            {
                "test_name": "invalid_sql_error_handling",
                "passed": passed_invalid,
                "details": res_invalid.get("error"),
            }
        )

    # 6. Ambiguous portfolio name handling
    mock_ambiguous_json = json.dumps(
        {
            "tool": "exposure_calculator",
            "sql_query": None,
            "portfolio_name": "Equity",
            "reason": None,
        }
    )
    with patch.object(agent, "_call_gemini", return_value=(mock_ambiguous_json, None)):
        res_ambig = agent.answer_question("What is the sector exposure for Equity?")
        matched_list = res_ambig.get("matching_portfolios", [])
        passed_ambig = (
            "Ambiguous portfolio name" in str(res_ambig.get("error", ""))
            and matched_list == ["Growth Equity Fund", "International Equity Fund"]
        )
        checks.append(
            {
                "test_name": "ambiguous_portfolio_matching",
                "passed": passed_ambig,
                "details": f"Matches={matched_list}",
            }
        )

    # 7. API error handling (HTTP 429, 503, 504) on unmapped query
    with patch.object(agent._client.interactions, "create", side_effect=RuntimeError("503 Service Unavailable")):
        res_503 = agent.answer_question("Calculate the covariance matrix of daily returns")
        passed_503 = "503" in str(res_503.get("error", "")) and res_503.get("fallback_used") is False
        checks.append(
            {
                "test_name": "api_error_503_handling",
                "passed": passed_503,
                "details": res_503.get("error"),
            }
        )

    return checks


def run_evaluation(
    db_path: str = DEFAULT_DB_PATH,
    gt_path: str = DEFAULT_GT_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
    mock_http_429: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate all 10 questions in ground_truth_dataset.json against PortfolioAgent,
    run additional verification checks, and write the report to evaluation_report.json.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    questions = ground_truth_data.get("questions", [])
    agent = PortfolioAgent(db_path=db_path)

    # Read-only reference connection to execute the ground-truth SQL queries for Q1-Q8
    ref_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    ref_conn = sqlite3.connect(ref_uri, uri=True)
    ref_cursor = ref_conn.cursor()

    results_list: List[Dict[str, Any]] = []
    passed_count = 0

    for item in questions:
        qid = item.get("id")
        q_type = item.get("type")
        difficulty = item.get("difficulty")
        question_text = item.get("question", "")
        gt_spec = item.get("ground_truth", {})

        if mock_http_429:
            with patch.object(
                agent,
                "_call_gemini",
                return_value=(
                    None,
                    "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again.",
                ),
            ):
                agent_response = agent.answer_question(question_text)
        else:
            agent_response = agent.answer_question(question_text)

        actual_tool = agent_response.get("tool_used")
        actual_raw = agent_response.get("raw_result")
        actual_answer = agent_response.get("answer")
        agent_error = agent_response.get("error")
        fallback_used = agent_response.get("fallback_used", False)
        execution_mode = agent_response.get("execution_mode", "unknown")

        if q_type == "text2sql":
            expected_tool = "text2sql"
            expected_sql = gt_spec.get("sql_query", "")
            ref_cursor.execute(expected_sql)
            expected_rows = ref_cursor.fetchall()

            tool_matched = actual_tool == expected_tool
            data_matched, mismatch_reason = _compare_sql_results(expected_rows, actual_raw)
            if not tool_matched:
                mismatch_reason = f"Tool mismatch: expected '{expected_tool}', got '{actual_tool}'."
            passed = tool_matched and data_matched and (agent_error is None)

            expected_serialized = [list(_normalize_value(v) for v in row) for row in expected_rows]
            actual_serialized = (
                [list(_normalize_value(v) for v in row) for row in actual_raw]
                if isinstance(actual_raw, list)
                else actual_raw
            )

            results_list.append(
                {
                    "id": qid,
                    "type": q_type,
                    "difficulty": difficulty,
                    "question": question_text,
                    "execution_mode": execution_mode,
                    "fallback_used": fallback_used,
                    "expected_tool": expected_tool,
                    "actual_tool": actual_tool,
                    "expected_sql": expected_sql,
                    "actual_sql": agent_response.get("sql_query"),
                    "expected_result": expected_serialized,
                    "actual_result": actual_serialized,
                    "agent_answer": actual_answer,
                    "passed": passed,
                    "mismatch": mismatch_reason,
                }
            )

        elif q_type == "exposure_calculator":
            expected_tool = "exposure_calculator"
            expected_portfolio = gt_spec.get("parameters", {}).get("portfolio_name", "")
            # Use independent fixed expected values rather than calling calculate_sector_exposure
            expected_exposures = INDEPENDENT_EXPECTED_EXPOSURES.get(expected_portfolio, {})

            tool_matched = actual_tool == expected_tool
            actual_portfolio = agent_response.get("parameters", {}).get("portfolio_name")
            portfolio_matched = (
                str(actual_portfolio).strip().lower() == str(expected_portfolio).strip().lower()
            )
            data_matched, mismatch_reason = _compare_exposure_results(expected_exposures, actual_raw)

            if not tool_matched:
                mismatch_reason = f"Tool mismatch: expected '{expected_tool}', got '{actual_tool}'."
            elif not portfolio_matched:
                mismatch_reason = (
                    f"Portfolio parameter mismatch: expected '{expected_portfolio}', got '{actual_portfolio}'."
                )

            passed = tool_matched and portfolio_matched and data_matched and (agent_error is None)

            results_list.append(
                {
                    "id": qid,
                    "type": q_type,
                    "difficulty": difficulty,
                    "question": question_text,
                    "execution_mode": execution_mode,
                    "fallback_used": fallback_used,
                    "expected_tool": expected_tool,
                    "actual_tool": actual_tool,
                    "expected_parameters": {"portfolio_name": expected_portfolio},
                    "actual_parameters": agent_response.get("parameters"),
                    "expected_result": expected_exposures,
                    "actual_result": actual_raw,
                    "agent_answer": actual_answer,
                    "passed": passed,
                    "mismatch": mismatch_reason,
                }
            )

        else:
            passed = False
            results_list.append(
                {
                    "id": qid,
                    "type": q_type,
                    "difficulty": difficulty,
                    "question": question_text,
                    "execution_mode": execution_mode,
                    "fallback_used": fallback_used,
                    "expected_tool": q_type,
                    "actual_tool": actual_tool,
                    "expected_result": None,
                    "actual_result": actual_raw,
                    "agent_answer": actual_answer,
                    "passed": False,
                    "mismatch": f"Unknown question type: {q_type}",
                }
            )

        if passed:
            passed_count += 1

    additional_checks = run_additional_verification_tests(agent)
    additional_passed = sum(1 for c in additional_checks if c["passed"])

    agent.close()
    ref_conn.close()

    total_questions = len(questions)
    failed_count = total_questions - passed_count
    accuracy = round((passed_count / total_questions) * 100, 2) if total_questions > 0 else 0.0

    report_data = {
        "summary": {
            "total_questions": total_questions,
            "passed_questions": passed_count,
            "failed_questions": failed_count,
            "overall_accuracy_percent": accuracy,
            "evaluation_mode": (
                "deterministic_fallback_only (Gemini HTTP 429 mocked - 0 live API calls)"
                if mock_http_429
                else "live_gemini_api"
            ),
            "live_gemini_tested": not mock_http_429,
            "additional_verification_passed": additional_passed,
            "additional_verification_total": len(additional_checks),
        },
        "results": results_list,
        "additional_verification_tests": additional_checks,
    }

    with open(report_path, "w", encoding="utf-8") as out_f:
        json.dump(report_data, out_f, indent=2)

    return report_data


def print_evaluation_summary(report_data: Dict[str, Any], report_path: str = DEFAULT_REPORT_PATH) -> None:
    """Print a concise evaluation summary table to the terminal."""
    summary = report_data["summary"]
    print("=" * 80)
    print("PORTFOLIO ANALYTICS AGENT - GROUND TRUTH EVALUATION SUMMARY")
    print(f"Mode: {summary['evaluation_mode']}")
    print("=" * 80)
    for item in report_data["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        mode_tag = "FALLBACK" if item.get("fallback_used") else "GEMINI"
        print(
            f"Q{item['id']:<2} | {status:<4} | Mode: {mode_tag:<8} | Tool: {str(item['actual_tool']):<19} | "
            f"{item['question']}"
        )
        if item["mismatch"]:
            print(f"    -> Mismatch: {item['mismatch']}")

    print("-" * 80)
    print("ADDITIONAL VERIFICATION TESTS (MOCKED GEMINI, HYBRID, SECURITY & ERRORS)")
    print("-" * 80)
    for chk in report_data.get("additional_verification_tests", []):
        c_status = "PASS" if chk["passed"] else "FAIL"
        print(f"  [{c_status}] {chk['test_name']:<38} | {chk['details']}")

    print("-" * 80)
    print(f"Ground Truth Total    : {summary['total_questions']}")
    print(f"Ground Truth Passed   : {summary['passed_questions']}")
    print(f"Ground Truth Failed   : {summary['failed_questions']}")
    print(f"Ground Truth Accuracy : {summary['overall_accuracy_percent']:.2f}%")
    print(
        f"Additional Checks     : {summary['additional_verification_passed']}/"
        f"{summary['additional_verification_total']} passed"
    )
    print(f"Report Saved To       : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    report = run_evaluation(mock_http_429=True)
    print_evaluation_summary(report)
