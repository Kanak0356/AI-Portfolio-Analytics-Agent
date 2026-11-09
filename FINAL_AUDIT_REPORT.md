# Final End-to-End Audit Report — AI Portfolio Analytics Agent

**Audit Date:** 2026-11-09  
**Workspace:** `c:\Users\HP\OneDrive\Desktop\PL_AiPortfolioAgent`  
**Configured Gemini Model:** `gemini-3.1-flash-lite` (Live smoke test verified: `EXECUTION_MODE: gemini_live`, `FALLBACK_USED: False`)

---

## 1. Original Assessment & Extension Requirements vs. Implementation Checklist

| Category | Requirement | Status | Implementation File(s) & Verification |
| :--- | :--- | :--- | :--- |
| **Database & Schema** | Load `database_schema.sql` and all 9 CSV files (`data/*.csv`) into `portfolio_database.db` | **Implemented & Verified** | `database.py` creates `portfolio_database.db` with 9 tables (`603` total rows), foreign keys enabled, and 18 custom indices. `PRAGMA integrity_check` = `ok`; `PRAGMA foreign_key_check` = `[]`. |
| **Unmodified Source Files** | Preserve original `database_schema.sql`, `ground_truth_dataset.json`, `requirements.txt`, and all 9 CSV files unmodified | **Implemented & Verified** | Verified via SHA-256 byte-for-byte hash comparison against `Exercise_PL_AI_Engineer/Exercise_PL_AI_Engineer/` in `test_agent.py::test_database_integrity_and_unmodified_source_files`. |
| **Component 1: SQL Query Tool** | Convert natural-language portfolio questions into read-only SQL queries and execute against `portfolio_database.db` | **Implemented & Verified** | `PortfolioAgent.execute_sql()` in `agent.py` enforces 4-layer read-only protection (`?mode=ro`, `PRAGMA query_only = ON`, C-engine `set_authorizer`, and `_validate_readonly_sql`). |
| **Component 2: Exposure Calculator** | Calculate equity sector percentage exposures (`asset_type = 'Stock'`), excluding bonds, with exact, partial, and ambiguous portfolio matching | **Implemented & Verified** | `calculate_sector_exposure()` and `resolve_portfolio_name()` in `exposure_tool.py`. |
| **Component 3: Portfolio Agent & CLI** | Command-line and programmatic agent (`PortfolioAgent`) routing across `text2sql`, `exposure_calculator`, `hybrid`, and `unsupported` | **Implemented & Verified** | `PortfolioAgent` and `run_cli()` in `agent.py` support interactive REPL mode, single-question CLI mode, live `gemini-3.1-flash-lite` routing, and deterministic fallback. |
| **Component 4: Agent Evaluator** | Evaluate all 10 questions in `ground_truth_dataset.json` against expected answers | **Implemented & Verified** | `evaluator.py` achieves **10/10 (100.00%)** on ground-truth questions + **7/7** additional verification checks, writing `evaluation_report.json`. |
| **Hybrid Analytics** | Combine SQL Query Tool (AUM) and Exposure Calculator in one response with full tool metadata | **Implemented & Verified** | `tool = "hybrid"` in `agent.py` resolves portfolio name first, queries `total_aum` via `execute_sql()`, computes exposures via `calculate_sector_exposure()`, and returns structured output + metadata. |
| **Statistical Risk Analytics (NumPy, SciPy, Matplotlib, Seaborn)** | Calculate portfolio concentration and return statistics from actual DB data and render Matplotlib/Seaborn charts | **Implemented & Verified** | `risk_analytics.py` computes NumPy HHI, effective holdings, weighted daily return mean/std, and SciPy skewness, kurtosis, Shapiro-Wilk normality, parametric 95% VaR, and Pearson correlations, plus Matplotlib & Seaborn figures integrated into `app.py`. |
| **Streamlit Dashboard (`app.py`)** | Interactive 5-tab web dashboard with `"How This Answer Was Generated"` (`expanded=False`) and `width="stretch"` | **Implemented & Verified** | `app.py` verified via `streamlit.testing.v1.AppTest` with zero deprecation warnings. |
| **FastAPI REST API (`api.py`, `api_schemas.py`)** | Standalone REST API with `/health`, `/portfolios`, `/portfolios/{name}`, `/portfolios/{name}/exposure`, `/portfolios/{name}/analytics`, and `POST /agent/query` | **Implemented & Verified** | Implemented in `api.py` and `api_schemas.py` using FastAPI, Pydantic v2, and Uvicorn; verified via `httpx` / `TestClient` and live Uvicorn server check on `127.0.0.1:8000`. |

---

## 2. Test Results & Actual Pass/Fail Counts

### A. Single Live Gemini API Smoke Test (`gemini-3.1-flash-lite`)
- **Model ID:** `gemini-3.1-flash-lite`
- **Question:** `"How many portfolios do we have in total?"`
- **Result:** `EXECUTION_MODE: gemini_live`, `FALLBACK_USED: False`, `TOOL_USED: text2sql`, `SQL_QUERY: SELECT COUNT(*) FROM portfolios;`, `ANSWER: There are 13 portfolios in total.` (**PASS**)

### B. Ground-Truth Evaluator (`.\.venv\Scripts\python.exe evaluator.py`)
- **Ground-Truth Questions:** `10/10 Passed (100.00% Accuracy), 0 Failed`
- **Additional Verification Checks:** `7/7 Passed, 0 Failed`

### C. Full Pytest Suite (`.\.venv\Scripts\python.exe -m pytest -v`)
- **Total Tests:** `14`
- **Passed:** `14` / **Failed:** `0` (`39.27s`, 0 live API calls used during automated test suite)
  1. `test_ground_truth_evaluation_suite` — **PASSED**
  2. `test_independent_exposure_values` — **PASSED**
  3. `test_mocked_successful_gemini_sql_and_explainability` — **PASSED**
  4. `test_hybrid_execution_both_tools` — **PASSED**
  5. `test_cli_interface` — **PASSED**
  6. `test_portfolio_comparison_question` — **PASSED**
  7. `test_sql_security_enforcement` — **PASSED**
  8. `test_dashboard_concentration_analysis` — **PASSED**
  9. `test_streamlit_explainable_ai_expander_and_wording` — **PASSED**
  10. `test_alternative_phrasings_and_unsupported_questions` — **PASSED**
  11. `test_database_integrity_and_unmodified_source_files` — **PASSED**
  12. `test_numpy_scipy_matplotlib_seaborn_risk_analytics` — **PASSED**
  13. `test_fastapi_rest_endpoints_with_httpx` — **PASSED**
  14. `test_fastapi_validation_and_error_handling_with_httpx` — **PASSED**

### D. Live Uvicorn Server Verification (`127.0.0.1:8000`)
- `GET http://127.0.0.1:8000/health` → `200 OK`: `{"status": "ok", "database_connected": True, "total_portfolios": 13, "security_mode": "SQLite URI mode=ro + PRAGMA query_only=ON + Authorizer"}`
- `GET http://127.0.0.1:8000/portfolios/Growth%20Equity%20Fund/exposure` → `200 OK`: `{"portfolio_id": 1, "portfolio_name": "Growth Equity Fund", "sector_exposures": {"Technology": 93.0, "Consumer Discretionary": 7.0}}`

---

## 3. Dependency Usage Summary (`requirements.txt`)

All **13/13** packages in `requirements.txt` are classified as **USED** and verified by automated tests (see `DEPENDENCY_FEATURE_VERIFICATION.md` for the full matrix):
- `google-genai`, `python-dotenv`, `streamlit`, `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx`.

---

## 4. Exact Commands to Run the Project (Windows PowerShell)

```powershell
# 1. Install / verify all dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Run Ground-Truth Evaluator
.\.venv\Scripts\python.exe evaluator.py

# 3. Run Full 14-Test Pytest Suite
.\.venv\Scripts\python.exe -m pytest -v

# 4. Run Interactive CLI Agent
.\.venv\Scripts\python.exe agent.py

# 5. Launch Streamlit Dashboard (Terminal 1)
.\.venv\Scripts\python.exe -m streamlit run app.py

# 6. Launch FastAPI REST API Server on port 8000 (Terminal 2)
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```
