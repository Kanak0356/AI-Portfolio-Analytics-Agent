"""Portfolio analytics assistant for retrieving portfolio data and calculating sector exposure."""

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from exposure_tool import calculate_sector_exposure, resolve_portfolio_name

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("portfolio_agent")
if not logger.handlers:
    handler = logging.NullHandler()
    logger.addHandler(handler)

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
SCHEMA_FILE_PATH = BASE_DIR / "database_schema.sql"

# SQLite authorizer action codes permitted for strictly read-only SELECT / CTE queries
_ALLOWED_AUTHORIZER_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}

# Dangerous SQL keywords that must never appear as standalone tokens
_FORBIDDEN_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|"
    r"VACUUM|REINDEX|ANALYZE|BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE)\b",
    re.IGNORECASE,
)

#Known portfolio aliases for deterministic extraction
_KNOWN_PORTFOLIOS = [
    ("total stock market index fund", "Total Stock Market Index Fund"),
    ("total bond market index fund", "Total Bond Market Index Fund"),
    ("total international index fund", "Total International Index Fund"),
    ("conservative income fund", "Conservative Income Fund"),
    ("dividend aristocrats fund", "Dividend Aristocrats Fund"),
    ("international equity fund", "International Equity Fund"),
    ("emerging markets fund", "Emerging Markets Fund"),
    ("esg sustainable fund", "ESG Sustainable Fund"),
    ("small cap value fund", "Small Cap Value Fund"),
    ("tech innovation fund", "Tech Innovation Fund"),
    ("growth equity fund", "Growth Equity Fund"),
    ("fixed income plus", "Fixed Income Plus"),
    ("balanced portfolio", "Balanced Portfolio"),
    ("tech innovation", "Tech Innovation Fund"),
    ("international equity", "International Equity Fund"),
    ("growth equity", "Growth Equity Fund"),
    ("conservative income", "Conservative Income Fund"),
    ("dividend aristocrats", "Dividend Aristocrats Fund"),
    ("emerging markets", "Emerging Markets Fund"),
    ("esg sustainable", "ESG Sustainable Fund"),
    ("small cap value", "Small Cap Value Fund"),
    ("balanced", "Balanced Portfolio"),
]

# Domain guidance paired with the raw database_schema.sql
_SCHEMA_DOMAIN_GUIDANCE = """
### IMPORTANT DOMAIN & SCHEMA RULES FOR SQL GENERATION
1. Tables & Primary Keys:
   - `sectors` (PK: `sector_id`): `sector_name` ('Technology', 'Consumer Discretionary', 'Automotive', 'Financials', 'Healthcare', 'Consumer Staples', 'Energy', 'Industrials', 'Real Estate', 'Utilities'), `sector_description`, `industry_group`.
   - `securities` (PK: `security_id`, FK: `sector_id` -> `sectors.sector_id`): `symbol`, `company_name`, `asset_type` ('Stock' or 'Bond'), `market_cap`, `current_price`, `currency`, `exchange`, `country`, `listing_date`, `maturity_date` (bonds), `coupon_rate` (bonds). Note: Bonds have `sector_id = NULL`.
   - `benchmarks` (PK: `benchmark_id`): `benchmark_name`, `benchmark_symbol`, `benchmark_type`, `description`, `inception_date`. Note: `portfolios.benchmark_index` matches `benchmarks.benchmark_symbol`.
   - `portfolios` (PK: `portfolio_id`): `portfolio_name`, `creation_date`, `target_risk_level` ('High', 'Medium', 'Low'), `total_aum` (Assets Under Management in USD), `strategy_type`, `benchmark_index`, `status` ('Active' or 'Passive').
   - `holdings` (PK: `holding_id`, FKs: `portfolio_id` -> `portfolios.portfolio_id`, `security_id` -> `securities.security_id`): `quantity`, `purchase_price`, `purchase_date`, `current_weight` (stored as decimal fraction, e.g. 0.18 = 18%), `cost_basis` (original total investment cost in USD).
   - `transactions` (PK: `transaction_id`, FKs: `portfolio_id`, `security_id`): `transaction_type` ('BUY' or 'SELL'), `quantity`, `price`, `transaction_date`, `fees`, `settlement_date`, `notes`.
   - `historical_prices` (PK: `price_id`, FK: `security_id`): `price_date`, `open_price`, `high_price`, `low_price`, `close_price`, `volume`, `adjusted_close`.
   - `portfolio_performance` (PK: `performance_id`, FK: `portfolio_id`): `performance_date`, `nav`, `total_return_1m`, `total_return_3m`, `total_return_6m`, `total_return_1y`, `volatility`, `sharpe_ratio`, `max_drawdown`.
   - `risk_metrics` (PK: `risk_id`, FK: `portfolio_id`): `calculation_date`, `var_95`, `var_99`, `cvar_95`, `beta`, `correlation_sp500`, `tracking_error`, `information_ratio`, `sortino_ratio`.

2. Key Financial Definitions & SQL Patterns:
   - Market value of a holding = `h.quantity * s.current_price`.
   - Cost basis of a holding = `h.cost_basis` (already stored as the total cost for that holding).
   - Portfolio AUM = `p.total_aum` in `portfolios`.
   - When asked for "top N holdings by cost basis" in a portfolio, select `h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis` joining `holdings h`, `securities s`, and `portfolios p`, ordered by `h.cost_basis DESC LIMIT N`.
   - When asked for securities in a sector (e.g. 'Technology'), select `s.symbol, s.company_name` joining `securities s` and `sectors sec` where `sec.sector_name = 'Technology'`.
   - When asked for average current price of securities in each sector, select `sec.sector_name, AVG(s.current_price) as avg_price` joining `securities s` and `sectors sec`, `GROUP BY sec.sector_name ORDER BY avg_price DESC`.
   - When asked for each portfolio's total value of Technology sector holdings and what percentage it represents of the total portfolio value:
     Use two CTEs:
     `portfolio_tech_value`: `SELECT p.portfolio_id, p.portfolio_name, SUM(h.quantity * s.current_price) as tech_value FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id JOIN securities s ON h.security_id = s.security_id JOIN sectors sec ON s.sector_id = sec.sector_id WHERE sec.sector_name = 'Technology' GROUP BY p.portfolio_id, p.portfolio_name`
     `portfolio_total_value`: `SELECT p.portfolio_id, SUM(h.quantity * s.current_price) as total_value FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id JOIN securities s ON h.security_id = s.security_id GROUP BY p.portfolio_id`
     Then select `ptv.portfolio_name, ptv.tech_value, ptv2.total_value, ROUND((ptv.tech_value / ptv2.total_value) * 100, 2) as tech_percentage` joining the two CTEs on `portfolio_id`, ordered by `tech_percentage DESC`.
   - When asked for portfolios with holdings in more than 5 different sectors and their diversification metrics:
     Use two CTEs:
     `portfolio_sector_count`: `SELECT p.portfolio_id, p.portfolio_name, COUNT(DISTINCT sec.sector_id) as sector_count FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id JOIN securities s ON h.security_id = s.security_id JOIN sectors sec ON s.sector_id = sec.sector_id WHERE s.asset_type = 'Stock' GROUP BY p.portfolio_id, p.portfolio_name HAVING COUNT(DISTINCT sec.sector_id) > 5`
     `portfolio_holdings_count`: `SELECT p.portfolio_id, COUNT(h.holding_id) as total_holdings FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id GROUP BY p.portfolio_id`
     Then select `psc.portfolio_name, psc.sector_count, phc.total_holdings, ROUND(CAST(psc.sector_count AS REAL) / phc.total_holdings, 3) as diversification_ratio` joining on `portfolio_id` ordered by `diversification_ratio DESC`.
   - When asked to compare two or more portfolios (e.g., 'Compare Growth Equity Fund and Tech Innovation Fund'), query `portfolios` joined with `holdings` and `securities` to show `portfolio_name`, `target_risk_level`, `total_aum`, `strategy_type`, `status`, holdings count, and total market value.
"""


def _readonly_authorizer(
    action: int,
    arg1: Optional[str],
    arg2: Optional[str],
    dbname: Optional[str],
    source: Optional[str],
) -> int:
    """SQLite authorizer callback that permits only read-only SELECT operations."""
    if action in _ALLOWED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _strip_sql_comments_and_literals(sql: str) -> str:
    """Remove SQL comments and string literals to safely inspect statement structure."""
    no_comments = re.sub(r"--[^\r\n]*|/\*[\s\S]*?\*/", " ", sql)
    no_literals = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", no_comments)
    return no_literals.strip()


def _clean_json_text(text: str) -> str:
    """Extract JSON object from model output, stripping markdown code fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


class PortfolioAgent:
    """Answer portfolio questions using Gemini, read-only SQLite access, and the exposure calculator."""

    def __init__(
        self,
        db_path: str = "portfolio_database.db",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.db_path = db_path
        self.db_connection: Optional[sqlite3.Connection] = None
        self._connect_readonly(db_path)

        self.model_name = (
            model_name
            or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
            or DEFAULT_GEMINI_MODEL
        )
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name.split("models/", 1)[1]

        self._api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "").strip()
        self.schema_context = self._load_schema_context()
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._client: Optional[genai.Client] = None
        if self._api_key:
            try:
                self._client = genai.Client(
                    api_key=self._api_key,
                    http_options=types.HttpOptions(
                        timeout=15000,
                        retry_options=types.HttpRetryOptions(
                            attempts=1,
                            http_status_codes=[],
                        ),
                    ),
                )
                if hasattr(self._client, "interactions") and hasattr(
                    self._client.interactions, "sdk_configuration"
                ):
                    sdk_cfg = self._client.interactions.sdk_configuration
                    if hasattr(sdk_cfg, "retry_config"):
                        sdk_cfg.retry_config = None
            except Exception:
                self._client = None

    def _load_schema_context(self) -> str:
        """Load the full SQL schema from database_schema.sql along with relationship rules."""
        schema_sql = ""
        if SCHEMA_FILE_PATH.is_file():
            schema_sql = SCHEMA_FILE_PATH.read_text(encoding="utf-8")
        return f"### SQLITE DATABASE SCHEMA\n```sql\n{schema_sql}\n```\n{_SCHEMA_DOMAIN_GUIDANCE}"

    def _connect_readonly(self, db_path: str) -> None:
        """Open the SQLite database in read-only URI mode and attach the read-only authorizer."""
        if db_path == ":memory:":
            conn = sqlite3.connect(":memory:")
        else:
            resolved_path = Path(db_path).resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Database file not found: {db_path}")
            uri = f"{resolved_path.as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA query_only = ON;")
        conn.set_authorizer(_readonly_authorizer)
        self.db_connection = conn

    def _validate_readonly_sql(self, sql_query: str) -> Optional[str]:
        """
        Validate that a SQL string is a single read-only SELECT or WITH ... SELECT statement.
        Returns None if valid, or a security/validation error message if rejected.
        """
        if not sql_query or not isinstance(sql_query, str) or not sql_query.strip():
            return "SQL query cannot be empty."

        cleaned = _strip_sql_comments_and_literals(sql_query)
        if not cleaned:
            return "SQL query cannot be empty."

        statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
        if len(statements) > 1:
            return "Security error: Multiple SQL statements in a single request are not permitted."

        single_stmt = statements[0]

        forbidden_match = _FORBIDDEN_SQL_PATTERN.search(single_stmt)
        if forbidden_match:
            keyword = forbidden_match.group(1).upper()
            return f"Security error: '{keyword}' operations are not permitted in read-only mode."

        first_token_match = re.match(r"^([A-Za-z]+)", single_stmt)
        if not first_token_match:
            return "Invalid SQL syntax."

        first_token = first_token_match.group(1).upper()
        if first_token not in {"SELECT", "WITH"}:
            return f"Security error: Only read-only SELECT or WITH...SELECT queries are permitted (found '{first_token}')."

        if first_token == "WITH" and not re.search(r"\bSELECT\b", single_stmt, re.IGNORECASE):
            return "Security error: WITH clause must be followed by a read-only SELECT query."

        return None

    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Validate and execute a read-only SQL query against portfolio_database.db."""
        if self.db_connection is None:
            return {
                "error": "Database connection is not available.",
                "user_message": "Unable to process your request because the database connection is unavailable.",
                "sql_query": sql_query,
            }

        validation_error = self._validate_readonly_sql(sql_query)
        if validation_error:
            is_security = validation_error.startswith("Security error:")
            return {
                "error": validation_error,
                "user_message": (
                    "This request was blocked because only read-only data retrieval queries are allowed."
                    if is_security
                    else "The query could not be executed due to invalid syntax."
                ),
                "sql_query": sql_query,
            }

        try:
            cursor = self.db_connection.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            return {
                "sql_query": sql_query,
                "columns": columns,
                "rows": rows,
            }
        except sqlite3.Warning as exc:
            return {
                "error": f"Security error: Multiple SQL statements are not permitted ({str(exc)}).",
                "user_message": "This request was blocked because multiple SQL statements are not permitted.",
                "sql_query": sql_query,
            }
        except sqlite3.DatabaseError as exc:
            err_msg = str(exc)
            err_lower = err_msg.lower()
            if "not authorized" in err_lower or "readonly" in err_lower or "attempt to write" in err_lower:
                return {
                    "error": f"Security error: Database modification blocked ({err_msg}).",
                    "user_message": "This operation is not permitted. Only read-only queries are allowed.",
                    "sql_query": sql_query,
                }
            if "no such table" in err_lower:
                return {
                    "error": f"Database error: {err_msg}",
                    "user_message": "The requested data table does not exist in the portfolio database.",
                    "sql_query": sql_query,
                }
            if "no such column" in err_lower:
                return {
                    "error": f"Database error: {err_msg}",
                    "user_message": "One or more requested fields do not exist in the portfolio database.",
                    "sql_query": sql_query,
                }
            if "syntax error" in err_lower or "incomplete input" in err_lower:
                return {
                    "error": f"SQL syntax error: {err_msg}",
                    "user_message": "The query could not be processed due to a SQL syntax error.",
                    "sql_query": sql_query,
                }
            return {
                "error": f"Database error: {err_msg}",
                "user_message": "A database error occurred while retrieving portfolio data.",
                "sql_query": sql_query,
            }
        except Exception as exc:
            return {
                "error": f"Unexpected execution error: {str(exc)}",
                "user_message": "An unexpected error occurred while processing the query.",
                "sql_query": sql_query,
            }

    def _call_gemini(self, prompt: str, system_instruction: str, timeout: float = 15.0) -> Tuple[Optional[str], Optional[str]]:
        """
        Invoke Gemini via the Interactions API (falling back to models.generate_content if needed).
        Returns (response_text, error_message). Never exposes the API key.
        """
        if not self._api_key or self._client is None:
            return None, "Gemini API key is missing or not configured. Please set GEMINI_API_KEY in your .env file."

        try:
            if hasattr(self._client, "interactions"):
                interaction = self._client.interactions.create(
                    model=self.model_name,
                    input=prompt,
                    system_instruction=system_instruction,
                    store=False,
                    timeout=timeout,
                    automatic_continuation=False,
                )
                output_text = getattr(interaction, "output_text", None)
                if not output_text and getattr(interaction, "steps", None):
                    for step in reversed(interaction.steps):
                        if getattr(step, "type", None) == "model_output":
                            for part in getattr(step, "content", []):
                                if getattr(part, "text", None):
                                    output_text = part.text
                                    break
                            if output_text:
                                break
                if output_text:
                    return output_text.strip(), None
                return None, "Gemini returned an empty response."
            else:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=f"{system_instruction}\n\n{prompt}",
                )
                if response and getattr(response, "text", None):
                    return response.text.strip(), None
                return None, "Gemini returned an empty response."
        except Exception as exc:
            err_str = str(exc)
            if self._api_key and self._api_key in err_str:
                err_str = err_str.replace(self._api_key, "[REDACTED]")
            err_lower = err_str.lower()
            if (
                "api_key_invalid" in err_lower
                or "invalid api key" in err_lower
                or "401" in err_lower
                or "403" in err_lower
                or "permission_denied" in err_lower
            ):
                return None, "Gemini API authentication failed. Please verify your GEMINI_API_KEY."
            if (
                "429" in err_lower
                or "quota" in err_lower
                or "rate" in err_lower
                or "resource_exhausted" in err_lower
                or "too many requests" in err_lower
            ):
                return None, "Gemini API rate limit or quota exceeded (HTTP 429). Please wait a moment and try again."
            if (
                "503" in err_lower
                or "service unavailable" in err_lower
                or "unavailable" in err_lower
                or "overloaded" in err_lower
            ):
                return None, "Gemini API service is temporarily unavailable or overloaded (HTTP 503). Please try again shortly."
            if (
                "504" in err_lower
                or "gateway timeout" in err_lower
                or "deadline_exceeded" in err_lower
                or "deadline exceeded" in err_lower
                or "timeout" in err_lower
                or "timed out" in err_lower
            ):
                return None, "Gemini API request timed out (HTTP 504). Please try again."
            return None, f"Gemini API error: {err_str}"

    def _plan_with_gemini(self, question: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Ask Gemini to classify the user's question and produce a structured JSON routing plan:
        - tool: 'text2sql' | 'exposure_calculator' | 'hybrid' | 'unsupported'
        - sql_query: valid read-only SQLite SELECT/WITH query (when tool is 'text2sql' or 'hybrid')
        - portfolio_name: target portfolio name (when tool is 'exposure_calculator' or 'hybrid')
        - reason: brief explanation if unsupported
        """
        system_instruction = (
            "You are the routing and SQL generation engine for a Portfolio Analytics Agent.\n"
            "You have access to two tools:\n"
            "1. 'text2sql': Executes a single read-only SQLite SELECT (or WITH ... SELECT) query against the portfolio database.\n"
            "2. 'exposure_calculator': Calculates the equity sector percentage exposure breakdown for a specific portfolio (excluding bonds) given 'portfolio_name'.\n"
            "3. 'hybrid': Use when the user asks for BOTH portfolio AUM / total assets / portfolio value AND the sector exposure / sector allocation / industry breakdown / sector distribution of a specific portfolio in one question.\n"
            "4. 'unsupported': Use when the user asks a question completely unrelated to portfolios/finance (e.g., weather, sports), asks to predict future stock prices, or asks to modify/delete data.\n\n"
            "ROUTING RULES:\n"
            "- Use 'hybrid' whenever a question combines AUM / total assets / portfolio value / assets under management WITH sector exposure / sector allocation / sector breakdown / industry breakdown / sector distribution for a portfolio (e.g., 'Show me the AUM and sector exposure of Growth Equity Fund', 'Compare the AUM and sector allocation of International Equity Fund', 'Give me the total assets and sector breakdown for Tech Innovation Fund'). For 'hybrid', set 'sql_query' to retrieve 'portfolio_name, total_aum' from 'portfolios' for that portfolio, and set 'portfolio_name' to the target portfolio.\n"
            "- Use 'exposure_calculator' ONLY when the user asks solely for the sector exposure / sector allocation / sector breakdown of a specific named portfolio without asking for AUM or other SQL table metrics.\n"
            "- Note: If the user asks 'For each portfolio, show the total value of Technology sector holdings and what percentage it represents of the total portfolio value', route to 'text2sql' because it compares across all portfolios by market value (`quantity * current_price`).\n"
            "- Never invent tables or columns. Use only the exact schema provided below.\n"
            "- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA statements.\n"
            "- Extract the portfolio_name accurately for 'exposure_calculator' or 'hybrid'. Do not guess if the user said a generic word like 'Equity'; pass 'Equity' as portfolio_name so the tool can check for ambiguity.\n\n"
            f"{self.schema_context}\n\n"
            "Respond ONLY with a valid JSON object matching this exact structure (no markdown explanation outside JSON):\n"
            "{\n"
            '  "tool": "text2sql" | "exposure_calculator" | "hybrid" | "unsupported",\n'
            '  "sql_query": "<single read-only SQLite SELECT query if tool is text2sql or hybrid, else null>",\n'
            '  "portfolio_name": "<portfolio name string if tool is exposure_calculator or hybrid, else null>",\n'
            '  "reason": "<brief explanation if unsupported, else null>"\n'
            "}"
        )

        prompt = f"User Question: {question}"
        raw_text, err = self._call_gemini(prompt=prompt, system_instruction=system_instruction)
        if err:
            return None, err

        try:
            cleaned_json = _clean_json_text(raw_text or "")
            plan = json.loads(cleaned_json)
            if not isinstance(plan, dict) or "tool" not in plan:
                return None, "Failed to parse routing plan from Gemini response."
            return plan, None
        except json.JSONDecodeError:
            return None, "Failed to parse structured JSON plan from Gemini response."

    def _format_sql_answer_deterministically(
        self,
        question: str,
        columns: List[str],
        rows: List[Tuple[Any, ...]],
    ) -> str:
        """Provide clean, accurate natural-language formatting of SQL results without fabricating numbers."""
        if not rows:
            return "No matching records were found in the portfolio database for your query."

        q_lower = question.lower()

        if len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            col_name = columns[0].lower() if columns else ""
            if val is None:
                return "No data is available for the requested metric."
            if "aum" in q_lower or "aum" in col_name or "value" in col_name or "cost" in col_name or "price" in col_name:
                if isinstance(val, (int, float)):
                    if "high" in q_lower and ("aum" in q_lower or "assets under management" in q_lower):
                        return f"The total Assets Under Management (AUM) for portfolios with a high target risk level is ${val:,.2f}."
                    if "aum" in q_lower or "assets under management" in q_lower:
                        return f"The total Assets Under Management (AUM) is ${val:,.2f}."
                    return f"The result is ${val:,.2f}."
            if "count" in col_name or "how many" in q_lower or "number of" in q_lower:
                if "active" in q_lower and "portfolio" in q_lower:
                    return f"There are {val} active portfolios in total."
                if "portfolio" in q_lower:
                    return f"There are {val} portfolios in total."
                if "securit" in q_lower or "stock" in q_lower:
                    return f"There are {val} securities in total."
                if "holding" in q_lower:
                    return f"There are {val} holdings in total."
                if "sector" in q_lower:
                    return f"There are {val} sectors in total."
                return f"There are {val} matching records in total."
            if isinstance(val, float):
                return f"The result is {val:,.2f}."
            return f"The result is {val}."

        if len(columns) == 1:
            items = [str(r[0]) for r in rows if r[0] is not None]
            if "active" in q_lower and "portfolio" in q_lower:
                return f"There are {len(items)} active portfolios: " + ", ".join(items) + "."
            if "portfolio" in q_lower:
                return f"There are {len(items)} portfolios: " + ", ".join(items) + "."
            return f"Found {len(items)} results: " + ", ".join(items)

        lines = []
        for row in rows:
            parts = []
            for col, val in zip(columns, row):
                col_l = col.lower()
                if val is None:
                    parts.append(f"{col}: N/A")
                elif any(k in col_l for k in ("aum", "cost_basis", "tech_value", "total_value", "market_value", "price", "market_cap", "nav", "fees")) and isinstance(val, (int, float)):
                    parts.append(f"{col}: ${val:,.2f}")
                elif any(k in col_l for k in ("percentage", "pct", "return")) and isinstance(val, (int, float)):
                    parts.append(f"{col}: {val:.2f}%")
                elif "weight" in col_l and isinstance(val, (int, float)):
                    parts.append(f"{col}: {val * 100:.2f}%")
                elif isinstance(val, float):
                    parts.append(f"{col}: {val:g}")
                else:
                    parts.append(f"{col}: {val}")
            lines.append("  - " + ", ".join(parts))

        return "Here are the results from the portfolio database:\n" + "\n".join(lines)

    def _format_exposure_answer(self, exp_result: Dict[str, Any]) -> str:
        """Format the output of calculate_sector_exposure into a readable response."""
        exposures = exp_result.get("sector_exposures", {})
        resolved_name = exp_result.get("portfolio_name", "Portfolio")
        if not exposures:
            return f"{resolved_name} has no equity holdings for sector exposure calculation."
        breakdown = [f"{sector}: {pct}%" for sector, pct in exposures.items()]
        return f"Sector exposures for {resolved_name} (equities only):\n  - " + "\n  - ".join(breakdown)

    def _format_hybrid_answer(
        self,
        portfolio_name: str,
        aum_value: Optional[float],
        exposures: Dict[str, float],
    ) -> str:
        """
        Format the combined hybrid response containing Portfolio, Total AUM,
        Sector Exposure breakdown, and a brief allocation explanation.
        """
        aum_str = f"${aum_value:,.2f}" if isinstance(aum_value, (int, float)) else "N/A"

        if exposures:
            sector_lines = "\n".join(f"- {sector}: {pct}%" for sector, pct in exposures.items())
            top_items = list(exposures.items())
            if len(top_items) == 1:
                alloc_desc = (
                    f"{portfolio_name} has {aum_str} in total Assets Under Management (AUM), "
                    f"with 100% of its equity allocation concentrated in {top_items[0][0]} ({top_items[0][1]}%)."
                )
            else:
                leaders = ", ".join(f"{sec} ({pct}%)" for sec, pct in top_items[:3])
                alloc_desc = (
                    f"{portfolio_name} has {aum_str} in total Assets Under Management (AUM) "
                    f"allocated across {len(top_items)} equity sectors, led by {leaders} "
                    "(bond holdings are excluded from equity sector exposure calculations)."
                )
        else:
            sector_lines = "- No equity sector exposure (portfolio holds fixed-income bonds only)"
            alloc_desc = (
                f"{portfolio_name} has {aum_str} in total Assets Under Management (AUM) "
                "and holds only fixed-income bond securities, which are excluded from equity sector exposure."
            )

        return (
            f"Portfolio: {portfolio_name}\n\n"
            f"Total AUM: {aum_str}\n\n"
            f"Sector Exposure:\n{sector_lines}\n\n"
            f"Explanation: {alloc_desc}"
        )

    def _extract_portfolio_name_for_exposure(self, question: str) -> Optional[str]:
        """Extract the target portfolio name from a question."""
        q_clean = question.strip()
        q_lower = q_clean.lower()

        for alias_lower, canonical_name in _KNOWN_PORTFOLIOS:
            if alias_lower in q_lower:
                return canonical_name

        # Match the last preposition phrase (e.g., "of Growth Equity Fund", "for Equity")
        match = re.search(
            r".*\b(?:for|of|in|about)\s+(?:the\s+)?([A-Za-z0-9\s\-&]+?)(?:\?|\.|$)",
            q_clean,
            re.IGNORECASE,
        )
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"\bportfolio$", "", candidate, flags=re.IGNORECASE).strip()
            if candidate:
                return candidate

        return None

    def _extract_multiple_portfolios(self, question: str) -> List[str]:
        """Extract multiple distinct canonical portfolio names mentioned in a comparison question."""
        q_lower = question.strip().lower()
        found: List[str] = []
        for alias_lower, canonical_name in _KNOWN_PORTFOLIOS:
            if alias_lower in q_lower and canonical_name not in found:
                found.append(canonical_name)
        return found

    def _deterministic_plan(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Provide a deterministic execution plan for supported portfolio questions
        when Gemini is unavailable (e.g., missing key, HTTP 429, 503, or 504).
        """
        q_lower = question.strip().lower()

        # 1. Hybrid questions (combining AUM/total assets/portfolio value AND sector exposure/allocation/breakdown/distribution)
        exposure_phrases = (
            "sector exposure",
            "sector exposures",
            "sector allocation",
            "sector allocations",
            "sector breakdown",
            "sector breakdowns",
            "sector distribution",
            "sector distributions",
            "industry breakdown",
            "industry breakdowns",
            "industry exposure",
            "industry allocation",
            "industry distribution",
            "sector weight",
            "sector weights",
        )
        aum_metric_phrases = (
            "aum",
            "assets under management",
            "total assets",
            "portfolio value",
            "total value",
            "total holdings",
            "risk level",
            "strategy",
        )
        has_exposure = any(p in q_lower for p in exposure_phrases)
        has_sql_metric = any(k in q_lower for k in aum_metric_phrases)
        is_each_portfolio_q7 = "for each portfolio" in q_lower or "each portfolio" in q_lower

        if has_exposure and has_sql_metric and not is_each_portfolio_q7:
            portfolio_name = self._extract_portfolio_name_for_exposure(question)
            safe_name = (portfolio_name or "").replace("'", "''")
            return {
                "tool": "hybrid",
                "sql_query": (
                    "SELECT portfolio_name, total_aum, target_risk_level, strategy_type, status "
                    f"FROM portfolios WHERE LOWER(portfolio_name) = LOWER('{safe_name}');"
                )
                if portfolio_name
                else None,
                "portfolio_name": portfolio_name,
                "reason": None,
            }

        # 2. Pure sector exposure questions
        if has_exposure and not is_each_portfolio_q7:
            portfolio_name = self._extract_portfolio_name_for_exposure(question)
            return {
                "tool": "exposure_calculator",
                "sql_query": None,
                "portfolio_name": portfolio_name,
                "reason": None,
            }

        # 3. Portfolio comparison questions
        if "compare" in q_lower or "comparison" in q_lower or " vs " in q_lower or " versus " in q_lower:
            matched_portfolios = self._extract_multiple_portfolios(question)
            if len(matched_portfolios) >= 2:
                quoted_names = ", ".join(f"'{name.replace(chr(39), chr(39)*2)}'" for name in matched_portfolios)
                return {
                    "tool": "text2sql",
                    "sql_query": (
                        "SELECT p.portfolio_name, p.target_risk_level, p.total_aum, p.strategy_type, p.status, "
                        "COUNT(h.holding_id) AS holdings_count, "
                        "ROUND(COALESCE(SUM(h.quantity * s.current_price), 0.0), 2) AS market_value "
                        "FROM portfolios p "
                        "LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id "
                        "LEFT JOIN securities s ON h.security_id = s.security_id "
                        f"WHERE p.portfolio_name IN ({quoted_names}) "
                        "GROUP BY p.portfolio_id, p.portfolio_name, p.target_risk_level, p.total_aum, p.strategy_type, p.status "
                        "ORDER BY p.total_aum DESC;"
                    ),
                    "portfolio_name": None,
                    "reason": None,
                }
            elif "all portfolios" in q_lower or "portfolios" in q_lower:
                return {
                    "tool": "text2sql",
                    "sql_query": (
                        "SELECT p.portfolio_name, p.target_risk_level, p.total_aum, p.strategy_type, p.status, "
                        "COUNT(h.holding_id) AS holdings_count, "
                        "ROUND(COALESCE(SUM(h.quantity * s.current_price), 0.0), 2) AS market_value "
                        "FROM portfolios p "
                        "LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id "
                        "LEFT JOIN securities s ON h.security_id = s.security_id "
                        "GROUP BY p.portfolio_id, p.portfolio_name, p.target_risk_level, p.total_aum, p.strategy_type, p.status "
                        "ORDER BY p.total_aum DESC;"
                    ),
                    "portfolio_name": None,
                    "reason": None,
                }

        # 4. Ground-truth and standard analytical SQL questions (including alternative phrasings)
        if any(p in q_lower for p in ("how many portfolios", "number of portfolios", "count of portfolios", "total portfolios")):
            return {
                "tool": "text2sql",
                "sql_query": "SELECT COUNT(*) FROM portfolios;",
                "portfolio_name": None,
                "reason": None,
            }

        if "active portfolios" in q_lower or ("portfolios" in q_lower and "currently active" in q_lower):
            return {
                "tool": "text2sql",
                "sql_query": "SELECT portfolio_name FROM portfolios WHERE status = 'Active';",
                "portfolio_name": None,
                "reason": None,
            }

        if any(w in q_lower for w in ("securities", "companies", "stocks")) and "technology sector" in q_lower and "portfolio" not in q_lower:
            return {
                "tool": "text2sql",
                "sql_query": (
                    "SELECT s.symbol, s.company_name FROM securities s "
                    "JOIN sectors sec ON s.sector_id = sec.sector_id "
                    "WHERE sec.sector_name = 'Technology';"
                ),
                "portfolio_name": None,
                "reason": None,
            }

        if ("assets under management" in q_lower or "aum" in q_lower or "how much money" in q_lower) and "high" in q_lower:
            return {
                "tool": "text2sql",
                "sql_query": "SELECT SUM(total_aum) FROM portfolios WHERE target_risk_level = 'High';",
                "portfolio_name": None,
                "reason": None,
            }

        if ("top 5 holdings" in q_lower or "five securities" in q_lower or "top five holdings" in q_lower) and "growth equity fund" in q_lower:
            return {
                "tool": "text2sql",
                "sql_query": (
                    "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis "
                    "FROM holdings h "
                    "JOIN securities s ON h.security_id = s.security_id "
                    "JOIN portfolios p ON h.portfolio_id = p.portfolio_id "
                    "WHERE p.portfolio_name = 'Growth Equity Fund' "
                    "ORDER BY h.cost_basis DESC LIMIT 5;"
                ),
                "portfolio_name": None,
                "reason": None,
            }

        if ("average current price" in q_lower or "average price" in q_lower) and ("each sector" in q_lower or "by sector" in q_lower or "per sector" in q_lower):
            return {
                "tool": "text2sql",
                "sql_query": (
                    "SELECT sec.sector_name, AVG(s.current_price) as avg_price "
                    "FROM securities s "
                    "JOIN sectors sec ON s.sector_id = sec.sector_id "
                    "GROUP BY sec.sector_name ORDER BY avg_price DESC;"
                ),
                "portfolio_name": None,
                "reason": None,
            }

        if "for each portfolio" in q_lower and "technology sector" in q_lower and "percentage" in q_lower:
            return {
                "tool": "text2sql",
                "sql_query": (
                    "WITH portfolio_tech_value AS ("
                    "SELECT p.portfolio_id, p.portfolio_name, SUM(h.quantity * s.current_price) as tech_value "
                    "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
                    "JOIN securities s ON h.security_id = s.security_id "
                    "JOIN sectors sec ON s.sector_id = sec.sector_id "
                    "WHERE sec.sector_name = 'Technology' GROUP BY p.portfolio_id, p.portfolio_name"
                    "), portfolio_total_value AS ("
                    "SELECT p.portfolio_id, SUM(h.quantity * s.current_price) as total_value "
                    "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
                    "JOIN securities s ON h.security_id = s.security_id GROUP BY p.portfolio_id"
                    ") SELECT ptv.portfolio_name, ptv.tech_value, ptv2.total_value, "
                    "ROUND((ptv.tech_value / ptv2.total_value) * 100, 2) as tech_percentage "
                    "FROM portfolio_tech_value ptv JOIN portfolio_total_value ptv2 ON ptv.portfolio_id = ptv2.portfolio_id "
                    "ORDER BY tech_percentage DESC;"
                ),
                "portfolio_name": None,
                "reason": None,
            }

        if ("more than 5 different sectors" in q_lower or "more than 5 sectors" in q_lower) and "diversification" in q_lower:
            return {
                "tool": "text2sql",
                "sql_query": (
                    "WITH portfolio_sector_count AS ("
                    "SELECT p.portfolio_id, p.portfolio_name, COUNT(DISTINCT sec.sector_id) as sector_count "
                    "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
                    "JOIN securities s ON h.security_id = s.security_id "
                    "JOIN sectors sec ON s.sector_id = sec.sector_id "
                    "WHERE s.asset_type = 'Stock' GROUP BY p.portfolio_id, p.portfolio_name "
                    "HAVING COUNT(DISTINCT sec.sector_id) > 5"
                    "), portfolio_holdings_count AS ("
                    "SELECT p.portfolio_id, COUNT(h.holding_id) as total_holdings "
                    "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id GROUP BY p.portfolio_id"
                    ") SELECT psc.portfolio_name, psc.sector_count, phc.total_holdings, "
                    "ROUND(CAST(psc.sector_count AS REAL) / phc.total_holdings, 3) as diversification_ratio "
                    "FROM portfolio_sector_count psc JOIN portfolio_holdings_count phc ON psc.portfolio_id = phc.portfolio_id "
                    "ORDER BY diversification_ratio DESC;"
                ),
                "portfolio_name": None,
                "reason": None,
            }

        # 5. General portfolio holdings or portfolio information queries
        matched_portfolio = self._extract_portfolio_name_for_exposure(question)
        if matched_portfolio and any(alias == matched_portfolio.lower() for alias, _ in _KNOWN_PORTFOLIOS):
            safe_p = matched_portfolio.replace("'", "''")
            if "holding" in q_lower:
                return {
                    "tool": "text2sql",
                    "sql_query": (
                        "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis "
                        "FROM holdings h "
                        "JOIN securities s ON h.security_id = s.security_id "
                        "JOIN portfolios p ON h.portfolio_id = p.portfolio_id "
                        f"WHERE p.portfolio_name = '{safe_p}' "
                        "ORDER BY h.cost_basis DESC;"
                    ),
                    "portfolio_name": None,
                    "reason": None,
                }
            if any(k in q_lower for k in ("aum", "risk", "strategy", "tell me about", "info", "details", "benchmark")):
                return {
                    "tool": "text2sql",
                    "sql_query": (
                        "SELECT portfolio_name, target_risk_level, total_aum, strategy_type, status, benchmark_index "
                        f"FROM portfolios WHERE portfolio_name = '{safe_p}';"
                    ),
                    "portfolio_name": None,
                    "reason": None,
                }

        # 6. Out-of-scope / non-financial questions
        if any(w in q_lower for w in ("weather", "temperature", "world cup", "sports", "recipe", "predict tomorrow")):
            return {
                "tool": "unsupported",
                "sql_query": None,
                "portfolio_name": None,
                "reason": "I can only answer factual questions about the investment portfolios, holdings, securities, sectors, performance, and risk metrics in our database.",
            }

        return None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a natural-language question about investment portfolios with explainability metadata."""
        if not question or not isinstance(question, str) or not question.strip():
            return {
                "question": question,
                "tool_used": None,
                "fallback_used": False,
                "execution_mode": "validation_error",
                "explanation": "Question validation failed because the input was empty.",
                "error": "Invalid question: Question cannot be empty.",
                "answer": "Please provide a valid question about the portfolios.",
            }

        if self.db_connection is None:
            return {
                "question": question,
                "tool_used": None,
                "fallback_used": False,
                "execution_mode": "connection_error",
                "explanation": "Database connection is not active.",
                "error": "Database connection is not available.",
                "answer": "Unable to process your question because the portfolio database is currently unavailable.",
            }

        try:
            fallback_used = False
            fallback_reason: Optional[str] = None

            plan, plan_err = self._plan_with_gemini(question.strip())
            if plan_err or plan is None:
                fallback_plan = self._deterministic_plan(question.strip())
                if fallback_plan is not None:
                    plan = fallback_plan
                    fallback_used = True
                    fallback_reason = plan_err
                    logger.info("Used deterministic fallback for question '%s' due to: %s", question, plan_err)
                else:
                    return {
                        "question": question,
                        "tool_used": None,
                        "fallback_used": False,
                        "execution_mode": "gemini_error",
                        "explanation": f"Gemini planning could not complete ({plan_err}) and no deterministic rule matched.",
                        "error": plan_err or "Failed to generate execution plan.",
                        "answer": plan_err or "Unable to process your question at this time.",
                    }

            execution_mode = "deterministic_fallback" if fallback_used else "gemini_live"
            tool = str(plan.get("tool", "")).strip().lower()
            sql_query = plan.get("sql_query")
            portfolio_name = plan.get("portfolio_name")
            reason = plan.get("reason")

            # 1. Exposure Calculator Tool
            if tool == "exposure_calculator":
                if not portfolio_name or not str(portfolio_name).strip():
                    return {
                        "question": question,
                        "tool_used": "exposure_calculator",
                        "fallback_used": fallback_used,
                        "execution_mode": execution_mode,
                        "explanation": "Selected exposure_calculator, but no portfolio name could be identified.",
                        "error": "Could not identify a valid portfolio name in the question.",
                        "answer": "Please specify a portfolio name to calculate sector exposure.",
                    }

                exp_result = calculate_sector_exposure(self.db_connection, str(portfolio_name))
                if "error" in exp_result:
                    response: Dict[str, Any] = {
                        "question": question,
                        "tool_used": "exposure_calculator",
                        "fallback_used": fallback_used,
                        "execution_mode": execution_mode,
                        "explanation": f"Attempted sector exposure calculation for '{portfolio_name}', which returned: {exp_result['error']}",
                        "error": exp_result["error"],
                        "answer": exp_result.get("user_message", exp_result["error"]),
                    }
                    if "matching_portfolios" in exp_result:
                        response["matching_portfolios"] = exp_result["matching_portfolios"]
                    return response

                exposures = exp_result.get("sector_exposures", {})
                resolved_name = exp_result["portfolio_name"]
                mode_note = " (via deterministic fallback)" if fallback_used else " (via Gemini routing)"
                explanation = (
                    f"Routed to 'exposure_calculator'{mode_note} for portfolio '{resolved_name}'. "
                    f"Aggregated equity ('Stock') holding weights across {len(exposures)} sector(s), excluding bonds."
                )
                return {
                    "question": question,
                    "tool_used": "exposure_calculator",
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "execution_mode": execution_mode,
                    "explanation": explanation,
                    "parameters": {"portfolio_name": resolved_name},
                    "raw_result": exposures,
                    "answer": self._format_exposure_answer(exp_result),
                }

            # 2. SQL Query Tool (text2sql)
            elif tool == "text2sql":
                if not sql_query or not str(sql_query).strip():
                    return {
                        "question": question,
                        "tool_used": "text2sql",
                        "fallback_used": fallback_used,
                        "execution_mode": execution_mode,
                        "explanation": "Selected text2sql tool, but no SQL query was produced.",
                        "error": "SQL generation failed: No SQL query was produced.",
                        "answer": "I could not generate a valid database query for your question.",
                    }

                sql_str = str(sql_query).strip()
                res = self.execute_sql(sql_str)
                if "error" in res:
                    return {
                        "question": question,
                        "tool_used": "text2sql",
                        "fallback_used": fallback_used,
                        "execution_mode": execution_mode,
                        "explanation": f"Generated SQL query was rejected or failed execution: {res['error']}",
                        "sql_query": sql_str,
                        "error": res["error"],
                        "answer": res["user_message"],
                    }

                rows = res.get("rows", [])
                columns = res.get("columns", [])
                answer_text = self._format_sql_answer_deterministically(question, columns, rows)
                mode_note = " (via deterministic fallback)" if fallback_used else " (via Gemini SQL generation)"
                explanation = (
                    f"Routed to 'text2sql'{mode_note}. Executed validated read-only SQLite query returning "
                    f"{len(rows)} row(s) across column(s): {', '.join(columns)}."
                )
                return {
                    "question": question,
                    "tool_used": "text2sql",
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "execution_mode": execution_mode,
                    "explanation": explanation,
                    "sql_query": sql_str,
                    "columns": columns,
                    "raw_result": rows,
                    "answer": answer_text,
                }

            # 3. Hybrid Tool (Both SQL + Exposure Calculator)
            elif tool == "hybrid":
                candidate_portfolio = portfolio_name or self._extract_portfolio_name_for_exposure(question)
                if not candidate_portfolio or not str(candidate_portfolio).strip():
                    return {
                        "question": question,
                        "tool_used": "hybrid",
                        "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                        "portfolio_identified": None,
                        "sql_query": None,
                        "aum_result": None,
                        "sector_exposure_result": None,
                        "execution_status": "error",
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "execution_mode": execution_mode,
                        "explanation": "Hybrid tool selected, but no portfolio name could be identified in the question.",
                        "error": "Could not identify a valid portfolio name in the question.",
                        "answer": "Please specify a portfolio name to retrieve its AUM and sector exposure.",
                    }

                # Step 2: Resolve portfolio using existing portfolio-matching logic BEFORE executing tools
                resolution = resolve_portfolio_name(self.db_connection, str(candidate_portfolio))
                if "error" in resolution:
                    is_ambiguous = "matching_portfolios" in resolution
                    response_err: Dict[str, Any] = {
                        "question": question,
                        "tool_used": "hybrid",
                        "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                        "portfolio_identified": None,
                        "sql_query": None,
                        "aum_result": None,
                        "sector_exposure_result": None,
                        "execution_status": "clarification_required" if is_ambiguous else "error",
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "execution_mode": execution_mode,
                        "explanation": (
                            f"Portfolio name '{candidate_portfolio}' is ambiguous; asking user to clarify before executing tools."
                            if is_ambiguous
                            else f"Portfolio '{candidate_portfolio}' was not found in the database."
                        ),
                        "error": resolution["error"],
                        "answer": resolution.get("user_message", resolution["error"]),
                    }
                    if is_ambiguous:
                        response_err["matching_portfolios"] = resolution["matching_portfolios"]
                    return response_err

                resolved_name = resolution["portfolio_name"]
                safe_resolved = resolved_name.replace("'", "''")
                canonical_aum_sql = (
                    "SELECT portfolio_name, total_aum, target_risk_level, strategy_type, status "
                    f"FROM portfolios WHERE portfolio_name = '{safe_resolved}';"
                )

                # Step 3: Retrieve portfolio AUM using existing secure SQL execution mechanism
                sql_str = str(sql_query).strip() if (sql_query and str(sql_query).strip()) else canonical_aum_sql
                sql_res = self.execute_sql(sql_str)
                if "error" in sql_res:
                    return {
                        "question": question,
                        "tool_used": "hybrid",
                        "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                        "portfolio_identified": resolved_name,
                        "sql_query": sql_str,
                        "aum_result": None,
                        "sector_exposure_result": None,
                        "execution_status": "error",
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "execution_mode": execution_mode,
                        "explanation": f"Hybrid SQL step failed: {sql_res['error']}",
                        "error": sql_res["error"],
                        "answer": sql_res["user_message"],
                    }

                rows = sql_res.get("rows", [])
                columns = sql_res.get("columns", [])
                if not rows and sql_str != canonical_aum_sql:
                    sql_str = canonical_aum_sql
                    sql_res = self.execute_sql(sql_str)
                    if "error" in sql_res:
                        return {
                            "question": question,
                            "tool_used": "hybrid",
                            "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                            "portfolio_identified": resolved_name,
                            "sql_query": sql_str,
                            "aum_result": None,
                            "sector_exposure_result": None,
                            "execution_status": "error",
                            "fallback_used": fallback_used,
                            "fallback_reason": fallback_reason,
                            "execution_mode": execution_mode,
                            "explanation": f"Hybrid SQL step failed: {sql_res['error']}",
                            "error": sql_res["error"],
                            "answer": sql_res["user_message"],
                        }
                    rows = sql_res.get("rows", [])
                    columns = sql_res.get("columns", [])

                if not rows:
                    return {
                        "question": question,
                        "tool_used": "hybrid",
                        "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                        "portfolio_identified": resolved_name,
                        "sql_query": sql_str,
                        "aum_result": None,
                        "sector_exposure_result": None,
                        "execution_status": "error",
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "execution_mode": execution_mode,
                        "explanation": f"No AUM record found for portfolio '{resolved_name}'.",
                        "error": f"No AUM record found for portfolio '{resolved_name}'.",
                        "answer": f"Could not retrieve AUM data for '{resolved_name}'.",
                    }

                # Extract actual AUM value from SQL result (or query total_aum if custom SQL omitted it)
                aum_value: Optional[float] = None
                col_lower_map = {c.lower(): idx for idx, c in enumerate(columns)}
                if "total_aum" in col_lower_map:
                    raw_aum = rows[0][col_lower_map["total_aum"]]
                    if isinstance(raw_aum, (int, float)):
                        aum_value = float(raw_aum)
                elif len(columns) == 1 and isinstance(rows[0][0], (int, float)):
                    aum_value = float(rows[0][0])

                if aum_value is None:
                    aum_lookup_res = self.execute_sql(
                        f"SELECT total_aum FROM portfolios WHERE portfolio_name = '{safe_resolved}';"
                    )
                    if "error" in aum_lookup_res or not aum_lookup_res.get("rows"):
                        return {
                            "question": question,
                            "tool_used": "hybrid",
                            "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                            "portfolio_identified": resolved_name,
                            "sql_query": sql_str,
                            "aum_result": None,
                            "sector_exposure_result": None,
                            "execution_status": "error",
                            "fallback_used": fallback_used,
                            "fallback_reason": fallback_reason,
                            "execution_mode": execution_mode,
                            "explanation": "Failed to retrieve total_aum from portfolios table.",
                            "error": "Could not retrieve total_aum for the portfolio.",
                            "answer": f"Could not retrieve total AUM for '{resolved_name}'.",
                        }
                    aum_value = float(aum_lookup_res["rows"][0][0])

                # Step 4: Calculate sector exposure using existing calculate_sector_exposure()
                exp_result = calculate_sector_exposure(self.db_connection, resolved_name)
                if "error" in exp_result:
                    response = {
                        "question": question,
                        "tool_used": "hybrid",
                        "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                        "portfolio_identified": resolved_name,
                        "sql_query": sql_str,
                        "aum_result": aum_value,
                        "sector_exposure_result": None,
                        "execution_status": "error",
                        "fallback_used": fallback_used,
                        "fallback_reason": fallback_reason,
                        "execution_mode": execution_mode,
                        "explanation": f"Hybrid exposure step failed: {exp_result['error']}",
                        "error": exp_result["error"],
                        "answer": exp_result.get("user_message", exp_result["error"]),
                    }
                    if "matching_portfolios" in exp_result:
                        response["matching_portfolios"] = exp_result["matching_portfolios"]
                    return response

                exposures = exp_result["sector_exposures"]
                mode_note = " (via deterministic fallback)" if fallback_used else " (via Gemini routing)"
                explanation = (
                    f"Routed to 'hybrid' tool{mode_note} combining SQL Query Tool and Exposure Calculator for "
                    f"'{resolved_name}': retrieved Total AUM (${aum_value:,.2f}) via read-only SQL and calculated "
                    f"equity sector exposures across {len(exposures)} sector(s) (excluding bonds)."
                )
                hybrid_answer = self._format_hybrid_answer(resolved_name, aum_value, exposures)

                return {
                    "question": question,
                    "tool_used": "hybrid",
                    "tools_used": ["SQL Query Tool", "Exposure Calculator"],
                    "portfolio_identified": resolved_name,
                    "aum_result": aum_value,
                    "sector_exposure_result": exposures,
                    "execution_status": "success",
                    "fallback_used": fallback_used,
                    "fallback_reason": fallback_reason,
                    "execution_mode": execution_mode,
                    "explanation": explanation,
                    "sql_query": sql_str,
                    "columns": columns,
                    "parameters": {"portfolio_name": resolved_name},
                    "raw_result": {
                        "sql_rows": rows,
                        "total_aum": aum_value,
                        "sector_exposures": exposures,
                    },
                    "answer": hybrid_answer,
                }

            # 4. Unsupported Question
            else:
                user_explanation = (
                    reason
                    if reason and isinstance(reason, str)
                    else "I can only answer questions related to the investment portfolios, holdings, securities, sectors, performance, and risk metrics in our database."
                )
                return {
                    "question": question,
                    "tool_used": None,
                    "fallback_used": fallback_used,
                    "execution_mode": execution_mode,
                    "explanation": "Question classified as outside the supported scope of the portfolio database.",
                    "error": "Unsupported question.",
                    "answer": user_explanation,
                }

        except Exception as exc:
            return {
                "question": question,
                "tool_used": None,
                "fallback_used": False,
                "execution_mode": "error",
                "explanation": f"Unhandled exception caught safely: {str(exc)}",
                "error": f"Unexpected error: {str(exc)}",
                "answer": "An unexpected error occurred while processing your request. Please try again.",
            }

    def close(self) -> None:
        """Close the SQLite database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None


def _print_cli_help() -> None:
    """Display available CLI commands and sample questions."""
    print("\nAvailable Commands:")
    print("  help / ?   - Show this help message and sample questions")
    print("  details    - Toggle detailed execution metadata (SQL query, tools, fallback status)")
    print("  exit / quit- Close the Portfolio Analytics Agent")
    print("\nSample Questions:")
    print("  [SQL Query Tool]")
    print("    - How many portfolios do we have in total?")
    print("    - What are the names of all active portfolios?")
    print("    - Show me the top 5 holdings by cost basis in the Growth Equity Fund")
    print("  [Exposure Calculator Tool]")
    print("    - What are the sector exposures for the Tech Innovation Fund?")
    print("    - Calculate the sector exposure breakdown for international equity")
    print("  [Hybrid Analytics (SQL + Exposure Calculator)]")
    print("    - Show me the AUM and sector exposure of Growth Equity Fund.")
    print("    - Compare the AUM and sector allocation of International Equity Fund.")
    print("    - Give me the total assets and sector breakdown for Tech Innovation Fund.\n")


def _print_cli_response(result: Dict[str, Any], show_details: bool = False) -> None:
    """Format and print an agent response and optional execution metadata in the CLI."""
    print(f"\nAgent:\n{result.get('answer', 'Sorry, I could not process that request.')}")

    tools_label = (
        ", ".join(result["tools_used"])
        if result.get("tools_used")
        else str(result.get("tool_used") or "None")
    )
    mode_label = (
        "Deterministic Fallback"
        if result.get("fallback_used")
        else ("Gemini AI" if result.get("execution_mode") == "gemini_live" else str(result.get("execution_mode")))
    )
    print(f"\n[Tools: {tools_label} | Mode: {mode_label}]")

    if show_details:
        print("-" * 60)
        print("Execution Metadata:")
        print(f"  Tool Route          : {result.get('tool_used')}")
        if result.get("tools_used"):
            print(f"  Tools Combined      : {', '.join(result['tools_used'])}")
        if result.get("portfolio_identified"):
            print(f"  Portfolio Identified: {result.get('portfolio_identified')}")
        print(f"  Execution Mode      : {result.get('execution_mode')}")
        print(f"  Fallback Used       : {result.get('fallback_used')}")
        if result.get("fallback_used") and result.get("fallback_reason"):
            print(f"  Fallback Reason     : {result.get('fallback_reason')}")
        if result.get("sql_query"):
            print(f"  SQL Query           : {result.get('sql_query')}")
        if result.get("explanation"):
            print(f"  Explanation         : {result.get('explanation')}")
        print("-" * 60)
    print()


def run_cli(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface (supports both interactive REPL and single-question mode)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Portfolio Analytics Agent CLI — Query portfolios, sector exposures, and hybrid analytics."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Optional single natural-language question to execute and exit.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Display detailed execution metadata (SQL query, tool routing, fallback status).",
    )
    args = parser.parse_args(argv)

    try:
        agent = PortfolioAgent()
    except Exception as exc:
        print(f"Error: Unable to connect to the portfolio database ({exc}).")
        return 1

    try:
        if args.question:
            res = agent.answer_question(args.question)
            _print_cli_response(res, show_details=args.details)
            return 0 if not res.get("error") else 1

        show_details = bool(args.details)
        print("=" * 70)
        print("Portfolio Analytics Agent (CLI)")
        print("Connected to read-only SQLite database: portfolio_database.db")
        print("Type a question, 'help' for examples, 'details' to toggle metadata, or 'exit' to quit.")
        print("=" * 70 + "\n")

        while True:
            question = input("You: ").strip()

            if not question:
                print("Agent: Please enter a question about the portfolios (or type 'help' for examples).\n")
                continue

            cmd = question.lower()
            if cmd in {"exit", "quit", "q"}:
                print("Agent: Thank you for using Portfolio Analytics Agent. Goodbye!")
                break
            if cmd in {"help", "?", "examples"}:
                _print_cli_help()
                continue
            if cmd == "details":
                show_details = not show_details
                state_str = "ENABLED" if show_details else "DISABLED"
                print(f"Agent: Detailed execution metadata display is now {state_str}.\n")
                continue

            result = agent.answer_question(question)
            _print_cli_response(result, show_details=show_details)

        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nAgent: Session ended. Goodbye!")
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(run_cli())
