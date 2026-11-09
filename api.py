"""FastAPI REST API for the Portfolio Analytics Agent, reusing existing read-only SQLite, exposure, and risk tools."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
import uvicorn

from agent import PortfolioAgent
from api_schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    HealthResponse,
    PortfolioAnalyticsResponse,
    PortfolioDetailResponse,
    PortfolioListResponse,
    PortfolioSummaryItem,
    SectorExposureResponse,
)
from exposure_tool import calculate_sector_exposure, resolve_portfolio_name
from risk_analytics import compute_portfolio_risk_analytics

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "portfolio_database.db")

_shared_agent: Optional[PortfolioAgent] = None


def get_api_agent() -> PortfolioAgent:
    """Return a shared read-only PortfolioAgent instance for the FastAPI application."""
    global _shared_agent
    if _shared_agent is None or _shared_agent.db_connection is None:
        _shared_agent = PortfolioAgent(db_path=DB_PATH)
    return _shared_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanly close the shared read-only PortfolioAgent connection."""
    agent = get_api_agent()
    yield
    if agent is not None:
        agent.close()


app = FastAPI(
    title="Portfolio Analytics Agent REST API",
    description=(
        "REST API providing read-only portfolio details, equity sector exposures, "
        "NumPy/SciPy statistical risk analytics, and natural-language AI agent querying."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _resolve_or_raise(agent: PortfolioAgent, portfolio_name: str) -> str:
    """Resolve portfolio name or raise 400 (ambiguous/invalid) or 404 (not found)."""
    if agent.db_connection is None:
        raise HTTPException(status_code=503, detail="Portfolio database connection is unavailable.")

    resolution = resolve_portfolio_name(agent.db_connection, portfolio_name)
    if "error" in resolution:
        err_msg = resolution.get("user_message", resolution["error"])
        if "matching_portfolios" in resolution:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": err_msg,
                    "matching_portfolios": resolution["matching_portfolios"],
                },
            )
        if "not found" in resolution["error"].lower():
            raise HTTPException(status_code=404, detail=err_msg)
        raise HTTPException(status_code=400, detail=err_msg)

    return str(resolution["portfolio_name"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Verify API and read-only SQLite database health."""
    agent = get_api_agent()
    res = agent.execute_sql("SELECT COUNT(*) FROM portfolios;")
    if "error" in res or not res.get("rows"):
        raise HTTPException(status_code=503, detail="Database health check failed.")

    count = int(res["rows"][0][0])
    return HealthResponse(
        status="ok",
        database_connected=True,
        total_portfolios=count,
        security_mode="SQLite URI mode=ro + PRAGMA query_only=ON + Authorizer",
    )


@app.get("/portfolios", response_model=PortfolioListResponse, tags=["Portfolios"])
def list_portfolios() -> PortfolioListResponse:
    """List all available portfolios in the database."""
    agent = get_api_agent()
    res = agent.execute_sql(
        """
        SELECT portfolio_id, portfolio_name, creation_date, target_risk_level,
               total_aum, strategy_type, benchmark_index, status
        FROM portfolios
        ORDER BY portfolio_id ASC;
        """
    )
    if "error" in res:
        raise HTTPException(status_code=500, detail=res["user_message"])

    items = [
        PortfolioSummaryItem(
            portfolio_id=int(r[0]),
            portfolio_name=str(r[1]),
            creation_date=str(r[2]),
            target_risk_level=str(r[3]),
            total_aum=float(r[4]),
            strategy_type=str(r[5]),
            benchmark_index=str(r[6]),
            status=str(r[7]),
        )
        for r in res.get("rows", [])
    ]
    return PortfolioListResponse(count=len(items), portfolios=items)


@app.get("/portfolios/{portfolio_name}", response_model=PortfolioDetailResponse, tags=["Portfolios"])
def get_portfolio_details(portfolio_name: str) -> PortfolioDetailResponse:
    """Retrieve details, AUM, and holdings summary for a specific portfolio."""
    agent = get_api_agent()
    resolved_name = _resolve_or_raise(agent, portfolio_name)
    safe_name = resolved_name.replace("'", "''")

    res = agent.execute_sql(
        f"""
        SELECT p.portfolio_id, p.portfolio_name, p.creation_date, p.target_risk_level,
               p.total_aum, p.strategy_type, p.benchmark_index, p.status,
               COUNT(h.holding_id) AS holdings_count,
               ROUND(COALESCE(SUM(h.quantity * s.current_price), 0.0), 2) AS holdings_market_value
        FROM portfolios p
        LEFT JOIN holdings h ON p.portfolio_id = h.portfolio_id
        LEFT JOIN securities s ON h.security_id = s.security_id
        WHERE p.portfolio_name = '{safe_name}'
        GROUP BY p.portfolio_id, p.portfolio_name, p.creation_date, p.target_risk_level,
                 p.total_aum, p.strategy_type, p.benchmark_index, p.status;
        """
    )
    if "error" in res or not res.get("rows"):
        raise HTTPException(status_code=404, detail=f"Portfolio '{resolved_name}' not found.")

    r = res["rows"][0]
    return PortfolioDetailResponse(
        portfolio_id=int(r[0]),
        portfolio_name=str(r[1]),
        creation_date=str(r[2]),
        target_risk_level=str(r[3]),
        total_aum=float(r[4]),
        strategy_type=str(r[5]),
        benchmark_index=str(r[6]),
        status=str(r[7]),
        holdings_count=int(r[8]),
        holdings_market_value=float(r[9]),
    )


@app.get("/portfolios/{portfolio_name}/exposure", response_model=SectorExposureResponse, tags=["Portfolios"])
def get_portfolio_exposure(portfolio_name: str) -> SectorExposureResponse:
    """Calculate equity sector exposures (excluding bonds) for a portfolio."""
    agent = get_api_agent()
    resolved_name = _resolve_or_raise(agent, portfolio_name)
    exp_res = calculate_sector_exposure(agent.db_connection, resolved_name)
    if "error" in exp_res:
        raise HTTPException(status_code=400, detail=exp_res.get("user_message", exp_res["error"]))

    return SectorExposureResponse(
        portfolio_id=int(exp_res["portfolio_id"]),
        portfolio_name=str(exp_res["portfolio_name"]),
        sector_exposures=exp_res.get("sector_exposures", {}),
    )


@app.get("/portfolios/{portfolio_name}/analytics", response_model=PortfolioAnalyticsResponse, tags=["Portfolios"])
def get_portfolio_analytics(portfolio_name: str) -> PortfolioAnalyticsResponse:
    """Retrieve combined AUM, sector exposure, NumPy/SciPy statistical metrics, and recorded DB risk metrics."""
    agent = get_api_agent()
    resolved_name = _resolve_or_raise(agent, portfolio_name)
    analytics = compute_portfolio_risk_analytics(agent, resolved_name)
    if "error" in analytics:
        raise HTTPException(status_code=400, detail=analytics["error"])

    return PortfolioAnalyticsResponse(**analytics)


@app.post("/agent/query", response_model=AgentQueryResponse, tags=["AI Agent"])
def query_agent(payload: AgentQueryRequest) -> AgentQueryResponse:
    """Submit a natural-language analytics question to PortfolioAgent."""
    agent = get_api_agent()
    res = agent.answer_question(payload.question)
    if res.get("execution_mode") == "connection_error":
        raise HTTPException(status_code=503, detail=res.get("answer", "Database connection unavailable."))

    return AgentQueryResponse(
        question=str(res.get("question", payload.question)),
        answer=str(res.get("answer", "")),
        tool_used=res.get("tool_used"),
        tools_used=res.get("tools_used"),
        portfolio_identified=res.get("portfolio_identified"),
        execution_mode=str(res.get("execution_mode", "unknown")),
        fallback_used=bool(res.get("fallback_used", False)),
        fallback_reason=res.get("fallback_reason"),
        explanation=res.get("explanation"),
        sql_query=res.get("sql_query"),
        aum_result=res.get("aum_result"),
        sector_exposure_result=res.get("sector_exposure_result"),
        execution_status=res.get("execution_status"),
        error=res.get("error"),
        matching_portfolios=res.get("matching_portfolios"),
        raw_result=res.get("raw_result"),
    )


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
