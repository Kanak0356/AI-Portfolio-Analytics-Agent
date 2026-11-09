"""Pydantic request and response schemas for the Portfolio Analytics FastAPI REST API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status ('ok' or 'degraded')")
    database_connected: bool = Field(..., description="Whether portfolio_database.db is accessible in read-only mode")
    total_portfolios: int = Field(..., description="Total number of portfolios in the database")
    security_mode: str = Field(..., description="Active SQLite security protections")


class PortfolioSummaryItem(BaseModel):
    portfolio_id: int
    portfolio_name: str
    creation_date: str
    target_risk_level: str
    total_aum: float
    strategy_type: str
    benchmark_index: str
    status: str


class PortfolioListResponse(BaseModel):
    count: int
    portfolios: List[PortfolioSummaryItem]


class PortfolioDetailResponse(BaseModel):
    portfolio_id: int
    portfolio_name: str
    creation_date: str
    target_risk_level: str
    total_aum: float
    strategy_type: str
    benchmark_index: str
    status: str
    holdings_count: int
    holdings_market_value: float


class SectorExposureResponse(BaseModel):
    portfolio_id: int
    portfolio_name: str
    sector_exposures: Dict[str, float]


class ConcentrationMetricsSchema(BaseModel):
    holdings_count: int
    total_holdings_market_value: float
    hhi_concentration_index: float
    effective_number_of_holdings: float
    mean_holding_weight_pct: float
    std_holding_weight_pct: float
    max_single_holding_weight_pct: float


class ReturnStatisticsSchema(BaseModel):
    data_source: str
    observation_days: int
    mean_daily_return_pct: Optional[float] = None
    std_daily_return_pct: Optional[float] = None
    min_daily_return_pct: Optional[float] = None
    max_daily_return_pct: Optional[float] = None
    median_daily_return_pct: Optional[float] = None
    skewness_scipy: Optional[float] = None
    kurtosis_scipy: Optional[float] = None
    parametric_var_95_daily_pct_scipy: Optional[float] = None
    shapiro_normality_p_value_scipy: Optional[float] = None


class RecordedDatabaseMetricsSchema(BaseModel):
    has_recorded_performance_history: bool
    latest_performance_date: Optional[str] = None
    latest_nav: Optional[float] = None
    total_return_1m_pct: Optional[float] = None
    total_return_3m_pct: Optional[float] = None
    total_return_6m_pct: Optional[float] = None
    total_return_1y_pct: Optional[float] = None
    recorded_volatility_pct: Optional[float] = None
    recorded_sharpe_ratio: Optional[float] = None
    recorded_max_drawdown_pct: Optional[float] = None
    recorded_var_95_pct: Optional[float] = None
    recorded_var_99_pct: Optional[float] = None
    recorded_cvar_95_pct: Optional[float] = None
    recorded_beta: Optional[float] = None
    recorded_correlation_sp500: Optional[float] = None
    recorded_tracking_error: Optional[float] = None
    recorded_information_ratio: Optional[float] = None
    recorded_sortino_ratio: Optional[float] = None


class PortfolioAnalyticsResponse(BaseModel):
    portfolio_id: int
    portfolio_name: str
    target_risk_level: str
    total_aum: float
    strategy_type: str
    benchmark_index: str
    status: str
    sector_exposures: Dict[str, float]
    calculated_concentration_metrics_numpy: ConcentrationMetricsSchema
    calculated_return_statistics_numpy_scipy: ReturnStatisticsSchema
    recorded_database_metrics: RecordedDatabaseMetricsSchema


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural-language portfolio analytics question")

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank or whitespace only.")
        return cleaned


class AgentQueryResponse(BaseModel):
    question: str
    answer: str
    tool_used: Optional[str] = None
    tools_used: Optional[List[str]] = None
    portfolio_identified: Optional[str] = None
    execution_mode: str
    fallback_used: bool
    fallback_reason: Optional[str] = None
    explanation: Optional[str] = None
    sql_query: Optional[str] = None
    aum_result: Optional[float] = None
    sector_exposure_result: Optional[Dict[str, float]] = None
    execution_status: Optional[str] = None
    error: Optional[str] = None
    matching_portfolios: Optional[List[str]] = None
    raw_result: Optional[Any] = None
