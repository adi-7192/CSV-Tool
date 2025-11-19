"""
Pydantic response models
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


class ChatResponse(BaseModel):
    """AI chat response"""
    answer: str
    sql: Optional[str] = None
    conversation_id: Optional[str] = None


class MetricsResponse(BaseModel):
    """Metrics response with core KPIs including net revenue and rates"""
    revenue: float
    refunds: float
    shipping_loss: float
    free_replacement_cost: float
    net_revenue: float
    net_margin: float
    refund_rate: float
    orders: int
    avg_order_value: float
    success_rate: float
    transaction_breakdown: Dict[str, Any]


class UploadResponse(BaseModel):
    """Upload response with data quality warnings."""
    success: bool
    rows_processed: int
    rows_inserted: int
    validation_warnings: list[str]
    ingestion_id: str
    filename: Optional[str] = None

