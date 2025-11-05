"""
Metrics API Endpoints - Serve KPIs to frontend
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from datetime import datetime, timedelta

from services.metrics_service import (
    calculate_metrics,
    get_revenue_trend,
    get_top_products,
)

router = APIRouter()


@router.get("/")
async def get_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type"),
    source_file: Optional[str] = Query(None, description="Filter by source file"),
):
    """
    Get core business metrics with transaction-aware calculations
    
    Returns:
        - gross_revenue: Gross revenue from shipments (SUM of Shipment transactions)
        - refund_amount: Total refund amount (SUM of Refund transactions)
        - cancellation_amount: Total cancellation amount (SUM of Cancel transactions)
        - free_replacement_cost: Estimated cost of free replacements (2x ASIN price)
        - net_revenue: Net revenue after all deductions
            Formula: gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
        - net_margin: Net margin percentage (net_revenue / gross_revenue * 100)
        - shipping_loss: Shipping costs lost on refunds (separate from refund_amount)
        - orders: Number of successful orders
        - avg_order_value: Average order value
        - success_rate: % of orders not refunded
        - transaction_breakdown: Count and amounts by transaction type
        - revenue: Alias for gross_revenue (backward compatibility)
        - refunds: Alias for refund_amount (backward compatibility)
    """
    try:
        # Default to last 30 days if no dates provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        metrics = calculate_metrics(start_date, end_date, transaction_type, source_file)
        
        return {
            "data": metrics,
            "period": {
                "start_date": start_date,
                "end_date": end_date,
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trend")
async def get_trend(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    group_by: str = Query('day', description="Group by: day, week, month"),
):
    """
    Get revenue trend data for charts
    """
    try:
        trend_df = get_revenue_trend(start_date, end_date, group_by)
        
        # Convert DataFrame to list of dicts for JSON response
        trend_data = trend_df.to_dict('records')
        
        # Format dates as strings
        for item in trend_data:
            if 'date' in item:
                item['date'] = str(item['date'])
        
        return {
            "data": trend_data,
            "count": len(trend_data),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-products")
async def get_top_products_endpoint(
    limit: int = Query(10, ge=1, le=100, description="Number of products"),
    start_date: Optional[str] = Query(None, description="Start date"),
    end_date: Optional[str] = Query(None, description="End date"),
):
    """
    Get top products by revenue
    """
    try:
        products_df = get_top_products(limit, start_date, end_date)
        
        return {
            "data": products_df.to_dict('records'),
            "count": len(products_df),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
