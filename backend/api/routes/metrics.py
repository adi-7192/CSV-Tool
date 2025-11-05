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
    get_daily_trends,
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
    Get daily trend data with revenue, refunds, and orders grouped by date
    
    Returns:
        {
            "data": [
                {"date": "2025-07-01", "revenue": 245000.50, "refunds": 12300.00, "orders": 150},
                ...
            ],
            "revenue_trend": [
                {"date": "2025-07-01", "value": 245000.50},
                ...
            ],
            "refund_trend": [
                {"date": "2025-07-01", "value": 12300.00},
                ...
            ],
            "count": 92
        }
    """
    try:
        # If group_by is 'day', use the new comprehensive endpoint
        if group_by == 'day':
            trends = get_daily_trends(start_date, end_date)
            return trends
        else:
            # For week/month grouping, use the existing get_revenue_trend
            trend_df = get_revenue_trend(start_date, end_date, group_by)
            
            # Convert DataFrame to list of dicts for JSON response
            trend_data = trend_df.to_dict('records')
            
            # Format dates as strings
            for item in trend_data:
                if 'date' in item:
                    item['date'] = str(item['date']).split(' ')[0]  # Extract date part
                    # Add missing fields with defaults
                    if 'revenue' not in item:
                        item['revenue'] = item.get('value', 0)
                    if 'refunds' not in item:
                        item['refunds'] = 0.0
                    if 'orders' not in item:
                        item['orders'] = 0
            
            # Create trend arrays
            revenue_trend = [
                {'date': str(item.get('date', '')), 'value': float(item.get('revenue', item.get('value', 0)))}
                for item in trend_data
            ]
            refund_trend = [
                {'date': str(item.get('date', '')), 'value': float(item.get('refunds', 0))}
                for item in trend_data
            ]
            
            return {
                "data": trend_data,
                "revenue_trend": revenue_trend,
                "refund_trend": refund_trend,
                "count": len(trend_data),
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-products")
async def get_top_products_endpoint(
    limit: int = Query(50, ge=1, le=100, description="Number of products"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    metric: str = Query('revenue', description="Sort metric (revenue)"),
):
    """
    Get top products by SKU with comprehensive metrics
    
    Returns:
        {
            "data": [
                {
                    "sku": "SKU-001",
                    "asin": "B001234567",
                    "units_sold": 1245,
                    "revenue": 1956000.50,
                    "refund_ratio": 2.3,
                    "rating": 4.5,
                    "trend": 0
                },
                ...
            ],
            "count": 50
        }
    """
    try:
        products_df = get_top_products(limit, start_date, end_date, metric)
        
        # Convert DataFrame to list of dicts
        products_list = products_df.to_dict('records')
        
        # Ensure all numeric values are properly formatted
        for product in products_list:
            product['units_sold'] = int(product.get('units_sold', 0))
            product['revenue'] = float(product.get('revenue', 0))
            product['refund_ratio'] = float(product.get('refund_ratio', 0))
            product['rating'] = float(product.get('rating', 0))
            product['trend'] = float(product.get('trend', 0))
            # Ensure asin is a string
            product['asin'] = str(product.get('asin', ''))
            # Ensure sku is a string
            product['sku'] = str(product.get('sku', ''))
        
        return {
            "data": products_list,
            "count": len(products_list),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
