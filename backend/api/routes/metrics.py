"""
Metrics API Endpoints - Serve KPIs to frontend with tenant isolation
"""
from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
from datetime import datetime, timedelta
import logging

from services.metrics_service import (
    calculate_metrics,
    get_revenue_trend,
    get_top_products,
    get_daily_trends,
    get_revenue_by_city,
    get_movers_decliners,
    get_skus_by_city,
    get_top_products_performance,
    get_refunds_data,
    get_cancellations_data,
    get_free_replacements_data,
)
from utils.error_handler import format_error_response, log_error
from utils.validators import (
    validate_date_range,
    validate_date,
    validate_transaction_type,
    validate_group_by,
    validate_limit,
)
from utils.sanitizers import (
    sanitize_string,
    sanitize_date_string,
    sanitize_transaction_type,
    sanitize_city_name,
)
from utils.logger import api_logger, log_api_request, log_api_response
from utils.tenant_filter import get_tenant_filter_sql
from api.deps.auth_deps import get_current_user
from models.user import UserInDB

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("")
@router.get("/")
async def get_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    transaction_type: Optional[str] = Query(None, description="Filter by transaction type"),
    source_file: Optional[str] = Query(None, description="Filter by source file"),
    current_user: UserInDB = Depends(get_current_user),
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
    import time
    start_time = time.time()
    
    log_api_request(
        api_logger,
        'GET',
        '/api/metrics/',
        {
            'start_date': start_date,
            'end_date': end_date,
            'transaction_type': transaction_type,
            'source_file': source_file
        }
    )
    
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        transaction_type = sanitize_transaction_type(transaction_type)
        source_file = sanitize_string(source_file)
        
        # Default to last 30 days if no dates provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            # Validate date range
            validate_date_range(start_date, end_date)
        
        # Validate transaction type if provided
        if transaction_type:
            validate_transaction_type(transaction_type)
        
        # TENANT ISOLATION: Get tenant_id from authenticated user
        tenant_id = current_user.tenant_id
        if not tenant_id:
            tenant_id = str(current_user.id)
        
        # Check if sales table exists and has data FOR THIS TENANT
        from core.database import table_exists, execute_query
        has_data = False
        if table_exists('sales'):
            tenant_filter = get_tenant_filter_sql(tenant_id)
            count_result = execute_query(f"SELECT COUNT(*) as count FROM sales WHERE {tenant_filter}")
            if not count_result.empty and count_result.iloc[0]['count'] > 0:
                has_data = True
        
        # Pass tenant_id to calculate_metrics for data isolation
        metrics = calculate_metrics(start_date, end_date, transaction_type, source_file, tenant_id=tenant_id)
        
        duration_ms = (time.time() - start_time) * 1000
        log_api_response(api_logger, 'GET', '/api/metrics/', 200, duration_ms)
        
        return {
            "data": metrics,
            "has_data": has_data,
            "period": {
                "start_date": start_date,
                "end_date": end_date,
            }
        }
    
    except ValueError as e:
        duration_ms = (time.time() - start_time) * 1000
        log_api_response(api_logger, 'GET', '/api/metrics/', 400, duration_ms)
        log_error(e, 'get_metrics', {'start_date': start_date, 'end_date': end_date, 'transaction_type': transaction_type})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_api_response(api_logger, 'GET', '/api/metrics/', 500, duration_ms)
        log_error(e, 'get_metrics', {'start_date': start_date, 'end_date': end_date, 'transaction_type': transaction_type})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch metrics. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


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
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        group_by = sanitize_string(group_by, max_length=10) or 'day'
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_group_by(group_by)
        
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
    
    except ValueError as e:
        log_error(e, 'get_trend', {'start_date': start_date, 'end_date': end_date, 'group_by': group_by})
        error_response = format_error_response(e, status_code=400, user_message=f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_trend', {'start_date': start_date, 'end_date': end_date, 'group_by': group_by})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch trend data. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


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


@router.get("/revenue-by-city")
async def get_revenue_by_city_endpoint(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50, description="Number of cities to return"),
):
    """
    Get top cities by total revenue for a specific date range.
    
    Returns:
        {
            "data": [
                {"city": "Bangalore", "revenue": 1000000},
                {"city": "Mumbai", "revenue": 850000},
                ...
            ],
            "count": 10,
            "total_revenue": 5000000
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Default to last 30 days if no dates provided
        if not start_date or not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        else:
            # Validate date range
            validate_date_range(start_date, end_date)
        
        # Validate limit
        validate_limit(limit)
        
        result = get_revenue_by_city(start_date, end_date, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_revenue_by_city', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_revenue_by_city', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch revenue by city. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/revenue-by-city/skus/{city}")
async def get_city_skus_endpoint(
    city: str,
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50, description="Number of SKUs to return"),
):
    """
    Get top SKUs for a specific city.
    
    Returns:
        {
            "city": "Bangalore",
            "data": [
                {"sku": "SKU-001", "asin": "B01234567", "units": 100, "revenue": 50000},
                ...
            ],
            "count": 10
        }
    """
    try:
        # Sanitize inputs
        city = sanitize_city_name(city)
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Validate city name
        from utils.validators import validate_city_name
        validate_city_name(city)
        
        # Note: get_skus_by_city doesn't support date filtering (uses all-time data)
        # Date parameters are accepted but not used by the function
        # Validate date range if provided (for future compatibility)
        if start_date and end_date:
            validate_date_range(start_date, end_date)
        
        # Validate limit
        validate_limit(limit)
        
        result = get_skus_by_city(city, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_city_skus', {'city': city, 'start_date': start_date, 'end_date': end_date})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_city_skus', {'city': city, 'start_date': start_date, 'end_date': end_date})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch city SKUs. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/movers-decliners")
async def get_movers_decliners_endpoint(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=50, description="Number of SKUs per category"),
):
    """
    Get movers (fast growing) and decliners (declining) SKUs using adaptive moving average comparison.
    
    Automatically determines granularity based on date range:
    - 7-13 days: Daily comparison (last day vs daily average)
    - 14-89 days: Weekly comparison (last week vs weekly average)
    - 90+ days: Monthly comparison (last month vs monthly average)
    
    - Decliners: growth % <= -30%
    - Fast Movers: growth % >= +30%
    
    Returns:
        {
            "movers": [
                {"sku": "SKU-001", "revenue": 100000, "wow_change": 35.5},
                ...
            ],
            "decliners": [
                {"sku": "SKU-002", "revenue": 50000, "wow_change": -45.2},
                ...
            ],
            "label": "Last Week vs Weekly Avg",
            "granularity": "weekly"
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_limit(limit)
        
        result = get_movers_decliners(start_date, end_date, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_movers_decliners', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_movers_decliners', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch movers/decliners. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/top-products-performance")
async def get_top_products_performance_endpoint(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    view_type: str = Query('monthly', description="View type: 'monthly' or 'quarterly'"),
    limit: int = Query(10, ge=1, le=100, description="Number of products to return"),
):
    """
    Get top products performance tracker with period-by-period breakdown.
    
    Shows top products by total volume with monthly or quarterly performance tracking.
    
    Returns:
        {
            "products": [
                {
                    "sku": "GP47_NEW",
                    "periods": [145, 150, 139],
                    "growth_rates": [None, 3.4, -7.3],
                    "total_volume": 434
                },
                ...
            ],
            "period_labels": ["Jul", "Aug", "Sep"],
            "view_type": "monthly"
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        view_type = sanitize_string(view_type, max_length=20) or 'monthly'
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_limit(limit)
        
        if view_type not in ['monthly', 'quarterly']:
            raise ValueError("view_type must be 'monthly' or 'quarterly'")
        
        result = get_top_products_performance(start_date, end_date, view_type, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_top_products_performance', {'start_date': start_date, 'end_date': end_date, 'view_type': view_type, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_top_products_performance', {'start_date': start_date, 'end_date': end_date, 'view_type': view_type, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch top products performance. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


# ============================================================================
# PRODUCT QUALITY ISSUES ENDPOINTS
# ============================================================================

@router.get("/quality-issues/refunds")
async def get_refunds_data_endpoint(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=100, description="Number of products to return"),
):
    """
    Get refunds data for Product Quality Issues dashboard.
    
    Returns top products by refund percentage with lost revenue.
    
    Returns:
        {
            "data": [
                {
                    "sku": "SKU-001",
                    "units_sold": 100,
                    "refunds": 5,
                    "refund_percentage": 5.0,
                    "lost_revenue": 5000.0
                },
                ...
            ]
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_limit(limit)
        
        result = get_refunds_data(start_date, end_date, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_refunds_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_refunds_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch refunds data. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/quality-issues/cancellations")
async def get_cancellations_data_endpoint(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=100, description="Number of products to return"),
):
    """
    Get cancellations data for Product Quality Issues dashboard.
    
    Returns top products by cancellation percentage.
    
    Returns:
        {
            "data": [
                {
                    "sku": "SKU-001",
                    "units_ordered": 100,
                    "cancelled": 10,
                    "cancel_percentage": 10.0
                },
                ...
            ]
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_limit(limit)
        
        result = get_cancellations_data(start_date, end_date, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_cancellations_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_cancellations_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch cancellations data. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


@router.get("/quality-issues/replacements")
async def get_free_replacements_data_endpoint(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    limit: int = Query(10, ge=1, le=100, description="Number of products to return"),
):
    """
    Get free replacements data for Product Quality Issues dashboard.
    
    Returns top products by total replacement loss.
    
    Returns:
        {
            "data": [
                {
                    "sku": "SKU-001",
                    "replacements": 5,
                    "total_loss": 10000.0
                },
                ...
            ]
        }
    """
    try:
        # Sanitize inputs
        start_date = sanitize_date_string(start_date)
        end_date = sanitize_date_string(end_date)
        
        # Validate inputs
        validate_date_range(start_date, end_date)
        validate_limit(limit)
        
        result = get_free_replacements_data(start_date, end_date, limit)
        return result
    except ValueError as e:
        log_error(e, 'get_free_replacements_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=400, user_message=str(e))
        raise HTTPException(status_code=400, detail=error_response)
    except Exception as e:
        log_error(e, 'get_free_replacements_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        error_response = format_error_response(e, status_code=500, user_message="Failed to fetch replacements data. Please try again later.")
        raise HTTPException(status_code=500, detail=error_response)


