"""
Metrics Registry - Maps user questions to existing backend functions

This registry connects natural language questions to the 16+ metric functions
already implemented in metrics_service.py. The AI chat will check this registry
BEFORE attempting SQL generation, ensuring it uses proven, tested calculations.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MetricFunction:
    """Metadata about a metric function"""
    function_name: str  # Name of function in metrics_service
    keywords: List[str]  # Keywords that match this function
    description: str  # What this function does
    required_params: List[str]  # Required parameters (e.g., ['start_date', 'end_date'])
    optional_params: List[str]  # Optional parameters
    return_type: str  # What it returns (e.g., 'movers_decliners', 'revenue_breakdown')


# Registry mapping keywords to functions
METRICS_REGISTRY: Dict[str, MetricFunction] = {
    # Declining/Growing Products
    "declining_products": MetricFunction(
        function_name="get_movers_decliners",
        keywords=[
            "declining", "decline", "falling", "dropping", "decreasing", "losing",
            "worst performing", "worst products", "worst skus", "underperforming",
            "products are declining", "skus are declining", "which products are declining",
            "declining skus", "declining products", "products declining",
            "falling sales", "dropping sales", "decreasing sales"
        ],
        description="Find products with declining sales (30%+ drop vs baseline)",
        required_params=["start_date", "end_date"],
        optional_params=["limit"],
        return_type="movers_decliners"
    ),
    
    "growing_products": MetricFunction(
        function_name="get_movers_decliners",
        keywords=[
            "growing", "grow", "rising", "increasing", "improving", "best performing",
            "best products", "best skus", "top performers", "movers",
            "products are growing", "skus are growing", "which products are growing",
            "growing skus", "growing products", "products growing",
            "rising sales", "increasing sales", "improving sales"
        ],
        description="Find products with growing sales (30%+ growth vs baseline)",
        required_params=["start_date", "end_date"],
        optional_params=["limit"],
        return_type="movers_decliners"
    ),
    
    # Core Metrics
    "net_revenue": MetricFunction(
        function_name="calculate_metrics",
        keywords=[
            "net revenue", "net profit", "after refunds", "net earnings",
            "net sales", "net income", "net amount", "profit after refunds",
            "revenue after deductions", "actual revenue", "true revenue"
        ],
        description="Calculate net revenue after refunds, cancellations, and free replacements",
        required_params=["start_date", "end_date"],
        optional_params=["transaction_type", "source_file"],
        return_type="metrics_breakdown"
    ),
    
    "gross_revenue": MetricFunction(
        function_name="calculate_metrics",
        keywords=[
            "gross revenue", "total revenue", "revenue", "sales", "total sales",
            "what is my revenue", "how much revenue", "revenue amount"
        ],
        description="Calculate gross revenue from shipments",
        required_params=["start_date", "end_date"],
        optional_params=["transaction_type", "source_file"],
        return_type="metrics_breakdown"
    ),
    
    "refund_rate": MetricFunction(
        function_name="get_refunds_data",
        keywords=[
            "refund rate", "refund percentage", "return rate", "refund analysis",
            "refunds", "refund data", "refund breakdown", "refund statistics"
        ],
        description="Get refund rate and refund analysis",
        required_params=["start_date", "end_date"],
        optional_params=[],
        return_type="refunds_data"
    ),
    
    "cancellations": MetricFunction(
        function_name="get_cancellations_data",
        keywords=[
            "cancellations", "cancellation rate", "cancelled orders", "cancel data",
            "cancellation analysis", "cancelled", "cancels"
        ],
        description="Get cancellation data and analysis",
        required_params=["start_date", "end_date"],
        optional_params=[],
        return_type="cancellations_data"
    ),
    
    "free_replacements": MetricFunction(
        function_name="get_free_replacements_data",
        keywords=[
            "free replacements", "free replacement", "replacement cost",
            "free replacement cost", "replacements"
        ],
        description="Get free replacement cost analysis",
        required_params=["start_date", "end_date"],
        optional_params=[],
        return_type="free_replacements_data"
    ),
    
    # Top Products
    "top_products": MetricFunction(
        function_name="get_top_products",
        keywords=[
            "top products", "top skus", "best products", "best selling",
            "highest revenue products", "top selling products", "top n products",
            "top 10 products", "top 5 products", "top 20 products"
        ],
        description="Get top products by revenue or quantity",
        required_params=["start_date", "end_date"],
        optional_params=["limit", "metric"],
        return_type="top_products"
    ),
    
    "top_products_performance": MetricFunction(
        function_name="get_top_products_performance",
        keywords=[
            "product performance", "products performance", "sku performance",
            "how are products performing", "product metrics", "product stats"
        ],
        description="Get detailed performance metrics for top products",
        required_params=["start_date", "end_date"],
        optional_params=["limit"],
        return_type="products_performance"
    ),
    
    # Regional Analysis
    "revenue_by_city": MetricFunction(
        function_name="get_revenue_by_city",
        keywords=[
            "revenue by city", "revenue by region", "city revenue", "regional revenue",
            "which city", "best city", "top city", "city breakdown",
            "revenue breakdown by city", "sales by city", "sales by region"
        ],
        description="Get revenue breakdown by city/region",
        required_params=["start_date", "end_date"],
        optional_params=["limit"],
        return_type="revenue_by_city"
    ),
    
    "skus_by_city": MetricFunction(
        function_name="get_skus_by_city",
        keywords=[
            "products in", "skus in", "what sells in", "products sold in",
            "skus by city", "products by city", "city products"
        ],
        description="Get products/SKUs sold in a specific city",
        required_params=["city"],
        optional_params=["limit"],
        return_type="skus_by_city"
    ),
    
    "skus_by_region": MetricFunction(
        function_name="get_skus_by_region",
        keywords=[
            "products in region", "skus in region", "regional products",
            "products by region", "skus by region"
        ],
        description="Get products/SKUs by region",
        required_params=["start_date", "end_date"],
        optional_params=["limit"],
        return_type="skus_by_region"
    ),
    
    # Trends
    "daily_trends": MetricFunction(
        function_name="get_daily_trends",
        keywords=[
            "daily trend", "daily trends", "day by day", "daily sales",
            "daily revenue", "trend over days", "day trend"
        ],
        description="Get daily revenue and order trends",
        required_params=["start_date", "end_date"],
        optional_params=[],
        return_type="daily_trends"
    ),
    
    "revenue_trend": MetricFunction(
        function_name="get_revenue_trend",
        keywords=[
            "revenue trend", "sales trend", "trend", "over time",
            "revenue over time", "sales over time", "trending"
        ],
        description="Get revenue trend over time",
        required_params=["start_date", "end_date"],
        optional_params=[],
        return_type="revenue_trend"
    ),
    
    "compare_periods": MetricFunction(
        function_name="compare_periods",
        keywords=[
            "compare", "comparison", "vs", "versus", "difference between",
            "month over month", "month to month", "period comparison",
            "compare june", "compare july", "compare august", "compare september"
        ],
        description="Compare two time periods (e.g., June vs July)",
        required_params=["period1_start", "period1_end", "period2_start", "period2_end"],
        optional_params=["metrics"],
        return_type="period_comparison"
    ),
}


def find_matching_function(question: str) -> Optional[MetricFunction]:
    """
    Find the best matching metric function for a user question
    
    Args:
        question: User's natural language question
        
    Returns:
        MetricFunction if match found, None otherwise
    """
    question_lower = question.lower().strip()
    
    # Validate question first - don't match invalid questions
    from services.intent_classifier import is_valid_question
    is_valid, _ = is_valid_question(question)
    if not is_valid:
        logger.info(f"Skipping metrics registry match for invalid question: '{question}'")
        return None
    
    # Score each function by keyword matches
    matches = []
    for key, metric_func in METRICS_REGISTRY.items():
        score = 0
        matched_keywords = []
        
        for keyword in metric_func.keywords:
            if keyword in question_lower:
                score += len(keyword)  # Longer keywords = more specific = higher score
                matched_keywords.append(keyword)
        
        if score > 0:
            matches.append({
                'key': key,
                'function': metric_func,
                'score': score,
                'matched_keywords': matched_keywords
            })
    
    if not matches:
        return None
    
    # Return highest scoring match
    best_match = max(matches, key=lambda x: x['score'])
    logger.info(f"Matched function '{best_match['function'].function_name}' "
                f"(score: {best_match['score']}, keywords: {best_match['matched_keywords'][:3]})")
    
    return best_match['function']


def get_function_call_params(
    metric_func: MetricFunction,
    question: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Build parameters for calling a metric function
    
    Args:
        metric_func: The metric function to call
        question: Original user question (for extracting entities)
        start_date: Start date filter
        end_date: End date filter
        **kwargs: Additional context
        
    Returns:
        Dictionary of parameters to pass to the function, or None if required params missing
    """
    params = {}
    
    # Add required params
    if 'start_date' in metric_func.required_params:
        if start_date:
            params['start_date'] = start_date
        else:
            # Try to get default date range from database
            try:
                from core.database import execute_query
                date_range = execute_query("SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM sales")
                if not date_range.empty and date_range.iloc[0]['min_date']:
                    params['start_date'] = str(date_range.iloc[0]['min_date'])[:10]  # YYYY-MM-DD
                    logger.info(f"Using database min date: {params['start_date']}")
                else:
                    logger.warning(f"Required param 'start_date' missing for {metric_func.function_name}")
                    return None  # Cannot proceed without required param
            except Exception as e:
                logger.error(f"Error getting default date range: {e}")
                return None
    
    if 'end_date' in metric_func.required_params:
        if end_date:
            params['end_date'] = end_date
        else:
            # Try to get default date range from database
            try:
                from core.database import execute_query
                date_range = execute_query("SELECT MIN(order_date) as min_date, MAX(order_date) as max_date FROM sales")
                if not date_range.empty and date_range.iloc[0]['max_date']:
                    params['end_date'] = str(date_range.iloc[0]['max_date'])[:10]  # YYYY-MM-DD
                    logger.info(f"Using database max date: {params['end_date']}")
                else:
                    logger.warning(f"Required param 'end_date' missing for {metric_func.function_name}")
                    return None  # Cannot proceed without required param
            except Exception as e:
                logger.error(f"Error getting default date range: {e}")
                return None
    
    # Extract city from question if needed
    if 'city' in metric_func.required_params:
        question_lower = question.lower()
        # Common Indian cities
        cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'hyderabad', 'kolkata', 'pune', 'ahmedabad']
        for city in cities:
            if city in question_lower:
                params['city'] = city.capitalize()
                break
        
        if 'city' not in params:
            logger.warning(f"Required param 'city' not found in question for {metric_func.function_name}")
    
    # Extract limit from question if needed
    if 'limit' in metric_func.optional_params:
        import re
        limit_match = re.search(r'\btop\s+(\d+)\b', question.lower())
        if limit_match:
            params['limit'] = int(limit_match.group(1))
        else:
            params['limit'] = 10  # Default
    
    # Add any additional kwargs
    params.update(kwargs)
    
    return params


def call_metric_function(
    function_name: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call a metric function from metrics_service
    
    Args:
        function_name: Name of function to call
        params: Parameters to pass
        
    Returns:
        Result dictionary from the function
    """
    try:
        from services import metrics_service
        
        if not hasattr(metrics_service, function_name):
            logger.error(f"Function '{function_name}' not found in metrics_service")
            return {'error': f"Function {function_name} not found"}
        
        func = getattr(metrics_service, function_name)
        result = func(**params)
        
        logger.info(f"Successfully called {function_name} with params: {params}")
        return result
        
    except Exception as e:
        logger.error(f"Error calling {function_name}: {e}", exc_info=True)
        return {'error': str(e)}


def get_all_registered_functions() -> List[str]:
    """Get list of all registered function names"""
    return list(set(func.function_name for func in METRICS_REGISTRY.values()))

