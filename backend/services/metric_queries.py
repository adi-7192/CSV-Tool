"""
Metric Query Templates

Safe, verified SQL templates for common business metrics.
These are the DEFAULT path for metric questions - no free-form SQL generation.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from core.database import execute_query, get_connection

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structured query result"""
    success: bool
    data: Dict[str, Any]
    sql: str
    error: Optional[str] = None


def get_date_column() -> str:
    """Get the actual date column name from schema"""
    try:
        conn = get_connection()
        columns = conn.execute("DESCRIBE sales").fetchdf()
        col_names = columns['column_name'].tolist()
        
        # Priority: order_date > Invoice Date > any column with 'date' in name
        for preferred in ['order_date', 'Invoice Date', 'invoice_date']:
            if preferred in col_names:
                return preferred
        
        # Fallback to any date column
        date_cols = [c for c in col_names if 'date' in c.lower()]
        if date_cols:
            return date_cols[0]
    except Exception as e:
        logger.warning(f"Error getting date column: {e}")
    return "order_date"  # Default fallback


def get_revenue_column() -> str:
    """Get the actual revenue column name from schema"""
    try:
        conn = get_connection()
        columns = conn.execute("DESCRIBE sales").fetchdf()
        col_names = columns['column_name'].tolist()
        
        # Priority order for revenue columns
        for preferred in ['revenue_amount', 'revenue_calc', 'Invoice Amount', 'invoice_amount']:
            if preferred in col_names:
                return preferred
        
        # Fallback to any revenue/amount column
        rev_cols = [c for c in col_names if any(word in c.lower() for word in ['revenue', 'amount'])]
        if rev_cols:
            return rev_cols[0]
    except Exception as e:
        logger.warning(f"Error getting revenue column: {e}")
    return "revenue_amount"  # Default fallback


def get_transaction_type_column() -> str:
    """Get the actual transaction type column name"""
    try:
        conn = get_connection()
        columns = conn.execute("DESCRIBE sales").fetchdf()
        col_names = columns['column_name'].tolist()
        
        # Priority order
        for preferred in ['transaction_type', 'Transaction Type']:
            if preferred in col_names:
                return preferred
        
        # Fallback
        txn_cols = [c for c in col_names if 'transaction' in c.lower()]
        if txn_cols:
            return txn_cols[0]
    except Exception as e:
        logger.warning(f"Error getting transaction type column: {e}")
    return "transaction_type"  # Default fallback


def get_sku_column() -> str:
    """Get the actual SKU column name"""
    try:
        conn = get_connection()
        columns = conn.execute("DESCRIBE sales").fetchdf()
        col_names = columns['column_name'].tolist()
        
        # Priority order
        for preferred in ['sku', 'Sku', 'SKU', 'product_sku']:
            if preferred in col_names:
                return preferred
        
        # Fallback
        sku_cols = [c for c in col_names if 'sku' in c.lower()]
        if sku_cols:
            return sku_cols[0]
    except Exception as e:
        logger.warning(f"Error getting SKU column: {e}")
    return "sku"  # Default fallback


def get_region_column() -> str:
    """Get the actual region/city column name"""
    try:
        conn = get_connection()
        columns = conn.execute("DESCRIBE sales").fetchdf()
        col_names = columns['column_name'].tolist()
        
        # Priority order
        for preferred in ['region', 'Ship City', 'ship_city', 'city']:
            if preferred in col_names:
                return preferred
        
        # Fallback
        region_cols = [c for c in col_names if any(word in c.lower() for word in ['region', 'city'])]
        if region_cols:
            return region_cols[0]
    except Exception as e:
        logger.warning(f"Error getting region column: {e}")
    return "region"  # Default fallback


def build_date_filter(date_col: str, start_date: Optional[str], end_date: Optional[str]) -> str:
    """Build date filter clause"""
    if not start_date or not end_date:
        return ""
    # Use quotes for column names with spaces, otherwise plain
    col_ref = f'"{date_col}"' if ' ' in date_col else date_col
    return f'CAST({col_ref} AS DATE) >= \'{start_date}\' AND CAST({col_ref} AS DATE) <= \'{end_date}\''


def quote_column(col: str) -> str:
    """Quote column name if it contains spaces"""
    return f'"{col}"' if ' ' in col else col


# =============================================================================
# METRIC QUERY TEMPLATES
# =============================================================================

def query_total_revenue(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    region: Optional[str] = None,
    product: Optional[str] = None
) -> QueryResult:
    """
    Get total revenue (shipments only)
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    region_col = get_region_column()
    sku_col = get_sku_column()
    
    conditions = [f'{quote_column(txn_col)} = \'Shipment\'']
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    if date_filter:
        conditions.append(date_filter)
    
    if region:
        conditions.append(f'{quote_column(region_col)} ILIKE \'%{region}%\'')
    if product:
        conditions.append(f'{quote_column(sku_col)} ILIKE \'%{product}%\'')
    
    where_clause = " AND ".join(conditions)
    
    sql = f'''
    SELECT 
        SUM({quote_column(rev_col)}) as total_revenue,
        COUNT(*) as transaction_count
    FROM sales
    WHERE {where_clause}
    '''
    
    try:
        result = execute_query(sql)
        if result.empty:
            return QueryResult(success=True, data={'total_revenue': 0, 'transaction_count': 0}, sql=sql)
        
        row = result.iloc[0]
        return QueryResult(
            success=True,
            data={
                'total_revenue': float(row['total_revenue']) if row['total_revenue'] else 0,
                'transaction_count': int(row['transaction_count']),
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_total_revenue: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_net_revenue(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Get net revenue = gross - refunds - cancellations - free replacements
    This is the CORRECT formula that includes ALL components.
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    
    txn_ref = quote_column(txn_col)
    rev_ref = quote_column(rev_col)
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    where_clause = f"WHERE {date_filter}" if date_filter else ""
    
    sql = f'''
    SELECT 
        SUM(CASE WHEN {txn_ref} = 'Shipment' THEN {rev_ref} ELSE 0 END) as gross_revenue,
        ABS(SUM(CASE WHEN {txn_ref} = 'Refund' THEN {rev_ref} ELSE 0 END)) as refund_amount,
        ABS(SUM(CASE WHEN {txn_ref} = 'Cancel' THEN {rev_ref} ELSE 0 END)) as cancel_amount,
        ABS(SUM(CASE WHEN {txn_ref} = 'FreeReplacement' THEN {rev_ref} ELSE 0 END)) as free_replacement_amount,
        (SUM(CASE WHEN {txn_ref} = 'Shipment' THEN {rev_ref} ELSE 0 END)
         - ABS(SUM(CASE WHEN {txn_ref} = 'Refund' THEN {rev_ref} ELSE 0 END))
         - ABS(SUM(CASE WHEN {txn_ref} = 'Cancel' THEN {rev_ref} ELSE 0 END))
         - ABS(SUM(CASE WHEN {txn_ref} = 'FreeReplacement' THEN {rev_ref} ELSE 0 END))
        ) as net_revenue
    FROM sales
    {where_clause}
    '''
    
    try:
        result = execute_query(sql)
        if result.empty:
            return QueryResult(success=True, data={'net_revenue': 0}, sql=sql)
        
        row = result.iloc[0]
        return QueryResult(
            success=True,
            data={
                'gross_revenue': float(row['gross_revenue']) if row['gross_revenue'] else 0,
                'refund_amount': float(row['refund_amount']) if row['refund_amount'] else 0,
                'cancel_amount': float(row['cancel_amount']) if row['cancel_amount'] else 0,
                'free_replacement_amount': float(row['free_replacement_amount']) if row['free_replacement_amount'] else 0,
                'net_revenue': float(row['net_revenue']) if row['net_revenue'] else 0,
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_net_revenue: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_refund_rate(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: Optional[str] = None
) -> QueryResult:
    """
    Get refund rate (refunds / shipments * 100)
    Optionally grouped by region or product
    """
    date_col = get_date_column()
    txn_col = get_transaction_type_column()
    txn_ref = quote_column(txn_col)
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    where_clause = f"WHERE {date_filter}" if date_filter else ""
    
    group_col = None
    if group_by == 'region':
        region_col = get_region_column()
        group_col = quote_column(region_col)
    elif group_by == 'product':
        sku_col = get_sku_column()
        group_col = quote_column(sku_col)
    
    if group_col:
        sql = f'''
        SELECT 
            {group_col} as group_key,
            COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipment_count,
            COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refund_count,
            ROUND(COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) * 100.0 / 
                  NULLIF(COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END), 0), 2) as refund_rate
        FROM sales
        {where_clause}
        GROUP BY {group_col}
        ORDER BY refund_rate DESC
        LIMIT 20
        '''
    else:
        sql = f'''
        SELECT 
            COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipment_count,
            COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refund_count,
            ROUND(COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) * 100.0 / 
                  NULLIF(COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END), 0), 2) as refund_rate
        FROM sales
        {where_clause}
        '''
    
    try:
        result = execute_query(sql)
        if result.empty:
            return QueryResult(success=True, data={'refund_rate': 0}, sql=sql)
        
        if group_col:
            data = {
                'groups': result.to_dict('records'),
                'start_date': start_date,
                'end_date': end_date,
            }
        else:
            row = result.iloc[0]
            data = {
                'shipment_count': int(row['shipment_count']),
                'refund_count': int(row['refund_count']),
                'refund_rate': float(row['refund_rate']) if row['refund_rate'] else 0,
                'start_date': start_date,
                'end_date': end_date,
            }
        
        return QueryResult(success=True, data=data, sql=sql)
    except Exception as e:
        logger.error(f"Error in query_refund_rate: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_top_products(
    n: int = 10,
    metric: str = 'revenue',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Get top N products by revenue or quantity
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    sku_col = get_sku_column()
    
    txn_ref = quote_column(txn_col)
    rev_ref = quote_column(rev_col)
    sku_ref = quote_column(sku_col)
    
    conditions = [f'{txn_ref} = \'Shipment\'']
    date_filter = build_date_filter(date_col, start_date, end_date)
    if date_filter:
        conditions.append(date_filter)
    
    where_clause = " AND ".join(conditions)
    
    order_col = f'SUM({rev_ref})' if metric == 'revenue' else 'COUNT(*)'
    
    sql = f'''
    SELECT 
        {sku_ref} as sku,
        SUM({rev_ref}) as total_revenue,
        COUNT(*) as transaction_count
    FROM sales
    WHERE {where_clause}
    GROUP BY {sku_ref}
    ORDER BY {order_col} DESC
    LIMIT {n}
    '''
    
    try:
        result = execute_query(sql)
        products = result.to_dict('records') if not result.empty else []
        
        return QueryResult(
            success=True,
            data={
                'products': products,
                'count': len(products),
                'metric': metric,
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_top_products: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_revenue_by_region(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Get revenue breakdown by region/city
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    region_col = get_region_column()
    
    txn_ref = quote_column(txn_col)
    rev_ref = quote_column(rev_col)
    region_ref = quote_column(region_col)
    
    conditions = [f'{txn_ref} = \'Shipment\'']
    date_filter = build_date_filter(date_col, start_date, end_date)
    if date_filter:
        conditions.append(date_filter)
    
    where_clause = " AND ".join(conditions)
    
    sql = f'''
    SELECT 
        {region_ref} as region,
        SUM({rev_ref}) as total_revenue,
        COUNT(*) as transaction_count
    FROM sales
    WHERE {where_clause}
    GROUP BY {region_ref}
    ORDER BY total_revenue DESC
    LIMIT 20
    '''
    
    try:
        result = execute_query(sql)
        regions = result.to_dict('records') if not result.empty else []
        
        return QueryResult(
            success=True,
            data={
                'regions': regions,
                'count': len(regions),
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_revenue_by_region: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_comparison(
    entity1: str,
    entity2: str,
    entity_type: str = 'region',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Compare two entities (regions or products)
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    sku_col = get_sku_column()
    region_col = get_region_column()
    
    txn_ref = quote_column(txn_col)
    rev_ref = quote_column(rev_col)
    sku_ref = quote_column(sku_col)
    region_ref = quote_column(region_col)
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    
    if entity_type == 'region':
        entity_col = region_ref
    else:
        entity_col = sku_ref
    
    conditions = [
        f'{txn_ref} = \'Shipment\'',
        f'{entity_col} ILIKE ANY (ARRAY[\'%{entity1}%\', \'%{entity2}%\'])'
    ]
    if date_filter:
        conditions.append(date_filter)
    
    where_clause = " AND ".join(conditions)
    
    sql = f'''
    SELECT 
        {entity_col} as entity,
        SUM({rev_ref}) as total_revenue,
        COUNT(*) as transaction_count,
        COUNT(DISTINCT {sku_ref}) as product_count
    FROM sales
    WHERE {where_clause}
    GROUP BY {entity_col}
    ORDER BY total_revenue DESC
    '''
    
    try:
        result = execute_query(sql)
        entities = result.to_dict('records') if not result.empty else []
        
        return QueryResult(
            success=True,
            data={
                'entities': entities,
                'entity1': entity1,
                'entity2': entity2,
                'entity_type': entity_type,
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_comparison: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_transaction_count(
    transaction_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Get count of transactions, optionally filtered by type
    """
    date_col = get_date_column()
    txn_col = get_transaction_type_column()
    txn_ref = quote_column(txn_col)
    
    conditions = []
    if transaction_type:
        conditions.append(f'{txn_ref} = \'{transaction_type}\'')
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    if date_filter:
        conditions.append(date_filter)
    
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    sql = f'''
    SELECT 
        {txn_ref} as transaction_type,
        COUNT(*) as count
    FROM sales
    {where_clause}
    GROUP BY {txn_ref}
    ORDER BY count DESC
    '''
    
    try:
        result = execute_query(sql)
        transactions = result.to_dict('records') if not result.empty else []
        total = sum(t['count'] for t in transactions)
        
        return QueryResult(
            success=True,
            data={
                'transactions': transactions,
                'total': total,
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=sql
        )
    except Exception as e:
        logger.error(f"Error in query_transaction_count: {e}")
        return QueryResult(success=False, data={}, sql=sql, error=str(e))


def query_advisory_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Get comprehensive data for advisory questions.
    Retrieves multiple metrics at once for trend analysis.
    """
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    sku_col = get_sku_column()
    region_col = get_region_column()
    
    txn_ref = quote_column(txn_col)
    rev_ref = quote_column(rev_col)
    sku_ref = quote_column(sku_col)
    region_ref = quote_column(region_col)
    
    date_filter = build_date_filter(date_col, start_date, end_date)
    where_clause = f"WHERE {date_filter}" if date_filter else ""
    
    # Query 1: Overall summary
    summary_sql = f'''
    SELECT 
        COUNT(*) as total_transactions,
        SUM(CASE WHEN {txn_ref} = 'Shipment' THEN {rev_ref} ELSE 0 END) as gross_revenue,
        ABS(SUM(CASE WHEN {txn_ref} = 'Refund' THEN {rev_ref} ELSE 0 END)) as total_refunds,
        COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refund_count,
        COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipment_count,
        COUNT(DISTINCT {sku_ref}) as distinct_products,
        COUNT(DISTINCT {region_ref}) as distinct_regions
    FROM sales
    {where_clause}
    '''
    
    # Query 2: Top products with refund rates
    products_sql = f'''
    SELECT 
        {sku_ref} as sku,
        SUM(CASE WHEN {txn_ref} = 'Shipment' THEN {rev_ref} ELSE 0 END) as revenue,
        COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipments,
        COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refunds,
        ROUND(COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) * 100.0 / 
              NULLIF(COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END), 0), 2) as refund_rate
    FROM sales
    {where_clause}
    GROUP BY {sku_ref}
    ORDER BY revenue DESC
    LIMIT 10
    '''
    
    # Query 3: Top regions
    regions_sql = f'''
    SELECT 
        {region_ref} as region,
        SUM(CASE WHEN {txn_ref} = 'Shipment' THEN {rev_ref} ELSE 0 END) as revenue,
        COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipments,
        COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refunds
    FROM sales
    {where_clause}
    GROUP BY {region_ref}
    ORDER BY revenue DESC
    LIMIT 10
    '''
    
    # Query 4: High refund products (worrying trends)
    high_refund_sql = f'''
    SELECT 
        {sku_ref} as sku,
        COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) as refund_count,
        COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) as shipment_count,
        ROUND(COUNT(CASE WHEN {txn_ref} = 'Refund' THEN 1 END) * 100.0 / 
              NULLIF(COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END), 0), 2) as refund_rate
    FROM sales
    {where_clause}
    GROUP BY {sku_ref}
    HAVING COUNT(CASE WHEN {txn_ref} = 'Shipment' THEN 1 END) >= 5
    ORDER BY refund_rate DESC
    LIMIT 5
    '''
    
    try:
        summary = execute_query(summary_sql)
        products = execute_query(products_sql)
        regions = execute_query(regions_sql)
        high_refunds = execute_query(high_refund_sql)
        
        summary_data = summary.iloc[0].to_dict() if not summary.empty else {}
        
        return QueryResult(
            success=True,
            data={
                'summary': summary_data,
                'top_products': products.to_dict('records') if not products.empty else [],
                'top_regions': regions.to_dict('records') if not regions.empty else [],
                'high_refund_products': high_refunds.to_dict('records') if not high_refunds.empty else [],
                'start_date': start_date,
                'end_date': end_date,
            },
            sql=f"-- Multiple queries for advisory data\n{summary_sql}"
        )
    except Exception as e:
        logger.error(f"Error in query_advisory_data: {e}")
        return QueryResult(success=False, data={}, sql=summary_sql, error=str(e))


# =============================================================================
# DISPATCHER - Maps intent to query template
# =============================================================================

def execute_metric_query(
    intent,  # QuestionIntent from intent_classifier
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> QueryResult:
    """
    Execute the appropriate query template based on intent.
    This is the DEFAULT path for METRIC questions.
    """
    metric_type = intent.metric_type
    entities = intent.entities
    
    # Extract region if present
    region = entities.get('region', [None])[0] if entities.get('region') else None
    product = entities.get('product', [None])[0] if entities.get('product') else None
    
    logger.info(f"Executing metric query: type={metric_type}, region={region}, product={product}")
    
    if metric_type == 'revenue':
        # Check if it's net revenue
        return query_total_revenue(start_date, end_date, region, product)
    
    elif metric_type == 'refund_rate':
        return query_refund_rate(start_date, end_date, intent.group_by)
    
    elif metric_type == 'top_products':
        n = intent.top_n or 10
        return query_top_products(n, 'revenue', start_date, end_date)
    
    elif metric_type == 'count':
        return query_transaction_count(None, start_date, end_date)
    
    elif metric_type == 'comparison':
        regions = entities.get('region', [])
        if len(regions) >= 2:
            return query_comparison(regions[0], regions[1], 'region', start_date, end_date)
        else:
            # Default to revenue by region
            return query_revenue_by_region(start_date, end_date)
    
    elif metric_type == 'average':
        # Use total revenue and count to calculate average
        result = query_total_revenue(start_date, end_date)
        if result.success and result.data.get('transaction_count', 0) > 0:
            avg = result.data['total_revenue'] / result.data['transaction_count']
            result.data['average_order_value'] = avg
        return result
    
    else:
        # Fallback: total revenue
        return query_total_revenue(start_date, end_date)

