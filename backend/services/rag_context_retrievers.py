"""
RAG Context Retrievers

Retrieves relevant data context from the database for RAG (Retrieval-Augmented Generation).
Provides statistical summaries, sample data, column statistics, temporal context, and entity context.

IMPORTANT: Uses dynamic column detection to avoid hardcoded column names.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import pandas as pd

from core.database import execute_query, get_connection, table_exists
from services.sql_validator import get_date_column, get_revenue_column, get_transaction_type_column

logger = logging.getLogger(__name__)


def get_statistical_summaries() -> Dict[str, Any]:
    """
    Get overall statistical summaries of the data
    
    Returns:
        Dictionary with statistics including total records, date ranges, etc.
    """
    if not table_exists('sales'):
        return {
            'total_records': 0,
            'date_range': None,
            'has_data': False
        }
    
    try:
        # Get actual column names dynamically
        date_col = get_date_column()
        rev_col = get_revenue_column()
        txn_col = get_transaction_type_column()
        
        # Get total record count
        total_query = "SELECT COUNT(*) as total FROM sales"
        total_df = execute_query(total_query)
        total_records = int(total_df.iloc[0]['total']) if not total_df.empty else 0
        
        if total_records == 0:
            return {
                'total_records': 0,
                'date_range': None,
                'has_data': False
            }
        
        # Get date range with dynamic column name
        date_range_query = f"""
        SELECT 
            MIN(CAST("{date_col}" AS DATE)) as min_date,
            MAX(CAST("{date_col}" AS DATE)) as max_date
        FROM sales
        WHERE "{date_col}" IS NOT NULL
        """
        
        try:
            date_range_df = execute_query(date_range_query)
        except Exception as e:
            logger.warning(f"Date range query failed: {e}")
            date_range_df = pd.DataFrame()
        
        date_range = None
        if not date_range_df.empty:
            min_date = date_range_df.iloc[0]['min_date']
            max_date = date_range_df.iloc[0]['max_date']
            if min_date and max_date:
                date_range = {
                    'min': str(min_date),
                    'max': str(max_date)
                }
        
        # Get transaction type distribution with dynamic column names
        txn_type_query = f"""
        SELECT 
            "{txn_col}" as transaction_type,
            COUNT(*) as count,
            SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as shipment_revenue
        FROM sales
        GROUP BY "{txn_col}"
        ORDER BY count DESC
        """
        
        try:
            txn_type_df = execute_query(txn_type_query)
            transaction_types = {}
            if not txn_type_df.empty:
                for _, row in txn_type_df.iterrows():
                    transaction_types[row['transaction_type']] = {
                        'count': int(row['count']),
                        'revenue': float(row.get('shipment_revenue', 0)) if row['transaction_type'] == 'Shipment' else None
                    }
        except Exception as e:
            logger.warning(f"Could not get transaction type distribution: {e}")
            transaction_types = {}
        
        return {
            'total_records': total_records,
            'date_range': date_range,
            'transaction_types': transaction_types,
            'has_data': True
        }
    
    except Exception as e:
        logger.error(f"Error getting statistical summaries: {e}")
        return {
            'total_records': 0,
            'date_range': None,
            'has_data': False,
            'error': str(e)
        }


def get_sample_data(question: str, limit: int = 5, relevant_columns: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """
    Get sample data rows relevant to the question
    
    Args:
        question: User question (for keyword matching)
        limit: Maximum number of sample rows to return
        relevant_columns: Set of column names that are relevant
        
    Returns:
        List of sample row dictionaries
    """
    if not table_exists('sales'):
        return []
    
    try:
        # Get actual column names
        date_col = get_date_column()
        txn_col = get_transaction_type_column()
        
        question_lower = question.lower()
        
        # Build WHERE clause based on question keywords
        where_clauses = []
        
        # If question mentions specific transaction types, filter by them
        if 'refund' in question_lower:
            where_clauses.append(f'"{txn_col}" = \'Refund\'')
        elif 'shipment' in question_lower or 'order' in question_lower:
            where_clauses.append(f'"{txn_col}" = \'Shipment\'')
        elif 'cancel' in question_lower:
            where_clauses.append(f'"{txn_col}" = \'Cancel\'')
        
        where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # If question mentions dates, try to get recent data
        if any(word in question_lower for word in ['recent', 'last', 'latest']):
            query = f'''SELECT * FROM sales {where_clause} 
                       ORDER BY CAST("{date_col}" AS DATE) DESC 
                       LIMIT {limit}'''
        else:
            # Just get random sample
            query = f"SELECT * FROM sales {where_clause} LIMIT {limit}"
        
        sample_df = execute_query(query)
        
        # Convert to list of dictionaries
        if sample_df.empty:
            return []
        
        # Limit columns if specified
        if relevant_columns:
            available_columns = [col for col in sample_df.columns if col in relevant_columns]
            if available_columns:
                sample_df = sample_df[available_columns]
        
        # Convert to list of dicts, handling NaN values
        samples = []
        for _, row in sample_df.head(limit).iterrows():
            sample_dict = {}
            for col in sample_df.columns:
                value = row[col]
                if pd.isna(value):
                    sample_dict[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    sample_dict[col] = str(value)
                else:
                    sample_dict[col] = value
            samples.append(sample_dict)
        
        return samples
    
    except Exception as e:
        logger.error(f"Error getting sample data: {e}")
        return []


def get_column_statistics(columns: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Get column-level statistics
    
    Args:
        columns: Set of column names to get statistics for (None = all relevant columns)
        
    Returns:
        Dictionary with column statistics
    """
    if not table_exists('sales'):
        return {}
    
    try:
        # Get actual column names
        date_col = get_date_column()
        rev_col = get_revenue_column()
        txn_col = get_transaction_type_column()
        
        stats = {}
        
        # Default columns to analyze if not specified
        if columns is None:
            columns = {rev_col, date_col, txn_col, 'sku', 'region', 'quantity'}
        
        # Get distinct values for categorical columns
        if txn_col in columns or 'transaction_type' in columns:
            try:
                txn_query = f'SELECT DISTINCT "{txn_col}" as txn_type FROM sales WHERE "{txn_col}" IS NOT NULL'
                txn_df = execute_query(txn_query)
                stats['transaction_type'] = {
                    'distinct_values': txn_df['txn_type'].tolist() if not txn_df.empty else [],
                    'type': 'categorical'
                }
            except Exception as e:
                logger.warning(f"Could not get transaction_type stats: {e}")
        
        # Get numeric statistics for revenue
        if rev_col in columns or 'revenue_amount' in columns:
            try:
                revenue_query = f"""
                SELECT 
                    MIN("{rev_col}") as min_revenue,
                    MAX("{rev_col}") as max_revenue,
                    AVG("{rev_col}") as avg_revenue,
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as total_shipment_revenue
                FROM sales
                WHERE "{rev_col}" IS NOT NULL
                """
                revenue_df = execute_query(revenue_query)
                if not revenue_df.empty:
                    stats['revenue_amount'] = {
                        'min': float(revenue_df.iloc[0]['min_revenue']) if revenue_df.iloc[0]['min_revenue'] is not None else None,
                        'max': float(revenue_df.iloc[0]['max_revenue']) if revenue_df.iloc[0]['max_revenue'] is not None else None,
                        'avg': float(revenue_df.iloc[0]['avg_revenue']) if revenue_df.iloc[0]['avg_revenue'] is not None else None,
                        'total_shipment_revenue': float(revenue_df.iloc[0]['total_shipment_revenue']) if revenue_df.iloc[0]['total_shipment_revenue'] is not None else None,
                        'type': 'numeric'
                    }
            except Exception as e:
                logger.warning(f"Could not get revenue_amount stats: {e}")
        
        # Get top regions
        if 'region' in columns:
            try:
                region_query = """
                SELECT region, COUNT(*) as count
                FROM sales
                WHERE region IS NOT NULL
                GROUP BY region
                ORDER BY count DESC
                LIMIT 10
                """
                region_df = execute_query(region_query)
                if not region_df.empty:
                    stats['region'] = {
                        'top_values': [
                            {'value': row['region'], 'count': int(row['count'])}
                            for _, row in region_df.iterrows()
                        ],
                        'type': 'categorical'
                    }
            except Exception as e:
                logger.warning(f"Could not get region stats: {e}")
        
        # Get top SKUs
        if 'sku' in columns:
            try:
                sku_query = """
                SELECT sku, COUNT(*) as count
                FROM sales
                WHERE sku IS NOT NULL
                GROUP BY sku
                ORDER BY count DESC
                LIMIT 10
                """
                sku_df = execute_query(sku_query)
                if not sku_df.empty:
                    stats['sku'] = {
                        'top_values': [
                            {'value': row['sku'], 'count': int(row['count'])}
                            for _, row in sku_df.iterrows()
                        ],
                        'type': 'categorical'
                    }
            except Exception as e:
                logger.warning(f"Could not get SKU stats: {e}")
        
        return stats
    
    except Exception as e:
        logger.error(f"Error getting column statistics: {e}")
        return {}


def get_temporal_context(date_range: Optional[Dict[str, Optional[str]]] = None, 
                        month_mentions: Optional[List[int]] = None,
                        year_mentions: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Get temporal context (date-related information)
    
    Args:
        date_range: Optional date range dict with 'start' and 'end'
        month_mentions: List of month numbers mentioned (1-12)
        year_mentions: List of years mentioned
        
    Returns:
        Dictionary with temporal context
    """
    if not table_exists('sales'):
        return {}
    
    try:
        # Get actual column names
        date_col = get_date_column()
        rev_col = get_revenue_column()
        txn_col = get_transaction_type_column()
        
        context = {}
        
        # Get monthly aggregations if month mentioned
        if month_mentions:
            for month in month_mentions[:1]:  # Limit to first month
                year = year_mentions[0] if year_mentions else datetime.now().year
                
                try:
                    monthly_query = f"""
                    SELECT 
                        "{txn_col}" as transaction_type,
                        COUNT(*) as count,
                        SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as revenue
                    FROM sales
                    WHERE DATE_PART('month', CAST("{date_col}" AS DATE)) = {month}
                      AND DATE_PART('year', CAST("{date_col}" AS DATE)) = {year}
                    GROUP BY "{txn_col}"
                    """
                    monthly_df = execute_query(monthly_query)
                    
                    if not monthly_df.empty:
                        context[f'month_{month}_{year}'] = {
                            'month': month,
                            'year': year,
                            'transactions': {
                                row['transaction_type']: {
                                    'count': int(row['count']),
                                    'revenue': float(row['revenue']) if row['transaction_type'] == 'Shipment' else None
                                }
                                for _, row in monthly_df.iterrows()
                            }
                        }
                except Exception as e:
                    logger.warning(f"Could not get monthly context for month {month}: {e}")
        
        # Get date range summary if provided
        if date_range and date_range.get('start') and date_range.get('end'):
            try:
                range_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as total_revenue,
                    COUNT(DISTINCT sku) as distinct_products,
                    COUNT(DISTINCT region) as distinct_regions
                FROM sales
                WHERE CAST("{date_col}" AS DATE) >= '{date_range['start']}'
                  AND CAST("{date_col}" AS DATE) <= '{date_range['end']}'
                """
                range_df = execute_query(range_query)
                
                if not range_df.empty:
                    context['date_range_summary'] = {
                        'start': date_range['start'],
                        'end': date_range['end'],
                        'total_records': int(range_df.iloc[0]['total_records']),
                        'total_revenue': float(range_df.iloc[0]['total_revenue']) if range_df.iloc[0]['total_revenue'] is not None else 0,
                        'distinct_products': int(range_df.iloc[0]['distinct_products']),
                        'distinct_regions': int(range_df.iloc[0]['distinct_regions'])
                    }
            except Exception as e:
                logger.warning(f"Could not get date range context: {e}")
        
        return context
    
    except Exception as e:
        logger.error(f"Error getting temporal context: {e}")
        return {}


def get_entity_context(entities: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Get context about specific entities mentioned in the question
    
    Args:
        entities: Dictionary with entity types and values (from question analysis)
        
    Returns:
        Dictionary with entity-specific context
    """
    if not table_exists('sales'):
        return {}
    
    # Get actual column names
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    
    context = {}
    
    try:
        # Product/SKU context
        if entities.get('products'):
            for product in entities['products'][:3]:  # Limit to 3 products
                try:
                    product_query = f"""
                    SELECT 
                        COUNT(*) as total_transactions,
                        SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as total_revenue,
                        COUNT(DISTINCT region) as regions_sold_in
                    FROM sales
                    WHERE sku ILIKE '%{product}%'
                    """
                    product_df = execute_query(product_query)
                    
                    if not product_df.empty and product_df.iloc[0]['total_transactions'] > 0:
                        context[f'product_{product}'] = {
                            'sku': product,
                            'total_transactions': int(product_df.iloc[0]['total_transactions']),
                            'total_revenue': float(product_df.iloc[0]['total_revenue']) if product_df.iloc[0]['total_revenue'] is not None else 0,
                            'regions_sold_in': int(product_df.iloc[0]['regions_sold_in'])
                        }
                except Exception as e:
                    logger.warning(f"Could not get product context for {product}: {e}")
        
        # Region context
        if entities.get('regions'):
            for region in entities['regions'][:3]:  # Limit to 3 regions
                try:
                    region_query = f"""
                    SELECT 
                        COUNT(*) as total_transactions,
                        SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN "{rev_col}" ELSE 0 END) as total_revenue,
                        COUNT(DISTINCT sku) as distinct_products
                    FROM sales
                    WHERE region ILIKE '%{region}%'
                    """
                    region_df = execute_query(region_query)
                    
                    if not region_df.empty and region_df.iloc[0]['total_transactions'] > 0:
                        context[f'region_{region}'] = {
                            'region': region,
                            'total_transactions': int(region_df.iloc[0]['total_transactions']),
                            'total_revenue': float(region_df.iloc[0]['total_revenue']) if region_df.iloc[0]['total_revenue'] is not None else 0,
                            'distinct_products': int(region_df.iloc[0]['distinct_products'])
                        }
                except Exception as e:
                    logger.warning(f"Could not get region context for {region}: {e}")
        
        # Transaction type context
        if entities.get('transaction_types'):
            for txn_type in entities['transaction_types']:
                try:
                    txn_query = f"""
                    SELECT 
                        COUNT(*) as count,
                        SUM(ABS("{rev_col}")) as total_amount
                    FROM sales
                    WHERE "{txn_col}" = '{txn_type}'
                    """
                    txn_df = execute_query(txn_query)
                    
                    if not txn_df.empty:
                        context[f'transaction_{txn_type}'] = {
                            'type': txn_type,
                            'count': int(txn_df.iloc[0]['count']),
                            'total_amount': float(txn_df.iloc[0]['total_amount']) if txn_df.iloc[0]['total_amount'] is not None else 0
                        }
                except Exception as e:
                    logger.warning(f"Could not get transaction type context for {txn_type}: {e}")
        
        return context
    
    except Exception as e:
        logger.error(f"Error getting entity context: {e}")
        return {}
