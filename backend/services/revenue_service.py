"""
Revenue Service - Revenue calculations, trends, and metrics
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from datetime import datetime, timedelta

from core.database import execute_query, table_exists
from services.column_mapping_service import ColumnMappingService
from utils.logger import app_logger, log_error

logger = logging.getLogger(__name__)

class RevenueService:
    """
    Service to handle revenue-related calculations and metrics.
    """

    @staticmethod
    def calculate_metrics(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate key revenue metrics (total revenue, orders, avg order value, etc.)
        """
        if not table_exists('sales'):
            return {
                'total_revenue': 0,
                'total_orders': 0,
                'avg_order_value': 0,
                'total_units': 0,
                'refund_rate': 0,
                'cancellation_rate': 0
            }
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            order_id_col = ColumnMappingService.get_order_id_column(columns)
            quantity_col = ColumnMappingService.get_quantity_column(columns)
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not revenue_col:
                return {
                    'total_revenue': 0,
                    'total_orders': 0,
                    'avg_order_value': 0,
                    'total_units': 0,
                    'refund_rate': 0,
                    'cancellation_rate': 0
                }
            
            # Check if date column needs casting
            needs_cast = False
            if date_col:
                col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
                needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
            
            # Build date filter
            date_filter = ""
            if start_date and end_date and date_col:
                if needs_cast:
                    date_filter = f'AND CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\''
                else:
                    date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
            
            # Calculate metrics
            if txn_col:
                sql = f"""
                SELECT 
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END) as total_revenue,
                    COUNT(DISTINCT CASE WHEN "{txn_col}" = 'Shipment' THEN {order_id_col if order_id_col else 'NULL'} END) as total_orders,
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({quantity_col if quantity_col else '1'}) ELSE 0 END) as total_units,
                    SUM(CASE WHEN "{txn_col}" = 'Refund' THEN ABS({revenue_col}) ELSE 0 END) as refund_amount,
                    COUNT(CASE WHEN "{txn_col}" = 'Refund' THEN 1 END) as refund_count,
                    COUNT(CASE WHEN "{txn_col}" = 'Cancellation' THEN 1 END) as cancellation_count,
                    COUNT(*) as total_transactions
                FROM sales
                WHERE 1=1 {date_filter}
                """
            else:
                sql = f"""
                SELECT 
                    SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END) as total_revenue,
                    COUNT(DISTINCT {order_id_col if order_id_col else 'NULL'}) as total_orders,
                    SUM(CASE WHEN {revenue_col} > 0 THEN ABS({quantity_col if quantity_col else '1'}) ELSE 0 END) as total_units,
                    SUM(CASE WHEN {revenue_col} < 0 THEN ABS({revenue_col}) ELSE 0 END) as refund_amount,
                    COUNT(CASE WHEN {revenue_col} < 0 THEN 1 END) as refund_count,
                    0 as cancellation_count,
                    COUNT(*) as total_transactions
                FROM sales
                WHERE 1=1 {date_filter}
                """
            
            df = execute_query(sql)
            
            if df.empty:
                return {
                    'total_revenue': 0,
                    'total_orders': 0,
                    'avg_order_value': 0,
                    'total_units': 0,
                    'refund_rate': 0,
                    'cancellation_rate': 0
                }
            
            row = df.iloc[0]
            total_revenue = float(row['total_revenue']) if row['total_revenue'] else 0
            total_orders = int(row['total_orders']) if row['total_orders'] else 0
            total_units = int(row['total_units']) if row['total_units'] else 0
            refund_amount = float(row['refund_amount']) if row['refund_amount'] else 0
            refund_count = int(row['refund_count']) if row['refund_count'] else 0
            cancellation_count = int(row['cancellation_count']) if row['cancellation_count'] else 0
            total_transactions = int(row['total_transactions']) if row['total_transactions'] else 0
            
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            refund_rate = (refund_count / total_transactions * 100) if total_transactions > 0 else 0
            cancellation_rate = (cancellation_count / total_transactions * 100) if total_transactions > 0 else 0
            
            return {
                'total_revenue': round(total_revenue, 2),
                'total_orders': total_orders,
                'avg_order_value': round(avg_order_value, 2),
                'total_units': total_units,
                'refund_rate': round(refund_rate, 2),
                'cancellation_rate': round(cancellation_rate, 2)
            }
            
        except Exception as e:
            log_error(e, 'calculate_metrics')
            return {
                'total_revenue': 0,
                'total_orders': 0,
                'avg_order_value': 0,
                'total_units': 0,
                'refund_rate': 0,
                'cancellation_rate': 0
            }

    @staticmethod
    def get_daily_trends(
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get daily revenue trends for the specified date range.
        """
        if not table_exists('sales'):
            return []
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            
            if not revenue_col or not date_col:
                return []
            
            # Check if date column needs casting
            needs_cast = False
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
            
            date_expr = f'CAST("{date_col}" AS DATE)' if needs_cast else f'"{date_col}"'
            
            if txn_col:
                sql = f"""
                SELECT 
                    {date_expr} as date,
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END) as revenue
                FROM sales
                WHERE {date_expr} >= '{start_date}' AND {date_expr} <= '{end_date}'
                GROUP BY {date_expr}
                ORDER BY {date_expr}
                """
            else:
                sql = f"""
                SELECT 
                    {date_expr} as date,
                    SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END) as revenue
                FROM sales
                WHERE {date_expr} >= '{start_date}' AND {date_expr} <= '{end_date}'
                GROUP BY {date_expr}
                ORDER BY {date_expr}
                """
            
            df = execute_query(sql)
            
            if df.empty:
                return []
            
            # Convert to list of dicts
            result = []
            for _, row in df.iterrows():
                result.append({
                    'date': str(row['date']),
                    'revenue': float(row['revenue'])
                })
            
            return result
            
        except Exception as e:
            log_error(e, 'get_daily_trends')
            return []

    @staticmethod
    def get_revenue_by_city(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get revenue breakdown by city/region.
        """
        if not table_exists('sales'):
            return []
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            city_col = ColumnMappingService.get_city_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            
            if not revenue_col or not city_col:
                return []
            
            # Build date filter
            date_filter = ""
            if start_date and end_date and date_col:
                needs_cast = False
                col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
                needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
                
                if needs_cast:
                    date_filter = f'AND CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\''
                else:
                    date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
            
            if txn_col:
                sql = f"""
                SELECT 
                    "{city_col}" as city,
                    SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END) as revenue
                FROM sales
                WHERE "{city_col}" IS NOT NULL AND "{city_col}" != '' {date_filter}
                GROUP BY "{city_col}"
                HAVING revenue > 0
                ORDER BY revenue DESC
                LIMIT {limit}
                """
            else:
                sql = f"""
                SELECT 
                    "{city_col}" as city,
                    SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END) as revenue
                FROM sales
                WHERE "{city_col}" IS NOT NULL AND "{city_col}" != '' {date_filter}
                GROUP BY "{city_col}"
                HAVING revenue > 0
                ORDER BY revenue DESC
                LIMIT {limit}
                """
            
            df = execute_query(sql)
            
            if df.empty:
                return []
            
            # Convert to list of dicts with normalized city names
            result = []
            for _, row in df.iterrows():
                normalized_city = ColumnMappingService.normalize_city_name(row['city'])
                if normalized_city:
                    result.append({
                        'city': normalized_city,
                        'revenue': float(row['revenue'])
                    })
            
            return result
            
        except Exception as e:
            log_error(e, 'get_revenue_by_city')
            return []
