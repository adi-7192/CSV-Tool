"""
Product Service - SKU performance and top products analysis
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from datetime import datetime, timedelta

from core.database import execute_query, table_exists
from services.column_mapping_service import ColumnMappingService
from utils.logger import app_logger, log_error

logger = logging.getLogger(__name__)

class ProductService:
    """
    Service to handle product performance and top products analysis.
    """

    @staticmethod
    def get_top_products(
        limit: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        metric: str = 'revenue',
    ) -> pd.DataFrame:
        """
        Get top products by SKU with comprehensive metrics.
        """
        if not table_exists('sales'):
            return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            sku_col = ColumnMappingService.get_asin_column(columns) # Prefer ASIN
            if not sku_col:
                 # Try finding a SKU column if ASIN not found
                for col in ['sku', 'SKU', 'Sku']:
                    if col in columns:
                        sku_col = col
                        break
            
            asin_col = ColumnMappingService.get_asin_column(columns)
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            quantity_col = ColumnMappingService.get_quantity_column(columns)
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            # Rating column
            rating_col = None
            for col in ['rating', 'Rating', 'star_rating']:
                if col in columns:
                    rating_col = col
                    break
            
            if not sku_col or not revenue_col:
                return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
            
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
            
            # Build SQL query
            asin_select = f'COALESCE(MAX("{asin_col}"), \'\')' if asin_col else '\'\''
            rating_select = f'COALESCE(AVG("{rating_col}"), 0)' if rating_col else '0'
            
            if txn_col:
                sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    {asin_select} as asin,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({quantity_col if quantity_col else '1'}) ELSE 0 END), 0) as units_sold,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Refund' THEN ABS({revenue_col}) ELSE 0 END), 0) as refund_amount,
                    {rating_select} as rating
                FROM sales
                WHERE 1=1 {date_filter}
                GROUP BY "{sku_col}"
                HAVING revenue > 0
                ORDER BY revenue DESC
                LIMIT {limit}
                """
            else:
                sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    {asin_select} as asin,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({quantity_col if quantity_col else '1'}) ELSE 0 END), 0) as units_sold,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                    COALESCE(SUM(CASE WHEN {revenue_col} < 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as refund_amount,
                    {rating_select} as rating
                FROM sales
                WHERE 1=1 {date_filter}
                GROUP BY "{sku_col}"
                HAVING revenue > 0
                ORDER BY revenue DESC
                LIMIT {limit}
                """
            
            df = execute_query(sql)
            
            if df.empty:
                return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
            
            # Calculate refund_ratio
            df['refund_ratio'] = df.apply(
                lambda row: round((row['refund_amount'] / row['revenue'] * 100) if row['revenue'] > 0 else 0, 1),
                axis=1
            )
            
            # Calculate trend (simplified for now, setting to 0 as in original code fallback)
            df['trend'] = 0.0 
            
            # Ensure rating is numeric
            df['rating'] = df['rating'].fillna(0).astype(float).round(1)
            
            # Round numeric columns
            df['units_sold'] = df['units_sold'].fillna(0).astype(int)
            df['revenue'] = df['revenue'].fillna(0).round(2)
            
            return df[['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend']]
            
        except Exception as e:
            log_error(e, 'get_top_products')
            return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])

    @staticmethod
    def get_movers_decliners(
        start_date: str,
        end_date: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get movers and decliners (products with highest increase/decrease in revenue).
        """
        if not table_exists('sales'):
            return {
                "movers": [],
                "decliners": [],
                "label": "Comparison",
                "granularity": "period"
            }
            
        try:
            # Parse dates
            current_start = datetime.strptime(start_date, '%Y-%m-%d').date()
            current_end = datetime.strptime(end_date, '%Y-%m-%d').date()
            duration = (current_end - current_start).days + 1
            
            # Calculate previous period
            previous_end = current_start - timedelta(days=1)
            previous_start = previous_end - timedelta(days=duration - 1)
            
            prev_start_str = previous_start.strftime('%Y-%m-%d')
            prev_end_str = previous_end.strftime('%Y-%m-%d')
            
            # Get columns
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            sku_col = ColumnMappingService.get_asin_column(columns) or 'sku'
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not revenue_col or not date_col:
                return {"movers": [], "decliners": [], "label": "No Data", "granularity": "period"}
                
            # Check date casting
            needs_cast = False
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
            
            date_expr = f'CAST("{date_col}" AS DATE)' if needs_cast else f'"{date_col}"'
            
            # Query current period
            sql_current = f"""
            SELECT 
                "{sku_col}" as sku,
                SUM({revenue_col}) as revenue
            FROM sales
            WHERE {date_expr} >= '{start_date}' AND {date_expr} <= '{end_date}'
            GROUP BY 1
            HAVING revenue > 0
            """
            df_current = execute_query(sql_current)
            
            # Query previous period
            sql_prev = f"""
            SELECT 
                "{sku_col}" as sku,
                SUM({revenue_col}) as revenue
            FROM sales
            WHERE {date_expr} >= '{prev_start_str}' AND {date_expr} <= '{prev_end_str}'
            GROUP BY 1
            HAVING revenue > 0
            """
            df_prev = execute_query(sql_prev)
            
            if df_current.empty:
                return {"movers": [], "decliners": [], "label": "No Data", "granularity": "period"}
                
            # Merge
            merged = pd.merge(
                df_current, 
                df_prev, 
                on='sku', 
                how='outer', 
                suffixes=('_curr', '_prev')
            ).fillna(0)
            
            # Calculate change
            merged['change'] = merged['revenue_curr'] - merged['revenue_prev']
            merged['pct_change'] = merged.apply(
                lambda x: ((x['revenue_curr'] - x['revenue_prev']) / x['revenue_prev'] * 100) 
                if x['revenue_prev'] > 0 else 100.0 if x['revenue_curr'] > 0 else 0.0,
                axis=1
            )
            
            # Get movers (top gainers)
            movers = merged[merged['change'] > 0].sort_values('change', ascending=False).head(limit)
            movers_list = []
            for _, row in movers.iterrows():
                movers_list.append({
                    'sku': row['sku'],
                    'current_revenue': float(row['revenue_curr']),
                    'previous_revenue': float(row['revenue_prev']),
                    'change': float(row['change']),
                    'pct_change': float(row['pct_change'])
                })
                
            # Get decliners (top losers)
            decliners = merged[merged['change'] < 0].sort_values('change', ascending=True).head(limit)
            decliners_list = []
            for _, row in decliners.iterrows():
                decliners_list.append({
                    'sku': row['sku'],
                    'current_revenue': float(row['revenue_curr']),
                    'previous_revenue': float(row['revenue_prev']),
                    'change': float(row['change']),
                    'pct_change': float(row['pct_change'])
                })
                
            return {
                "movers": movers_list,
                "decliners": decliners_list,
                "label": f"vs {prev_start_str} to {prev_end_str}",
                "granularity": "period"
            }
            
        except Exception as e:
            log_error(e, 'get_movers_decliners')
            return {
                "movers": [],
                "decliners": [],
                "label": "Error",
                "granularity": "period"
            }

    @staticmethod
    def get_skus_by_city(
        city: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get top SKUs for a specific city.
        """
        if not table_exists('sales'):
            return {'city': city, 'data': [], 'count': 0}
            
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            city_col = ColumnMappingService.get_city_column(columns)
            sku_col = ColumnMappingService.get_asin_column(columns) or 'sku'
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            quantity_col = ColumnMappingService.get_quantity_column(columns)
            
            if not city_col or not revenue_col:
                return {'city': city, 'data': [], 'count': 0}
            
            sql = f"""
            SELECT 
                "{sku_col}" as sku,
                SUM({revenue_col}) as revenue,
                SUM({quantity_col if quantity_col else '1'}) as units
            FROM sales
            WHERE "{city_col}" ILIKE '{city}'
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT {limit}
            """
            
            df = execute_query(sql)
            
            if df.empty:
                return {'city': city, 'data': [], 'count': 0}
                
            return {
                'city': city,
                'data': df.to_dict('records'),
                'count': len(df)
            }
            
        except Exception as e:
            log_error(e, 'get_skus_by_city')
            return {'city': city, 'data': [], 'count': 0}

    @staticmethod
    def get_top_products_performance(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        view_type: str = 'monthly',
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get top products performance (revenue trend over time).
        """
        if not table_exists('sales'):
            return {
                "products": [],
                "period_labels": [],
                "view_type": view_type
            }
            
        try:
            # Get columns
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            sku_col = ColumnMappingService.get_asin_column(columns) or 'sku'
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not revenue_col or not date_col:
                return {"products": [], "period_labels": [], "view_type": view_type}
                
            # Check date casting
            needs_cast = False
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
            
            date_expr = f'CAST("{date_col}" AS DATE)' if needs_cast else f'"{date_col}"'
            
            # Date filter
            date_filter = ""
            if start_date and end_date:
                date_filter = f"AND {date_expr} >= '{start_date}' AND {date_expr} <= '{end_date}'"
            
            # 1. Get Top N Products by Total Revenue
            sql_top = f"""
            SELECT 
                "{sku_col}" as sku,
                SUM({revenue_col}) as total_revenue
            FROM sales
            WHERE 1=1 {date_filter}
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT {limit}
            """
            df_top = execute_query(sql_top)
            
            if df_top.empty:
                return {"products": [], "period_labels": [], "view_type": view_type}
                
            top_skus = df_top['sku'].tolist()
            top_skus_str = "', '".join(top_skus)
            
            # 2. Get Daily/Monthly Revenue for Top Products
            # Determine date truncation based on view_type
            if view_type == 'daily':
                date_trunc = f"DATE_TRUNC('day', {date_expr})"
                date_format = '%Y-%m-%d'
            elif view_type == 'weekly':
                date_trunc = f"DATE_TRUNC('week', {date_expr})"
                date_format = '%Y-%m-%d' # Week start date
            else: # monthly
                date_trunc = f"DATE_TRUNC('month', {date_expr})"
                date_format = '%Y-%m'
                
            sql_trend = f"""
            SELECT 
                "{sku_col}" as sku,
                {date_trunc} as period,
                SUM({revenue_col}) as revenue
            FROM sales
            WHERE "{sku_col}" IN ('{top_skus_str}') {date_filter}
            GROUP BY 1, 2
            ORDER BY 2 ASC
            """
            df_trend = execute_query(sql_trend)
            
            if df_trend.empty:
                return {"products": [], "period_labels": [], "view_type": view_type}
                
            # Format data
            # Get all unique periods
            periods = sorted(df_trend['period'].unique())
            period_labels = [pd.to_datetime(p).strftime(date_format) for p in periods]
            
            products_data = []
            for sku in top_skus:
                sku_data = df_trend[df_trend['sku'] == sku]
                
                # Fill missing periods with 0
                data_points = []
                for period in periods:
                    match = sku_data[sku_data['period'] == period]
                    if not match.empty:
                        data_points.append(float(match['revenue'].values[0]))
                    else:
                        data_points.append(0.0)
                        
                products_data.append({
                    "name": sku,
                    "data": data_points,
                    "total": float(df_top[df_top['sku'] == sku]['total_revenue'].values[0])
                })
                
            return {
                "products": products_data,
                "period_labels": period_labels,
                "view_type": view_type
            }
            
        except Exception as e:
            log_error(e, 'get_top_products_performance')
            return {
                "products": [],
                "period_labels": [],
                "view_type": view_type
            }
