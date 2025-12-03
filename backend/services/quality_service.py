"""
Quality Service - Refunds, cancellations, and replacements analysis
"""
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

from core.database import execute_query, table_exists
from services.column_mapping_service import ColumnMappingService
from utils.logger import app_logger, log_error

logger = logging.getLogger(__name__)

class QualityService:
    """
    Service to handle quality-related metrics (refunds, cancellations, replacements).
    """

    @staticmethod
    def get_refunds_data(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get refund statistics and breakdown.
        """
        if not table_exists('sales'):
            return {
                'total_refunds': 0,
                'refund_amount': 0,
                'refund_rate': 0,
                'top_refund_reasons': []
            }
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            revenue_col = ColumnMappingService.get_revenue_column(columns)
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not revenue_col or not txn_col:
                return {
                    'total_refunds': 0,
                    'refund_amount': 0,
                    'refund_rate': 0,
                    'top_refund_reasons': []
                }
            
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
            
            # Get refund stats
            sql = f"""
            SELECT 
                COUNT(CASE WHEN "{txn_col}" = 'Refund' THEN 1 END) as total_refunds,
                SUM(CASE WHEN "{txn_col}" = 'Refund' THEN ABS({revenue_col}) ELSE 0 END) as refund_amount,
                COUNT(*) as total_transactions
            FROM sales
            WHERE 1=1 {date_filter}
            """
            
            df = execute_query(sql)
            
            if df.empty:
                return {
                    'total_refunds': 0,
                    'refund_amount': 0,
                    'refund_rate': 0,
                    'top_refund_reasons': []
                }
            
            row = df.iloc[0]
            total_refunds = int(row['total_refunds']) if row['total_refunds'] else 0
            refund_amount = float(row['refund_amount']) if row['refund_amount'] else 0
            total_transactions = int(row['total_transactions']) if row['total_transactions'] else 0
            
            refund_rate = (total_refunds / total_transactions * 100) if total_transactions > 0 else 0
            
            return {
                'total_refunds': total_refunds,
                'refund_amount': round(refund_amount, 2),
                'refund_rate': round(refund_rate, 2),
                'top_refund_reasons': []  # Can be extended if reason column exists
            }
            
        except Exception as e:
            log_error(e, 'get_refunds_data')
            return {
                'total_refunds': 0,
                'refund_amount': 0,
                'refund_rate': 0,
                'top_refund_reasons': []
            }

    @staticmethod
    def get_cancellations_data(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cancellation statistics.
        """
        if not table_exists('sales'):
            return {
                'total_cancellations': 0,
                'cancellation_rate': 0
            }
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not txn_col:
                return {
                    'total_cancellations': 0,
                    'cancellation_rate': 0
                }
            
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
            
            # Get cancellation stats
            sql = f"""
            SELECT 
                COUNT(CASE WHEN "{txn_col}" = 'Cancellation' THEN 1 END) as total_cancellations,
                COUNT(*) as total_transactions
            FROM sales
            WHERE 1=1 {date_filter}
            """
            
            df = execute_query(sql)
            
            if df.empty:
                return {
                    'total_cancellations': 0,
                    'cancellation_rate': 0
                }
            
            row = df.iloc[0]
            total_cancellations = int(row['total_cancellations']) if row['total_cancellations'] else 0
            total_transactions = int(row['total_transactions']) if row['total_transactions'] else 0
            
            cancellation_rate = (total_cancellations / total_transactions * 100) if total_transactions > 0 else 0
            
            return {
                'total_cancellations': total_cancellations,
                'cancellation_rate': round(cancellation_rate, 2)
            }
            
        except Exception as e:
            log_error(e, 'get_cancellations_data')
            return {
                'total_cancellations': 0,
                'cancellation_rate': 0
            }

    @staticmethod
    def get_free_replacements_data(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get free replacement statistics.
        """
        if not table_exists('sales'):
            return {
                'total_replacements': 0,
                'replacement_rate': 0
            }
        
        try:
            column_info = execute_query("DESCRIBE sales")
            columns = column_info['column_name'].values.tolist()
            
            txn_col = ColumnMappingService.get_transaction_type_column(columns)
            date_col = ColumnMappingService.get_date_column(columns)
            
            if not txn_col:
                return {
                    'total_replacements': 0,
                    'replacement_rate': 0
                }
            
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
            
            # Get replacement stats
            sql = f"""
            SELECT 
                COUNT(CASE WHEN "{txn_col}" = 'Free Replacement' THEN 1 END) as total_replacements,
                COUNT(*) as total_transactions
            FROM sales
            WHERE 1=1 {date_filter}
            """
            
            df = execute_query(sql)
            
            if df.empty:
                return {
                    'total_replacements': 0,
                    'replacement_rate': 0
                }
            
            row = df.iloc[0]
            total_replacements = int(row['total_replacements']) if row['total_replacements'] else 0
            total_transactions = int(row['total_transactions']) if row['total_transactions'] else 0
            
            replacement_rate = (total_replacements / total_transactions * 100) if total_transactions > 0 else 0
            
            return {
                'total_replacements': total_replacements,
                'replacement_rate': round(replacement_rate, 2)
            }
            
        except Exception as e:
            log_error(e, 'get_free_replacements_data')
            return {
                'total_replacements': 0,
                'replacement_rate': 0
            }
