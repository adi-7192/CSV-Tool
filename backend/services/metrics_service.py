"""
Metrics Service - Business KPI calculations

Extracted and refactored from legacy/app.py
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from core.database import execute_query, table_exists
from utils.error_handler import handle_service_error, log_error
from utils.logger import app_logger, log_function_entry, log_function_exit, log_error_with_context

logger = logging.getLogger(__name__)


# ============================================================================
# CITY NORMALIZATION - Shared function for consistent city name matching
# ============================================================================

def normalize_city_name(city_name: str) -> Optional[str]:
    """
    Normalize city name using mapping and title case.
    
    This function ensures consistent city name matching across:
    - get_revenue_by_city() (chart data)
    - get_skus_by_city() (modal data)
    
    Handles city name variations (Bangalore/Bengaluru, Delhi/New Delhi, etc.)
    and ensures Kolkata, Ahmedabad, and Navi Mumbai are properly normalized.
    
    Args:
        city_name: Raw city name from database or user input
    
    Returns:
        Normalized city name (e.g., "Kolkata", "Ahmedabad", "Navi Mumbai")
        Returns None if input is empty or invalid
    """
    if not city_name:
        return None
    
    # Handle pandas NaN values
    if pd.isna(city_name):
        return None
    
    city_str = str(city_name).strip().lower()
    if not city_str:
        return None
    
    # City normalization mapping
    # Maps common variations to standardized city names
    city_mapping = {
        'bangalore': 'Bangalore',
        'bengaluru': 'Bangalore',
        'delhi': 'New Delhi',
        'new delhi': 'New Delhi',
        'delhi ncr': 'New Delhi',
        'bombay': 'Mumbai',
        'kolkata': 'Kolkata',
        'calcutta': 'Kolkata',
        'ahmedabad': 'Ahmedabad',
        'ahmadabad': 'Ahmedabad',
        'hyderabad': 'Hyderabad',
        'pune': 'Pune',
        'poona': 'Pune',
        'chennai': 'Chennai',
        'madras': 'Chennai',
        'gurugram': 'Gurugram',
        'gurgaon': 'Gurugram',
        'noida': 'Noida',
        'navi mumbai': 'Navi Mumbai',
        'new mumbai': 'Navi Mumbai',
        'thane': 'Thane',
    }
    
    # Check mapping first
    if city_str in city_mapping:
        return city_mapping[city_str]
    
    # Otherwise, apply title case
    words = city_str.split()
    normalized_words = [word.capitalize() for word in words]
    return ' '.join(normalized_words)


def calculate_transaction_revenue(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate revenue based on transaction types - uses derived fields if available
    
    Extracted from legacy/app.py:calculate_transaction_revenue()
    
    Args:
        df: DataFrame with sales data
    
    Returns:
        Dictionary with revenue metrics
    """
    if df.empty:
        return {
            'gross_revenue': 0.0,
            'refunds': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_cost_loss': 0.0,
            'net_revenue': 0.0,
            'units_sold': 0,
            'orders': 0,
            'aov': 0.0,
            'transaction_breakdown': {}
        }
    
    # Check if we have derived fields from cleaning
    if 'revenue_calc' in df.columns and 'shipping_loss_calc' in df.columns:
        # Calculate directly from derived fields
        gross_revenue = df[df['revenue_calc'] > 0]['revenue_calc'].sum()
        refunds = abs(df[df['revenue_calc'] < 0]['revenue_calc'].sum())
        shipping_loss = df['shipping_loss_calc'].sum()
        units_sold = df['units_sold_calc'].sum() if 'units_sold_calc' in df.columns else 0
        
        # Get order count (handle various column names)
        order_id_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['invoice number', 'order id', 'orderid'])]
        orders = df[order_id_cols[0]].nunique() if order_id_cols else 0
        
        # Calculate FreeReplacement cost estimation (simplified)
        free_replacement_cost = 0.0
        if 'transaction_type' in df.columns:
            freereplacement_data = df[df['transaction_type'] == 'FreeReplacement']
            if not freereplacement_data.empty:
                # Estimate as average shipment revenue for FreeReplacement items
                shipment_data = df[df['transaction_type'] == 'Shipment']
                if not shipment_data.empty:
                    free_replacement_cost = freereplacement_data['revenue_calc'].sum()
        
        net_revenue = gross_revenue - refunds - shipping_loss - free_replacement_cost
        aov = net_revenue / orders if orders > 0 else 0.0
        
        # Build transaction breakdown
        breakdown = {}
        if 'transaction_type' in df.columns:
            for txn in df['transaction_type'].unique():
                if pd.notna(txn):
                    txn_data = df[df['transaction_type'] == txn]
                    breakdown[str(txn)] = {
                        'count': len(txn_data),
                        'revenue': float(txn_data['revenue_calc'].sum()),
                        'units': int(txn_data['units_sold_calc'].sum()) if 'units_sold_calc' in txn_data.columns else 0,
                    }
        
        return {
            'gross_revenue': float(gross_revenue),
            'refunds': float(refunds),
            'free_replacement_cost': float(free_replacement_cost),
            'shipping_cost_loss': float(shipping_loss),
            'net_revenue': float(net_revenue),
            'units_sold': int(units_sold),
            'orders': int(orders),
            'aov': float(aov),
            'transaction_breakdown': breakdown
        }
    
    # Fallback: use legacy calculation if derived fields not available
    if 'revenue_in_inr' in df.columns:
        gross_revenue = df['revenue_in_inr'].sum()
        units_sold = df['quantity'].sum() if 'quantity' in df.columns else 0
        order_id_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['invoice number', 'order id'])]
        orders = df[order_id_cols[0]].nunique() if order_id_cols else 0
        aov = gross_revenue / orders if orders > 0 else 0.0
        
        return {
            'gross_revenue': float(gross_revenue),
            'refunds': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_cost_loss': 0.0,
            'net_revenue': float(gross_revenue),
            'units_sold': int(units_sold),
            'orders': int(orders),
            'aov': float(aov),
            'transaction_breakdown': {
                'Shipment': {
                    'count': len(df),
                    'revenue': float(gross_revenue),
                    'units': int(units_sold),
                }
            }
        }
    
    # Return zeros if no revenue columns found
    return {
        'gross_revenue': 0.0,
        'refunds': 0.0,
        'free_replacement_cost': 0.0,
        'shipping_cost_loss': 0.0,
        'net_revenue': 0.0,
        'units_sold': 0,
        'orders': 0,
        'aov': 0.0,
        'transaction_breakdown': {}
    }


def get_filtered_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Get filtered data for date range and optional filters
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        transaction_type: Optional transaction type filter
        source_file: Optional source file filter
    
    Returns:
        DataFrame with filtered data or None if no data
    """
    if not table_exists('sales'):
        return None
    
    try:
        where_conditions = []
        
        # Date filtering (handle different date column names)
        date_col = None
        # Try to find date column
        column_info = execute_query("DESCRIBE sales")
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if date_col and start_date and end_date:
            where_conditions.append(f'"{date_col}" >= \'{start_date}\'')
            where_conditions.append(f'"{date_col}" <= \'{end_date}\'')
        
        # Transaction type filter
        if transaction_type and transaction_type != "All":
            txn_col = None
            txn_columns = ['Transaction Type', 'transaction_type']
            for col in txn_columns:
                if col in column_info['column_name'].values:
                    txn_col = col
                    break
            
            if txn_col:
                where_conditions.append(f'"{txn_col}" = \'{transaction_type}\'')
        
        # Source file filter
        if source_file:
            where_conditions.append(f'source_file = \'{source_file}\'')
        
        # Build query
        where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
        sql = f'SELECT * FROM sales WHERE {where_clause} ORDER BY "{date_col}"' if date_col else f'SELECT * FROM sales WHERE {where_clause}'
        
        df = execute_query(sql)
        return df if not df.empty else None
        
    except ValueError as e:
        log_error(e, 'get_filtered_data', {'start_date': start_date, 'end_date': end_date, 'transaction_type': transaction_type})
        return None
    except Exception as e:
        log_error(e, 'get_filtered_data', {'start_date': start_date, 'end_date': end_date, 'transaction_type': transaction_type})
        return None


def _calculate_free_replacement_cost(
    free_replacement_data: pd.DataFrame,
    shipment_data: pd.DataFrame,
) -> float:
    """
    Calculate estimated cost of free replacements using ASIN-based average pricing
    
    Logic (from legacy app):
    1. For each free replacement, get the ASIN
    2. Find average selling price of that ASIN from past shipments
    3. Cost = Average Price × 2 (original product + replacement)
    4. Add shipping costs if available
    
    Args:
        free_replacement_data: DataFrame with FreeReplacement transactions
        shipment_data: DataFrame with Shipment transactions (for pricing reference)
    
    Returns:
        Total estimated cost of all free replacements
    """
    total_cost = 0.0
    
    if free_replacement_data.empty:
        return total_cost
    
    # Find ASIN column (handle case variations)
    asin_col = None
    asin_candidates = ['asin', 'Asin', 'ASIN', 'Amazon ASIN']
    for col in free_replacement_data.columns:
        if col in asin_candidates:
            asin_col = col
            break
    
    if not asin_col or shipment_data.empty:
        # Fallback: Use overall average if ASIN not available
        if not shipment_data.empty:
            # Try to find revenue column
            revenue_col = None
            for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']:
                if col in shipment_data.columns:
                    revenue_col = col
                    break
            
            if revenue_col:
                avg_price = shipment_data[revenue_col].mean()
                count = len(free_replacement_data)
                total_cost = avg_price * 2 * count
                logger.warning(f"Using overall average price for FreeReplacement cost: ₹{avg_price:.2f}")
        return total_cost
    
    # Find revenue column in shipment data
    revenue_col = None
    for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']:
        if col in shipment_data.columns:
            revenue_col = col
            break
    
    if not revenue_col:
        return total_cost
    
    # Find shipping column
    shipping_col = None
    for col in ['shipping_amount', 'Shipping Amount', 'shipping_loss_calc']:
        if col in shipment_data.columns or col in free_replacement_data.columns:
            shipping_col = col
            break
    
    # Calculate cost per ASIN
    for _, row in free_replacement_data.iterrows():
        asin = row.get(asin_col)
        
        if pd.isna(asin) or asin == '':
            continue
        
        # Find average price for this ASIN from shipment data
        asin_shipments = shipment_data[shipment_data[asin_col] == asin] if asin_col in shipment_data.columns else pd.DataFrame()
        
        if not asin_shipments.empty:
            avg_revenue = asin_shipments[revenue_col].mean()
            
            # Cost = 2x average price (original + replacement)
            replacement_cost = avg_revenue * 2
            
            # Add shipping if available (2x for original + replacement shipping)
            if shipping_col and shipping_col in free_replacement_data.columns:
                row_shipping = row.get(shipping_col, 0)
                if pd.notna(row_shipping):
                    replacement_cost += row_shipping * 2
            elif shipping_col and shipping_col in shipment_data.columns and not asin_shipments.empty:
                avg_shipping = asin_shipments[shipping_col].mean()
                replacement_cost += avg_shipping * 2
            
            total_cost += replacement_cost
        else:
            # No pricing data for this ASIN, use overall average
            if not shipment_data.empty:
                overall_avg = shipment_data[revenue_col].mean()
                total_cost += overall_avg * 2
    
    return total_cost


def calculate_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate core business metrics (KPIs) using transaction-aware logic
    
    Net Revenue Calculation:
        net_revenue = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
    
    Transaction Types:
    - Shipment: Positive revenue (sale completed) → contributes to gross_revenue
    - Refund: Negative revenue + shipping loss → refund_amount deduction
    - Cancel: Cancellation amount → cancellation_amount deduction
    - FreeReplacement: Estimated cost (2x ASIN price) → free_replacement_cost deduction
    
    Args:
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        transaction_type: Optional transaction type filter
        source_file: Optional source file filter
    
    Returns:
        Dictionary with accurate KPI values accounting for all transaction types:
        - gross_revenue: Gross revenue from shipments (SUM of Shipment transactions)
        - refund_amount: Total refund amount (SUM of Refund transactions, absolute value)
        - cancellation_amount: Total cancellation amount (SUM of Cancel transactions, absolute value)
        - free_replacement_cost: Estimated cost of free replacements (2x ASIN price)
        - net_revenue: Net revenue after all deductions
            Formula: gross_revenue - refund_amount - cancellation_amount - free_replacement_cost
        - net_margin: Net margin percentage (net_revenue / gross_revenue * 100)
        - shipping_loss: Shipping costs lost on refunds (separate from refund_amount)
        - orders: Number of successful orders (unique order_ids from Shipments)
        - avg_order_value: Average order value (gross_revenue / orders)
        - success_rate: % of orders not refunded
        - transaction_breakdown: Count and amounts by transaction type
        - revenue: Alias for gross_revenue (backward compatibility)
        - refunds: Alias for refund_amount (backward compatibility)
    """
    log_function_entry(
        app_logger,
        'calculate_metrics',
        {
            'start_date': start_date,
            'end_date': end_date,
            'transaction_type': transaction_type,
            'source_file': source_file
        }
    )
    
    if not table_exists('sales'):
        app_logger.warning("Sales table does not exist")
        log_function_exit(app_logger, 'calculate_metrics', success=True)
        return {
            'gross_revenue': 0.0,
            'revenue': 0.0,  # Backward compatibility
            'refund_amount': 0.0,
            'refunds': 0.0,  # Backward compatibility
            'cancellation_amount': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_loss': 0.0,
            'net_revenue': 0.0,
            'net_margin': 0.0,
            'refund_rate': 0.0,
            'orders': 0,
            'units_sold': 0,
            'avg_order_value': 0.0,
            'success_rate': 0.0,
            'transaction_breakdown': {},
        }
    
    try:
        # Build date filter
        date_filter = ""
        if start_date and end_date:
            # Find date column name
            column_info = execute_query("DESCRIBE sales")
            date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
            date_col = None
            for col in date_columns:
                if col in column_info['column_name'].values:
                    date_col = col
                    break
            
            if date_col:
                # Check if date column is VARCHAR (needs casting)
                col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
                if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                    date_filter = f"AND CAST(\"{date_col}\" AS DATE) >= '{start_date}' AND CAST(\"{date_col}\" AS DATE) <= '{end_date}'"
                else:
                    date_filter = f"AND \"{date_col}\" >= '{start_date}' AND \"{date_col}\" <= '{end_date}'"
        
        # Build transaction type filter
        txn_filter = ""
        if transaction_type and transaction_type != "All":
            txn_col = None
            column_info = execute_query("DESCRIBE sales")
            txn_columns = ['Transaction Type', 'transaction_type']
            for col in txn_columns:
                if col in column_info['column_name'].values:
                    txn_col = col
                    break
            if txn_col:
                txn_filter = f"AND \"{txn_col}\" = '{transaction_type}'"
        
        # Build source file filter
        source_filter = ""
        if source_file:
            source_filter = f"AND source_file = '{source_file}'"
        
        # Get all transaction data (using standardized column names)
        all_data_sql = f"""
        SELECT 
            transaction_type,
            revenue_amount,
            shipping_amount,
            sku,
            order_id,
            order_date,
            quantity
        FROM sales
        WHERE 1=1 {date_filter} {txn_filter} {source_filter}
        """
        
        df = execute_query(all_data_sql)
        
        if df.empty:
            return {
                'gross_revenue': 0.0,
                'revenue': 0.0,  # Backward compatibility
                'refund_amount': 0.0,
                'refunds': 0.0,  # Backward compatibility
                'cancellation_amount': 0.0,
                'free_replacement_cost': 0.0,
                'shipping_loss': 0.0,
                'net_revenue': 0.0,
                'net_margin': 0.0,
                'refund_rate': 0.0,
                'orders': 0,
                'units_sold': 0,
                'avg_order_value': 0.0,
                'success_rate': 0.0,
                'transaction_breakdown': {},
            }
        
        # Normalize column names (handle variations)
        # Transaction type column
        if 'transaction_type' in df.columns:
            txn_col = 'transaction_type'
        elif 'Transaction Type' in df.columns:
            txn_col = 'Transaction Type'
            df['transaction_type'] = df['Transaction Type']
        else:
            logger.warning("No transaction type column found")
            return {
                'gross_revenue': 0.0,
                'revenue': 0.0,  # Backward compatibility
                'refund_amount': 0.0,
                'refunds': 0.0,  # Backward compatibility
                'cancellation_amount': 0.0,
                'free_replacement_cost': 0.0,
                'shipping_loss': 0.0,
                'net_revenue': 0.0,
                'net_margin': 0.0,
                'refund_rate': 0.0,
                'orders': 0,
                'avg_order_value': 0.0,
                'success_rate': 0.0,
                'transaction_breakdown': {},
            }
        
        # Revenue column (prioritize revenue_calc as it has transaction-aware signs)
        revenue_col = None
        for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']:
            if col in df.columns:
                revenue_col = col
                break
        
        if not revenue_col:
            logger.warning("No revenue column found")
            return {
                'gross_revenue': 0.0,
                'revenue': 0.0,  # Backward compatibility
                'refund_amount': 0.0,
                'refunds': 0.0,  # Backward compatibility
                'cancellation_amount': 0.0,
                'free_replacement_cost': 0.0,
                'shipping_loss': 0.0,
                'net_revenue': 0.0,
                'net_margin': 0.0,
                'refund_rate': 0.0,
                'orders': 0,
                'avg_order_value': 0.0,
                'success_rate': 0.0,
                'transaction_breakdown': {},
            }
        
        # Shipping column
        shipping_col = None
        for col in ['shipping_loss_calc', 'shipping_amount', 'Shipping Amount']:
            if col in df.columns:
                shipping_col = col
                break
        
        # Order ID column
        order_id_col = None
        for col in ['order_id', 'Invoice Number']:
            if col in df.columns:
                order_id_col = col
                break
        
        # ASIN column
        asin_col = None
        for col in ['asin', 'Asin', 'ASIN']:
            if col in df.columns:
                asin_col = col
                break
        
        # Initialize results
        results = {
            'gross_revenue': 0.0,  # Renamed from 'revenue' for clarity
            'revenue': 0.0,  # Keep for backward compatibility
            'refund_amount': 0.0,  # Renamed from 'refunds' for clarity
            'refunds': 0.0,  # Keep for backward compatibility
            'cancellation_amount': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_loss': 0.0,
            'net_revenue': 0.0,
            'net_margin': 0.0,
            'refund_rate': 0.0,
            'orders': 0,
            'units_sold': 0,
            'avg_order_value': 0.0,
            'success_rate': 0.0,
            'transaction_breakdown': {},
        }
        
        # Normalize transaction types (handle variations)
        df['transaction_type'] = df['transaction_type'].astype(str).str.strip()
        df['transaction_type'] = df['transaction_type'].replace({
            'Free Replacement': 'FreeReplacement',
            'Free_Replacement': 'FreeReplacement',
        })
        
        # Process each transaction type
        for txn_type in df['transaction_type'].unique():
            if pd.isna(txn_type) or txn_type == 'nan':
                continue
            
            type_data = df[df['transaction_type'] == txn_type].copy()
            count = len(type_data)
            
            if txn_type == 'Shipment':
                # Calculate gross revenue from shipments
                # Positive revenue (if revenue_calc exists, it's already positive; otherwise use absolute value)
                revenue = type_data[revenue_col].sum() if revenue_col in type_data.columns else 0.0
                if revenue_col == 'revenue_calc':
                    revenue = max(0, revenue)  # Ensure positive (should already be from cleaning)
                else:
                    revenue = abs(revenue)  # Use absolute value for raw amounts
                
                orders = type_data[order_id_col].nunique() if order_id_col else len(type_data)
                
                # Calculate units sold from quantity column
                quantity_col = None
                for col in ['quantity', 'Quantity', 'units_sold']:
                    if col in type_data.columns:
                        quantity_col = col
                        break
                
                units_sold = int(type_data[quantity_col].sum()) if quantity_col and quantity_col in type_data.columns else 0
                
                gross_revenue = float(revenue)
                results['gross_revenue'] = gross_revenue
                results['revenue'] = gross_revenue  # Keep for backward compatibility
                results['orders'] = int(orders)
                results['units_sold'] = units_sold
                results['transaction_breakdown']['Shipment'] = {
                    'count': count,
                    'revenue': gross_revenue,
                    'orders': int(orders),
                    'units_sold': units_sold,
                }
            
            elif txn_type == 'Refund':
                # Calculate refund deduction amount
                # If revenue_calc exists, it's already negative; otherwise take absolute value
                if revenue_col == 'revenue_calc':
                    refund_amount = abs(type_data[revenue_col].sum()) if revenue_col in type_data.columns else 0.0
                else:
                    # For raw amounts, they might be positive or negative
                    refund_amount = abs(type_data[revenue_col].sum()) if revenue_col in type_data.columns else 0.0
                
                shipping_loss = type_data[shipping_col].sum() if shipping_col and shipping_col in type_data.columns else 0.0
                
                refund_deduction = float(refund_amount)
                results['refund_amount'] = refund_deduction
                results['refunds'] = refund_deduction  # Keep for backward compatibility
                results['shipping_loss'] += float(shipping_loss)
                results['transaction_breakdown']['Refund'] = {
                    'count': count,
                    'refund_amount': refund_deduction,
                    'shipping_loss': float(shipping_loss),
                }
            
            elif txn_type == 'Cancel':
                # Cancellation amount (deduction from gross revenue)
                cancel_amount = abs(type_data[revenue_col].sum()) if revenue_col in type_data.columns else 0.0
                
                results['cancellation_amount'] = float(cancel_amount)
                results['transaction_breakdown']['Cancel'] = {
                    'count': count,
                    'revenue': float(cancel_amount),
                }
            
            elif txn_type in ['FreeReplacement', 'Free Replacement']:
                # Calculate estimated cost based on ASIN average prices
                shipment_data = df[df['transaction_type'] == 'Shipment'].copy()
                free_replacement_cost = _calculate_free_replacement_cost(
                    type_data, 
                    shipment_data
                )
                
                results['free_replacement_cost'] = float(free_replacement_cost)
                results['transaction_breakdown']['FreeReplacement'] = {
                    'count': count,
                    'estimated_cost': float(free_replacement_cost),
                }
        
        # Calculate net revenue
        # Formula: net_revenue = gross_revenue - refund_deduction - cancel_deduction - free_repl_deduction
        # Note: shipping_loss is already included in refund processing, but we keep it separate for reporting
        results['net_revenue'] = (
            results['gross_revenue']
            - results['refund_amount']
            - results['cancellation_amount']
            - results['free_replacement_cost']
        )
        
        # Calculate AOV (Average Order Value)
        if results['orders'] > 0:
            results['avg_order_value'] = results['gross_revenue'] / results['orders']
        
        # Calculate success rate (orders not refunded)
        refund_count = results['transaction_breakdown'].get('Refund', {}).get('count', 0)
        if results['orders'] > 0:
            results['success_rate'] = ((results['orders'] - refund_count) / results['orders']) * 100
        
        # Net margin and refund rate (% of gross revenue)
        gross_revenue_val = results['gross_revenue']
        if gross_revenue_val > 0:
            results['net_margin'] = (results['net_revenue'] / gross_revenue_val) * 100
            results['refund_rate'] = (results['refund_amount'] / gross_revenue_val) * 100
        else:
            results['net_margin'] = 0.0
            results['refund_rate'] = 0.0
        
        # Round all values
        for key in ['gross_revenue', 'revenue', 'refund_amount', 'refunds', 'cancellation_amount', 
                    'shipping_loss', 'free_replacement_cost', 'net_revenue', 'avg_order_value', 
                    'success_rate', 'net_margin', 'refund_rate']:
            if key in results:
                results[key] = round(results[key], 2)
        
        app_logger.info(f"Calculated metrics for period: {start_date} to {end_date}")
        app_logger.debug(f"Gross revenue: {results.get('gross_revenue', 0)}")
        app_logger.debug(f"Net revenue: {results.get('net_revenue', 0)}")
        app_logger.debug(f"Orders: {results.get('orders', 0)}")
        
        log_function_exit(app_logger, 'calculate_metrics', success=True)
        return results
    
    except ValueError as e:
        log_error_with_context(
            app_logger,
            e,
            'calculate_metrics',
            {
                'start_date': start_date,
                'end_date': end_date,
                'transaction_type': transaction_type,
                'source_file': source_file
            }
        )
        log_function_exit(app_logger, 'calculate_metrics', success=False)
        log_error(e, 'calculate_metrics', {'start_date': start_date, 'end_date': end_date})
        return handle_service_error(e, 'calculate_metrics', {'start_date': start_date, 'end_date': end_date})
    except Exception as e:
        log_error_with_context(
            app_logger,
            e,
            'calculate_metrics',
            {
                'start_date': start_date,
                'end_date': end_date,
                'transaction_type': transaction_type,
                'source_file': source_file
            }
        )
        log_function_exit(app_logger, 'calculate_metrics', success=False)
        log_error(e, 'calculate_metrics', {'start_date': start_date, 'end_date': end_date})
        return handle_service_error(e, 'calculate_metrics', {'start_date': start_date, 'end_date': end_date})


def get_daily_trends(
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Get daily trends with revenue, refunds, and orders grouped by date
    
    Groups data by DATE (not individual transactions) to create smooth trend lines.
    Sums revenue for each day to remove noise from spiky individual transaction data.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary with:
        - data: List of daily records with date, revenue, refunds, orders
        - revenue_trend: List of {date, value} for revenue chart (smooth, grouped by day)
        - refund_trend: List of {date, value} for refund chart (smooth, grouped by day)
        - count: Number of days
    """
    if not table_exists('sales'):
        return {
            'data': [],
            'revenue_trend': [],
            'refund_trend': [],
            'count': 0,
        }
    
    try:
        # Get column info
        column_info = execute_query("DESCRIBE sales")
        
        # Find date column
        date_columns = ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not date_col:
            return {
                'data': [],
                'revenue_trend': [],
                'refund_trend': [],
                'count': 0,
            }
        
        # Check if date column needs casting
        col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
        needs_cast = 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper()
        
        # Find revenue column
        revenue_col = None
        for col in ['revenue_amount', 'revenue_calc', 'Invoice Amount', 'revenue_in_inr']:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['transaction_type', 'Transaction Type']:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        # Find order_id column
        order_id_col = None
        for col in ['order_id', 'Invoice Number', 'Order ID']:
            if col in column_info['column_name'].values:
                order_id_col = col
                break
        
        if not revenue_col:
            return {
                'data': [],
                'revenue_trend': [],
                'refund_trend': [],
                'count': 0,
            }
        
        # Build date filter and grouping
        # CRITICAL: Use CAST to DATE for grouping (removes time component)
        # This ensures smooth trend lines with one point per day
        # Note: DuckDB doesn't have DATE() function, use CAST(... AS DATE) instead
        if needs_cast:
            date_filter = f"CAST(\"{date_col}\" AS DATE) >= '{start_date}' AND CAST(\"{date_col}\" AS DATE) <= '{end_date}'"
            date_group = f"CAST(\"{date_col}\" AS DATE)"
        else:
            date_filter = f"\"{date_col}\" >= '{start_date}' AND \"{date_col}\" <= '{end_date}'"
            date_group = f"CAST(\"{date_col}\" AS DATE)"
        
        # Query for daily metrics
        # Revenue from Shipments
        if txn_col and order_id_col:
            revenue_sql = f"""
            SELECT 
                {date_group} as date,
                COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                COALESCE(SUM(CASE WHEN "{txn_col}" = 'Refund' THEN ABS({revenue_col}) ELSE 0 END), 0) as refunds,
                COUNT(DISTINCT CASE WHEN "{txn_col}" = 'Shipment' THEN "{order_id_col}" ELSE NULL END) as orders
            FROM sales
            WHERE {date_filter}
            GROUP BY {date_group}
            ORDER BY date
            """
        elif txn_col:
            # Has transaction type but no order_id column
            revenue_sql = f"""
            SELECT 
                {date_group} as date,
                COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                COALESCE(SUM(CASE WHEN "{txn_col}" = 'Refund' THEN ABS({revenue_col}) ELSE 0 END), 0) as refunds,
                COUNT(CASE WHEN "{txn_col}" = 'Shipment' THEN 1 ELSE NULL END) as orders
            FROM sales
            WHERE {date_filter}
            GROUP BY {date_group}
            ORDER BY date
            """
        else:
            # No transaction type column - assume all positive amounts are shipments
            if order_id_col:
                revenue_sql = f"""
                SELECT 
                    {date_group} as date,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                    COALESCE(SUM(CASE WHEN {revenue_col} < 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as refunds,
                    COUNT(DISTINCT CASE WHEN {revenue_col} > 0 THEN "{order_id_col}" ELSE NULL END) as orders
                FROM sales
                WHERE {date_filter}
                GROUP BY {date_group}
                ORDER BY date
                """
            else:
                revenue_sql = f"""
                SELECT 
                    {date_group} as date,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue,
                    COALESCE(SUM(CASE WHEN {revenue_col} < 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as refunds,
                    COUNT(CASE WHEN {revenue_col} > 0 THEN 1 ELSE NULL END) as orders
                FROM sales
                WHERE {date_filter}
                GROUP BY {date_group}
                ORDER BY date
                """
        
        df = execute_query(revenue_sql)
        
        if df.empty:
            return {
                'data': [],
                'revenue_trend': [],
                'refund_trend': [],
                'count': 0,
            }
        
        # Convert to list of dictionaries
        data = []
        revenue_trend = []
        refund_trend = []
        
        for _, row in df.iterrows():
            date_str = str(row['date']).split(' ')[0]  # Extract date part if datetime
            revenue = float(row['revenue']) if pd.notna(row['revenue']) else 0.0
            refunds = float(row['refunds']) if pd.notna(row['refunds']) else 0.0
            orders = int(row['orders']) if pd.notna(row['orders']) else 0
            
            data.append({
                'date': date_str,
                'revenue': round(revenue, 2),
                'refunds': round(refunds, 2),
                'orders': orders,
            })
            
            revenue_trend.append({
                'date': date_str,
                'value': round(revenue, 2),
            })
            
            refund_trend.append({
                'date': date_str,
                'value': round(refunds, 2),
            })
        
        return {
            'data': data,
            'revenue_trend': revenue_trend,
            'refund_trend': refund_trend,
            'count': len(data),
        }
    
    except ValueError as e:
        log_error(e, 'get_daily_trends', {'start_date': start_date, 'end_date': end_date})
        return {
            'data': [],
            'revenue_trend': [],
            'refund_trend': [],
            'count': 0,
            'error': True,
            'message': f'Invalid date range: {str(e)}',
            'status': 400
        }
    except Exception as e:
        log_error(e, 'get_daily_trends', {'start_date': start_date, 'end_date': end_date})
        return {
            'data': [],
            'revenue_trend': [],
            'refund_trend': [],
            'count': 0,
            'error': True,
            'message': 'Failed to fetch daily trends. Please try again later.',
            'status': 500
        }


def get_revenue_trend(
    start_date: str,
    end_date: str,
    group_by: str = 'day'
) -> pd.DataFrame:
    """
    Get revenue trend data grouped by time period
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        group_by: Grouping period ('day', 'week', 'month')
    
    Returns:
        DataFrame with columns: date, revenue
    """
    try:
        # Determine date truncation based on grouping
        date_trunc_map = {
            'day': 'day',
            'week': 'week',
            'month': 'month',
        }
        
        trunc_period = date_trunc_map.get(group_by, 'day')
        
        # Try to find date and revenue columns
        column_info = execute_query("DESCRIBE sales")
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        revenue_columns = ['revenue_calc', 'revenue_in_inr', 'Revenue Amount']
        txn_columns = ['Transaction Type', 'transaction_type']
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        if not date_col or not revenue_col:
            return pd.DataFrame()
        
        # Build query with transaction type filter for shipments
        if txn_col:
            sql = f"""
            SELECT 
                DATE_TRUNC('{trunc_period}', "{date_col}") as date,
                SUM({revenue_col}) as revenue
            FROM sales
            WHERE "{txn_col}" = 'Shipment'
              AND "{date_col}" >= '{start_date}'
              AND "{date_col}" <= '{end_date}'
            GROUP BY DATE_TRUNC('{trunc_period}', "{date_col}")
            ORDER BY date
            """
        else:
            # No transaction type - use positive amounts
            sql = f"""
            SELECT 
                DATE_TRUNC('{trunc_period}', "{date_col}") as date,
                SUM({revenue_col}) as revenue
            FROM sales
            WHERE {revenue_col} > 0
              AND "{date_col}" >= '{start_date}'
              AND "{date_col}" <= '{end_date}'
            GROUP BY DATE_TRUNC('{trunc_period}', "{date_col}")
            ORDER BY date
            """
        
        return execute_query(sql)
    
    except ValueError as e:
        log_error(e, 'get_revenue_trend', {'start_date': start_date, 'end_date': end_date, 'group_by': group_by})
        return pd.DataFrame()
    except Exception as e:
        log_error(e, 'get_revenue_trend', {'start_date': start_date, 'end_date': end_date, 'group_by': group_by})
        return pd.DataFrame()


def get_top_products(
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metric: str = 'revenue',
) -> pd.DataFrame:
    """
    Get top products by SKU with comprehensive metrics
    
    Args:
        limit: Number of products to return (default: 50)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        metric: Sort metric (default: 'revenue')
    
    Returns:
        DataFrame with columns: sku, asin, units_sold, revenue, refund_ratio, rating, trend
    """
    if not table_exists('sales'):
        return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
    
    try:
        column_info = execute_query("DESCRIBE sales")
        sku_columns = ['Sku', 'sku', 'SKU']
        asin_columns = ['asin', 'Asin', 'ASIN', 'Amazon ASIN']
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        txn_columns = ['Transaction Type', 'transaction_type']
        order_id_columns = ['Invoice Number', 'order_id', 'Order ID']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        rating_columns = ['rating', 'Rating', 'star_rating']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        asin_col = None
        for col in asin_columns:
            if col in column_info['column_name'].values:
                asin_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        quantity_col = None
        for col in quantity_columns:
            if col in column_info['column_name'].values:
                quantity_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        rating_col = None
        for col in rating_columns:
            if col in column_info['column_name'].values:
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
        # Calculate revenue from Shipments, refunds from Refunds, units_sold from Shipments
        if txn_col:
            # Build ASIN selection
            asin_select = f'COALESCE(MAX("{asin_col}"), \'\')' if asin_col else '\'\''
            # Build rating selection
            rating_select = f'COALESCE(AVG("{rating_col}"), 0)' if rating_col else '0'
            
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
            # No transaction type - use positive amounts for shipments
            # Build ASIN selection
            asin_select = f'COALESCE(MAX("{asin_col}"), \'\')' if asin_col else '\'\''
            # Build rating selection
            rating_select = f'COALESCE(AVG("{rating_col}"), 0)' if rating_col else '0'
            
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
        
        # Calculate refund_ratio (refund_amount / revenue * 100)
        df['refund_ratio'] = df.apply(
            lambda row: round((row['refund_amount'] / row['revenue'] * 100) if row['revenue'] > 0 else 0, 1),
            axis=1
        )
        
        # Calculate trend (period-over-period growth)
        # Compare current period vs previous period (same duration, shifted back by period length)
        if start_date and end_date:
            try:
                from datetime import datetime, timedelta
                
                # Parse dates
                current_start = datetime.strptime(start_date, '%Y-%m-%d')
                current_end = datetime.strptime(end_date, '%Y-%m-%d')
                
                # Calculate period length in days
                period_length = (current_end - current_start).days + 1
                
                # Calculate previous period dates (same duration, shifted back)
                prev_end = current_start - timedelta(days=1)
                prev_start = prev_end - timedelta(days=period_length - 1)
                
                prev_start_str = prev_start.strftime('%Y-%m-%d')
                prev_end_str = prev_end.strftime('%Y-%m-%d')
                
                logger.info(f"📊 Trend calculation: Current period {start_date} to {end_date} ({period_length} days)")
                logger.info(f"📊 Previous period: {prev_start_str} to {prev_end_str}")
                
                # Build previous period date filter
                prev_date_filter = ""
                if date_col:
                    if needs_cast:
                        prev_date_filter = f'AND CAST("{date_col}" AS DATE) >= \'{prev_start_str}\' AND CAST("{date_col}" AS DATE) <= \'{prev_end_str}\''
                    else:
                        prev_date_filter = f'AND "{date_col}" >= \'{prev_start_str}\' AND "{date_col}" <= \'{prev_end_str}\''
                
                # Query previous period revenue by SKU
                if txn_col:
                    prev_sql = f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                    FROM sales
                    WHERE 1=1 {prev_date_filter}
                    GROUP BY "{sku_col}"
                    """
                else:
                    prev_sql = f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                    FROM sales
                    WHERE 1=1 {prev_date_filter}
                    GROUP BY "{sku_col}"
                    """
                
                prev_df = execute_query(prev_sql)
                logger.info(f"Previous period query returned {len(prev_df)} SKUs")
                
                if not prev_df.empty:
                    logger.debug(f"Sample previous period SKUs: {prev_df['sku'].head(5).tolist()}")
                    logger.debug(f"Sample previous period revenues: {prev_df['prev_revenue'].head(5).tolist()}")
                
                # Fallback: If no previous period data, try month-over-month comparison within current period
                use_fallback = False
                if prev_df.empty or len(prev_df) == 0:
                    logger.warning("No previous period data found. Attempting month-over-month comparison within current period.")
                    use_fallback = True
                    
                    # Split current period into two halves for comparison
                    mid_point = current_start + timedelta(days=period_length // 2)
                    first_half_end = mid_point - timedelta(days=1)
                    second_half_start = mid_point
                    
                    first_half_str = first_half_end.strftime('%Y-%m-%d')
                    second_half_start_str = second_half_start.strftime('%Y-%m-%d')
                    
                    logger.info(f"Fallback: Comparing first half ({start_date} to {first_half_str}) vs second half ({second_half_start_str} to {end_date})")
                    
                    # Build first half date filter
                    first_half_filter = ""
                    if date_col:
                        if needs_cast:
                            first_half_filter = f'AND CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{first_half_str}\''
                        else:
                            first_half_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{first_half_str}\''
                    
                    # Query first half revenue by SKU (as "previous")
                    if txn_col:
                        fallback_sql = f"""
                        SELECT 
                            "{sku_col}" as sku,
                            COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                        FROM sales
                        WHERE 1=1 {first_half_filter}
                        GROUP BY "{sku_col}"
                        """
                    else:
                        fallback_sql = f"""
                        SELECT 
                            "{sku_col}" as sku,
                            COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                        FROM sales
                        WHERE 1=1 {first_half_filter}
                        GROUP BY "{sku_col}"
                        """
                    
                    prev_df = execute_query(fallback_sql)
                    logger.info(f"Fallback query returned {len(prev_df)} SKUs")
                
                # Merge previous period data (or fallback first half data) with current period
                if not prev_df.empty and 'sku' in prev_df.columns and 'sku' in df.columns:
                    # Ensure SKU values are strings for proper matching
                    df['sku'] = df['sku'].astype(str).str.strip()
                    prev_df['sku'] = prev_df['sku'].astype(str).str.strip()
                    
                    logger.debug(f"Current df SKUs (first 5): {df['sku'].head(5).tolist()}")
                    logger.debug(f"Previous df SKUs (first 5): {prev_df['sku'].head(5).tolist()}")
                    logger.debug(f"Current df revenue (first 5): {df['revenue'].head(5).tolist()}")
                    
                    # Merge on SKU
                    df = df.merge(prev_df, on='sku', how='left')
                    
                    # Check if merge worked
                    if 'prev_revenue' not in df.columns:
                        logger.error("Merge failed: prev_revenue column not found after merge")
                        df['trend'] = 0.0
                    else:
                        df['prev_revenue'] = df['prev_revenue'].fillna(0)
                        
                        logger.debug(f"After merge - Sample: SKU={df['sku'].iloc[0] if len(df) > 0 else 'N/A'}, Current={df['revenue'].iloc[0] if len(df) > 0 else 'N/A'}, Previous={df['prev_revenue'].iloc[0] if len(df) > 0 else 'N/A'}")
                        
                        # Calculate trend: ((current - previous) / previous) * 100
                        df['trend'] = df.apply(
                            lambda row: round(
                                ((row['revenue'] - row['prev_revenue']) / row['prev_revenue'] * 100)
                                if row['prev_revenue'] > 0 else 0.0,
                                1
                            ),
                            axis=1
                        )
                        
                        # Log some sample trends
                        if len(df) > 0:
                            sample_trends = df[['sku', 'revenue', 'prev_revenue', 'trend']].head(5)
                            logger.info(f"Sample trend calculations:\n{sample_trends.to_string()}")
                        
                        # Drop temporary column
                        if 'prev_revenue' in df.columns:
                            df = df.drop(columns=['prev_revenue'])
                else:
                    # No previous period data available, set trend to 0
                    logger.warning(f"No previous period data found. prev_df empty: {prev_df.empty}, has sku column: {'sku' in prev_df.columns if not prev_df.empty else False}, df has sku: {'sku' in df.columns}")
                    df['trend'] = 0.0
            except Exception as e:
                logger.error(f"Error calculating trend: {e}")
                import traceback
                logger.error(traceback.format_exc())
                df['trend'] = 0.0
        else:
            # No date range provided, cannot calculate trend
            logger.warning(f"No date range provided for trend calculation. start_date: {start_date}, end_date: {end_date}")
            df['trend'] = 0.0
        
        # Ensure rating is numeric and rounded
        if 'rating' in df.columns:
            df['rating'] = df['rating'].fillna(0).apply(lambda x: round(float(x), 1))
        else:
            df['rating'] = 0.0
        
        # Round numeric columns
        df['units_sold'] = df['units_sold'].fillna(0).astype(int)
        df['revenue'] = df['revenue'].fillna(0).round(2)
        
        # Select and order columns
        result_df = df[['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend']].copy()
        
        return result_df
    
    except ValueError as e:
        log_error(e, 'get_top_products', {'limit': limit, 'start_date': start_date, 'end_date': end_date, 'metric': metric})
        return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
    except Exception as e:
        log_error(e, 'get_top_products', {'limit': limit, 'start_date': start_date, 'end_date': end_date, 'metric': metric})
        return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])


def get_revenue_by_city(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get top cities by total revenue for a specific date range.
    
    Uses "Ship To City" column with case-insensitive normalization.
    Handles city name variations (Bangalore/Bengaluru, Delhi/New Delhi, etc.).
    
    Args:
        start_date: Start date (YYYY-MM-DD) - if None, uses all-time data
        end_date: End date (YYYY-MM-DD) - if None, uses all-time data
        limit: Number of cities to return (default: 10)
    
    Returns:
        Dictionary with:
        - data: List of {"city": "...", "revenue": 100000}
        - count: Number of cities returned
        - total_revenue: Sum of all city revenues
    """
    if not table_exists('sales'):
        return {
            "data": [],
            "count": 0,
            "total_revenue": 0.0
        }
    
    try:
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        available_cols = list(column_info['column_name'].values)
        
        logger.debug(f"Available columns in sales table: {available_cols}")
        
        # Check if both columns exist
        has_ship_to_city = any(col.lower().strip() == 'ship to city' for col in available_cols)
        has_bill_from_city = any(col.lower().strip() == 'bill from city' for col in available_cols)
        
        # Find city/region column (try multiple variations)
        # PRIORITY: "Ship To City" is the correct column (customer delivery location)
        # EXCLUDE: "Bill From City" (this is wrong - it's the seller location)
        city_col = None
        
        # FIRST PRIORITY: "Ship To City" (exact case-insensitive match)
        for col in available_cols:
            col_lower = col.lower().strip()
            if col_lower == 'ship to city':
                city_col = col
                logger.info(f"Found correct city column: '{city_col}'")
                break
        
        # If not found, try "region" column (standardized name)
        if not city_col:
            for col in available_cols:
                if col.lower() == 'region':
                    city_col = col
                    logger.info(f"Found city column (region): '{city_col}'")
                    break
        
        # If still not found, try other variations (but EXCLUDE "Bill From City")
        if not city_col:
            region_variations = ['city', 'City', 'location', 'Location', 'shipping_city', 'Shipping City', 
                               'delivery_city', 'Delivery City', 'ship_to_city', 'ship_to_state', 
                               'Ship To City', 'Ship To State', 'ship to state']
            for col in region_variations:
                if col in available_cols:
                    if col.lower() != 'bill from city':
                        city_col = col
                        logger.info(f"Found city column (variation): '{city_col}'")
                        break
        
        # Last resort: Pattern matching (but EXCLUDE "Bill From City")
        if not city_col:
            for col in available_cols:
                col_lower = col.lower()
                if col_lower == 'bill from city':
                    continue
                if any(keyword in col_lower for keyword in ['city', 'region', 'location', 'state', 'ship']):
                    col_type = column_info[column_info['column_name'] == col]['column_type'].values[0]
                    if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                        city_col = col
                        logger.info(f"Found city column by pattern matching: '{city_col}'")
                        break
        
        if not city_col:
            logger.warning(f"City column not found. Available columns: {available_cols}")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # Find revenue column
        revenue_col = None
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        for col in revenue_columns:
            if col in available_cols:
                revenue_col = col
                logger.info(f"Found revenue column: '{revenue_col}'")
                break
        
        if not revenue_col:
            logger.warning(f"Revenue column not found. Available columns: {available_cols}")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # Find date column for filtering
        date_col = None
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        for col in date_columns:
            if col in available_cols:
                date_col = col
                logger.info(f"Found date column: '{date_col}'")
                break
        
        # Build date filter if dates are provided
        date_filter = ""
        if start_date and end_date and date_col:
            # Handle date column type (VARCHAR vs DATE)
            needs_cast = False
            if date_col:
                col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
                if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                    needs_cast = True
            
            if needs_cast:
                date_filter = f'AND CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\''
            else:
                date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
            logger.info(f"Applying date filter: {start_date} to {end_date}")
        elif start_date or end_date:
            logger.warning("Both start_date and end_date must be provided for date filtering")
        
        # DEBUG: Check if we have any data
        count_sql = f'SELECT COUNT(*) as total FROM sales WHERE "{city_col}" IS NOT NULL AND TRIM("{city_col}") != \'\'{date_filter}'
        count_df = execute_query(count_sql)
        total_records = int(count_df['total'].iloc[0]) if not count_df.empty else 0
        logger.info(f"Total records with city data: {total_records}")
        
        if total_records == 0:
            logger.warning("No records found with city data")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
        }
        
        # CRITICAL VERIFICATION: Ensure we're using "Ship To City" not "Bill From City"
        if city_col.lower() == 'bill from city':
            logger.error("Selected column is 'Bill From City' - should use 'Ship To City' instead")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        logger.info(f"Using city column: '{city_col}' for revenue by city calculation")
        
        # Fetch data with date filter if provided
        # Use ABS() for revenue to handle both positive and negative values
        sql = f"""
        SELECT 
            TRIM("{city_col}") as raw_city,
            ABS({revenue_col}) as revenue
        FROM sales
        WHERE "{city_col}" IS NOT NULL 
            AND TRIM("{city_col}") != ''
            AND {revenue_col} IS NOT NULL
            AND {revenue_col} != 0
            {date_filter}
        """
        
        raw_df = execute_query(sql)
        
        logger.info(f"Query returned {len(raw_df)} rows")
        
        if raw_df.empty:
            logger.warning("No city data found after query")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # Normalize city names using shared function
        # This ensures consistent matching with get_skus_by_city()
        raw_df['city'] = raw_df['raw_city'].apply(normalize_city_name)
        raw_df = raw_df[raw_df['city'].notna() & (raw_df['city'] != '')]
        
        # Group by normalized city and sum revenue
        city_revenue = raw_df.groupby('city')['revenue'].sum().reset_index()
        
        # Filter out zero/negative revenue
        city_revenue = city_revenue[city_revenue['revenue'] > 0]
        
        # Sort by revenue descending and get top N
        top_cities = city_revenue.sort_values('revenue', ascending=False).head(limit)
        
        # Convert to list of dicts
        cities_list = top_cities.to_dict('records')
        
        # Format revenue as float
        for city in cities_list:
            city['revenue'] = float(city['revenue'])
        
        total_revenue = float(top_cities['revenue'].sum())
        
        logger.info(f"Returning {len(cities_list)} cities, total revenue: ₹{total_revenue:,.2f}")
        
        return {
            "data": cities_list,
            "count": len(cities_list),
            "total_revenue": total_revenue
        }
    
    except ValueError as e:
        log_error(e, 'get_revenue_by_city', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {
            "data": [],
            "count": 0,
            "total_revenue": 0.0,
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_revenue_by_city', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {
            "data": [],
            "count": 0,
            "total_revenue": 0.0,
            "error": True,
            "message": "Failed to fetch revenue by city. Please try again later.",
            "status": 500
        }


def get_skus_by_city(city: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get top SKUs for a specific city.
    
    Shows units sold and revenue for each SKU.
    Uses all-time data (no date filters).
    
    Args:
        city: City name (will be normalized)
        limit: Number of SKUs to return (default: 10)
    
    Returns:
        Dictionary with:
        - city: Normalized city name
        - data: List of {"sku": "...", "asin": "...", "units": 100, "revenue": 50000}
        - count: Number of SKUs returned
    """
    if not table_exists('sales'):
        return {
            "city": city,
            "data": [],
            "count": 0
        }
    
    try:
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        available_cols = list(column_info['column_name'].values)
        
        logger.debug(f"Available columns in sales table (get_skus_by_city): {available_cols}")
        
        # Find city/region column
        city_col = None
        
        # FIRST PRIORITY: "Ship To City"
        for col in available_cols:
            if col.lower() == 'ship to city':
                city_col = col
                logger.info(f"Found city column (Ship To City): '{city_col}'")
                break
        
        # Try "region" column
        if not city_col:
            for col in available_cols:
                if col.lower() == 'region':
                    city_col = col
                    logger.info(f"Found city column (region): '{city_col}'")
                    break
        
        # Try other variations (but EXCLUDE "Bill From City")
        if not city_col:
            region_variations = ['city', 'City', 'location', 'Location', 'shipping_city', 'Shipping City', 
                               'delivery_city', 'Delivery City', 'ship_to_city', 'ship_to_state', 
                               'Ship To City', 'Ship To State', 'ship to state']
            for col in region_variations:
                if col in available_cols and col.lower() != 'bill from city':
                    city_col = col
                    logger.info(f"Found city column (variation): '{city_col}'")
                    break
        
        # Last resort: Pattern matching
        if not city_col:
            for col in available_cols:
                col_lower = col.lower()
                if col_lower == 'bill from city':
                    continue
                if any(keyword in col_lower for keyword in ['city', 'region', 'location', 'state', 'ship']):
                    col_type = column_info[column_info['column_name'] == col]['column_type'].values[0]
                    if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                        city_col = col
                        logger.info(f"Found city column by pattern matching: '{city_col}'")
                        break
        
        if not city_col:
            logger.warning(f"City column not found. Available columns: {available_cols}")
            return {
                "city": city,
                "data": [],
                "count": 0
            }
        
        sku_col = None
        for col in ['sku', 'Sku', 'SKU']:
            if col in available_cols:
                sku_col = col
                break
        
        asin_col = None
        for col in ['asin', 'Asin', 'ASIN']:
            if col in available_cols:
                asin_col = col
                break
        
        revenue_col = None
        for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']:
            if col in available_cols:
                revenue_col = col
                break
        
        quantity_col = None
        for col in ['quantity', 'Quantity', 'units_sold']:
            if col in available_cols:
                quantity_col = col
                break
        
        if not city_col or not sku_col or not revenue_col:
            logger.warning(f"Required columns not found. city_col: {city_col}, sku_col: {sku_col}, revenue_col: {revenue_col}")
            return {
                "city": city,
                "data": [],
                "count": 0
            }
        
        # Normalize input city name
        city_normalized = normalize_city_name(city)
        
        logger.info(f"Getting SKUs for city: '{city}' -> normalized: '{city_normalized}'")
        
        # Build SQL query
        # We need to normalize the city column in SQL to match the normalized input city
        # Since we normalize in Python, we'll fetch all cities matching the normalized name
        # and then filter in Python to ensure exact match
        
        asin_select = f'COALESCE(MAX("{asin_col}"), \'\')' if asin_col else '\'\''
        quantity_select = f'COALESCE(SUM({quantity_col}), 0)' if quantity_col else 'COUNT(*)'
        
        # Fetch all records for cities that might match (case-insensitive)
        # We'll normalize and filter in Python
        sql = f"""
        SELECT 
            "{sku_col}" as sku,
            {asin_select} as asin,
            {quantity_select} as units,
            COALESCE(SUM({revenue_col}), 0) as revenue,
            TRIM("{city_col}") as raw_city
        FROM sales
        WHERE "{city_col}" IS NOT NULL
            AND "{sku_col}" IS NOT NULL
            AND {revenue_col} != 0
        GROUP BY "{sku_col}", TRIM("{city_col}")
        HAVING revenue > 0
        ORDER BY units DESC
        LIMIT {limit * 5}
        """
        
        df = execute_query(sql)
        logger.info(f"Query returned {len(df)} rows for city '{city_normalized}'")
        
        if df.empty:
            logger.warning(f"No SKU data found for city: {city_normalized}")
            return {
                "city": city_normalized,
                "data": [],
                "count": 0
            }
        
        # Normalize city names in the result
        df['city_normalized'] = df['raw_city'].apply(normalize_city_name)
        
        city_data = df[df['city_normalized'] == city_normalized]
        
        if city_data.empty:
            logger.warning(f"No SKU data found for normalized city: {city_normalized}")
            # Try case-insensitive match as fallback
            city_data_fallback = df[df['city_normalized'].str.lower() == city_normalized.lower()] if 'city_normalized' in df.columns else pd.DataFrame()
            if not city_data_fallback.empty:
                city_data = city_data_fallback
            else:
                return {
                    "city": city_normalized,
                    "data": [],
                    "count": 0
                }
        
        logger.info(f"Found {len(city_data)} rows for city '{city_normalized}'")
        
        # Group by SKU (in case same SKU appears multiple times)
        sku_summary = city_data.groupby('sku').agg({
            'asin': 'first',
            'units': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        # Sort by units descending and get top N
        top_skus = sku_summary.sort_values('units', ascending=False).head(limit)
        
        # Convert to list of dicts
        skus_list = top_skus.to_dict('records')
        
        # Format values
        for sku in skus_list:
            sku['sku'] = str(sku.get('sku', ''))
            sku['asin'] = str(sku.get('asin', ''))
            sku['units'] = int(sku.get('units', 0))
            sku['revenue'] = float(sku.get('revenue', 0))
        
        logger.info(f"Returning {len(skus_list)} SKUs for city '{city_normalized}'")
        
        return {
            "city": city_normalized,
            "data": skus_list,
            "count": len(skus_list)
        }
    
    except ValueError as e:
        log_error(e, 'get_skus_by_city', {'city': city, 'limit': limit})
        return {
            "city": city,
            "data": [],
            "count": 0,
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_skus_by_city', {'city': city, 'limit': limit})
        return {
            "city": city,
            "data": [],
            "count": 0,
            "error": True,
            "message": "Failed to fetch SKUs by city. Please try again later.",
            "status": 500
        }


def get_movers_decliners(
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get movers (fast growing) and decliners (declining) SKUs using adaptive moving average comparison.
    
    Automatically determines granularity based on date range:
    - 7-13 days: Daily comparison (last day vs daily average)
    - 14-89 days: Weekly comparison (last week vs weekly average)
    - 90+ days: Monthly comparison (last month vs monthly average)
    
    Logic:
    1. Split selected date range into periods (days/weeks/months)
    2. For each SKU: calculate baseline = average of all periods EXCEPT last
    3. Compare last period to baseline
    4. Growth % = ((last - baseline) / baseline) × 100
    5. Movers: growth >= +30%, Decliners: growth <= -30%
    
    Args:
        start_date: Start date of period (YYYY-MM-DD)
        end_date: End date of period (YYYY-MM-DD)
        limit: Number of SKUs to return per category (default: 10)
    
    Returns:
        Dictionary with:
        - movers: List of SKUs with growth >= +30%
        - decliners: List of SKUs with growth <= -30%
        - label: Comparison label (e.g., "Last Week vs Weekly Avg")
        - granularity: Period granularity ("daily", "weekly", or "monthly")
        Each item: {"sku": "...", "revenue": 100000, "growth": 35.5}
    """
    if not table_exists('sales'):
        return {
            "movers": [],
            "decliners": [],
            "label": "",
            "granularity": ""
        }
    
    try:
        from datetime import datetime, timedelta
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate total days in range
        total_days = (end_dt - start_dt).days + 1
        
        # Validate minimum date range
        if total_days < 7:
            logger.warning(f"Date range too short ({total_days} days). Minimum 7 days required.")
            return {
                "movers": [],
                "decliners": [],
                "label": f"Date range too short ({total_days} days)",
                "granularity": ""
            }
        
        # Determine granularity based on date range
        if total_days <= 13:
            granularity = "daily"
            period_label = "Last Day vs Daily Avg"
        elif total_days <= 89:
            granularity = "weekly"
            period_label = "Last Week vs Weekly Avg"
        else:
            granularity = "monthly"
            period_label = "Last Month vs Monthly Avg"
        
        logger.info(f"📊 Movers & Decliners: Date range {start_date} to {end_date} ({total_days} days)")
        logger.info(f"📊 Granularity: {granularity} | Label: {period_label}")
        
        # Split date range into periods based on granularity
        periods = []
        if granularity == "daily":
            # Each day is a period
            current_date = start_dt
            while current_date <= end_dt:
                periods.append({
                    'start': current_date,
                    'end': current_date
                })
                current_date += timedelta(days=1)
        elif granularity == "weekly":
            # Each week is a period (7 days)
            current_date = start_dt
            while current_date <= end_dt:
                period_end = min(current_date + timedelta(days=6), end_dt)
                periods.append({
                    'start': current_date,
                    'end': period_end
                })
                current_date = period_end + timedelta(days=1)
        else:  # monthly
            # Each month is a period
            current_date = start_dt
            while current_date <= end_dt:
                # Calculate end of month
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                period_end = min(next_month - timedelta(days=1), end_dt)
                periods.append({
                    'start': current_date,
                    'end': period_end
                })
                current_date = period_end + timedelta(days=1)
        
        if len(periods) < 2:
            logger.warning(f"Not enough periods ({len(periods)}). Need at least 2 periods for comparison.")
            return {
                "movers": [],
                "decliners": [],
                "label": period_label,
                "granularity": granularity
            }
        
        logger.info(f"📊 Split into {len(periods)} periods")
        logger.info(f"📊 Last period: {periods[-1]['start'].strftime('%Y-%m-%d')} to {periods[-1]['end'].strftime('%Y-%m-%d')}")
        logger.info(f"📊 Baseline periods: {len(periods) - 1} periods (excluding last)")
        
        # Detect column names dynamically
        column_info = execute_query("DESCRIBE sales")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not sku_col or not revenue_col or not date_col:
            logger.warning("Required columns not found for movers/decliners")
            logger.warning(f"sku_col: {sku_col}, revenue_col: {revenue_col}, date_col: {date_col}")
            logger.warning(f"Available columns: {list(column_info['column_name'].values)}")
            return {
                "movers": [],
                "decliners": [],
                "label": period_label,
                "granularity": granularity
            }
        
        logger.info(f"Using columns - SKU: {sku_col}, Revenue: {revenue_col}, Transaction: {txn_col}, Date: {date_col}")
        
        # Handle date column type (VARCHAR vs DATE)
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        # Helper function to build date filter
        def build_date_filter(period_start, period_end):
            start_str = period_start.strftime('%Y-%m-%d')
            end_str = period_end.strftime('%Y-%m-%d')
            if needs_cast:
                return f'CAST("{date_col}" AS DATE) >= \'{start_str}\' AND CAST("{date_col}" AS DATE) <= \'{end_str}\''
            else:
                return f'"{date_col}" >= \'{start_str}\' AND "{date_col}" <= \'{end_str}\''
        
        # Helper function to build revenue query for a period
        def build_revenue_query(period_start, period_end):
            date_filter = build_date_filter(period_start, period_end)
            if txn_col:
                if revenue_col == 'revenue_calc':
                    return f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' AND {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as revenue
                    FROM sales
                    WHERE {date_filter}
                    GROUP BY "{sku_col}"
                    """
                else:
                    return f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue
                    FROM sales
                    WHERE {date_filter}
                    GROUP BY "{sku_col}"
                    """
            else:
                if revenue_col == 'revenue_calc':
                    return f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as revenue
                    FROM sales
                    WHERE {revenue_col} > 0 AND {date_filter}
                    GROUP BY "{sku_col}"
                    """
                else:
                    return f"""
                    SELECT 
                        "{sku_col}" as sku,
                        COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue
                    FROM sales
                    WHERE {revenue_col} > 0 AND {date_filter}
                    GROUP BY "{sku_col}"
                    """
        
        # Fetch revenue for each period
        period_revenues = {}
        for i, period in enumerate(periods):
            period_start = period['start']
            period_end = period['end']
            sql = build_revenue_query(period_start, period_end)
            
            logger.info(f"Fetching revenue for period {i+1}/{len(periods)}: {period_start.strftime('%Y-%m-%d')} to {period_end.strftime('%Y-%m-%d')}")
            period_df = execute_query(sql)
            
            # Ensure SKU values are strings
            if not period_df.empty:
                period_df['sku'] = period_df['sku'].astype(str).str.strip()
                period_df = period_df[period_df['revenue'] > 0]  # Filter zero revenue
            
            period_revenues[i] = period_df
            logger.info(f"Period {i+1} returned {len(period_df)} SKUs with revenue > 0")
        
        # Separate last period from baseline periods
        last_period_idx = len(periods) - 1
        last_period_df = period_revenues[last_period_idx].copy() if last_period_idx in period_revenues and not period_revenues[last_period_idx].empty else pd.DataFrame()
        
        if last_period_df.empty:
            logger.warning("Last period has no revenue data")
            return {
                "movers": [],
                "decliners": [],
                "label": period_label,
                "granularity": granularity
            }
        
        # Calculate baseline average for each SKU (average of all periods EXCEPT last)
        baseline_periods = [period_revenues[i] for i in range(last_period_idx) if i in period_revenues and not period_revenues[i].empty]
        
        if not baseline_periods:
            logger.warning("No baseline periods with data")
            return {
                "movers": [],
                "decliners": [],
                "label": period_label,
                "granularity": granularity
            }
        
        # Combine all baseline periods and calculate average revenue per SKU
        baseline_df = pd.concat(baseline_periods, ignore_index=True)
        baseline_avg = baseline_df.groupby('sku')['revenue'].mean().reset_index()
        baseline_avg.columns = ['sku', 'baseline_revenue']
        
        logger.info(f"Baseline calculated from {len(baseline_periods)} periods")
        logger.info(f"Baseline contains {len(baseline_avg)} unique SKUs")
        
        # Merge last period with baseline
        last_period_df.columns = ['sku', 'last_revenue']
        merged_df = last_period_df.merge(baseline_avg, on='sku', how='inner')
        
        logger.info(f"Merged dataframe has {len(merged_df)} SKUs with both last period and baseline data")
        
        # Calculate growth percentage: ((last - baseline) / baseline) × 100
        def calculate_growth(row):
            last = float(row['last_revenue'])
            baseline = float(row['baseline_revenue'])
            
            if baseline > 0:
                return round(((last - baseline) / baseline) * 100, 1)
            elif last > 0:
                # New product in last period (no baseline)
                return 999.0  # Indicates new product
            else:
                return 0.0
        
        merged_df['wow_change'] = merged_df.apply(calculate_growth, axis=1)
        
        logger.info(f"Growth calculated for {len(merged_df)} SKUs")
        logger.info(f"Growth range: {merged_df['wow_change'].min():.1f}% to {merged_df['wow_change'].max():.1f}%")
        
        # Filter movers (growth >= +30%)
        movers_df = merged_df[merged_df['wow_change'] >= 30].copy()
        movers_df = movers_df.sort_values('wow_change', ascending=False).head(limit)
        
        # Filter decliners (growth <= -30%)
        decliners_df = merged_df[merged_df['wow_change'] <= -30].copy()
        decliners_df = decliners_df.sort_values('wow_change', ascending=True).head(limit)
        
        logger.info(f"Found {len(movers_df)} movers and {len(decliners_df)} decliners")
        
        # Format results
        movers = []
        for _, row in movers_df.iterrows():
            wow_change = float(row['wow_change'])
            # Cap at 999% for display
            if wow_change >= 999:
                wow_change = 999.0
            movers.append({
                'sku': str(row['sku']),
                'revenue': float(row['last_revenue']),
                'wow_change': wow_change,
            })
        
        decliners = []
        for _, row in decliners_df.iterrows():
            wow_change = float(row['wow_change'])
            # Cap at -100% for display
            wow_change = max(wow_change, -100.0)
            decliners.append({
                'sku': str(row['sku']),
                'revenue': float(row['last_revenue']),
                'wow_change': wow_change,
            })
        
        logger.info(f"📊 Final results: {len(movers)} movers and {len(decliners)} decliners")
        
        return {
            "movers": movers,
            "decliners": decliners,
            "label": period_label,
            "granularity": granularity
        }
    
    except ValueError as e:
        log_error(e, 'get_movers_decliners', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {
            "movers": [],
            "decliners": [],
            "label": "Error",
            "granularity": "day",
            "error": True,
            "message": f"Invalid date range: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_movers_decliners', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {
            "movers": [],
            "decliners": [],
            "label": "Error",
            "granularity": "day",
            "error": True,
            "message": "Failed to fetch movers and decliners. Please try again later.",
            "status": 500
        }


def get_top_products_performance(
    start_date: str,
    end_date: str,
    view_type: str = 'monthly',
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get top products performance tracker with period-by-period breakdown.
    
    Shows top products by total volume with monthly or quarterly performance tracking.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        view_type: 'monthly' or 'quarterly' (default: 'monthly')
        limit: Number of top products to return (default: 10)
    
    Returns:
        Dictionary with:
        - products: List of product performance data
        - period_labels: List of period labels (e.g., ["Jul", "Aug", "Sep"])
        - view_type: The view type used
        Each product: {
            "sku": "...",
            "periods": [145, 150, 139],  # Volume for each period
            "growth_rates": [None, 3.4, -7.3],  # Period-over-period growth %
            "total_volume": 434
        }
    """
    if not table_exists('sales'):
        return {
            "products": [],
            "period_labels": [],
            "view_type": view_type
        }
    
    try:
        from datetime import datetime, timedelta
        
        # Parse dates
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        quantity_col = None
        for col in quantity_columns:
            if col in column_info['column_name'].values:
                quantity_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not sku_col or not date_col:
            logger.warning("Required columns not found for top products performance")
            return {
                "products": [],
                "period_labels": [],
                "view_type": view_type
            }
        
        # Handle date column type
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        # Split date range into periods
        periods = []
        period_labels = []
        
        if view_type == 'monthly':
            current_date = start_dt
            while current_date <= end_dt:
                # Calculate end of month
                if current_date.month == 12:
                    next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_month = current_date.replace(month=current_date.month + 1, day=1)
                period_end = min(next_month - timedelta(days=1), end_dt)
                
                periods.append({
                    'start': current_date,
                    'end': period_end
                })
                # Format label as "Jul", "Aug", etc.
                period_labels.append(current_date.strftime('%b'))
                
                current_date = period_end + timedelta(days=1)
        else:  # quarterly
            current_date = start_dt
            quarter = 1
            while current_date <= end_dt:
                # Calculate end of quarter
                if current_date.month <= 3:
                    quarter_end_month = 3
                elif current_date.month <= 6:
                    quarter_end_month = 6
                elif current_date.month <= 9:
                    quarter_end_month = 9
                else:
                    quarter_end_month = 12
                
                if quarter_end_month == 12:
                    quarter_end = datetime(current_date.year, 12, 31)
                else:
                    quarter_end = datetime(current_date.year, quarter_end_month + 1, 1) - timedelta(days=1)
                
                period_end = min(quarter_end, end_dt)
                
                periods.append({
                    'start': current_date,
                    'end': period_end
                })
                # Format label as "Q1", "Q2", etc.
                if current_date.month <= 3:
                    q_num = 1
                elif current_date.month <= 6:
                    q_num = 2
                elif current_date.month <= 9:
                    q_num = 3
                else:
                    q_num = 4
                period_labels.append(f"Q{q_num}")
                
                current_date = period_end + timedelta(days=1)
        
        if not periods:
            return {
                "products": [],
                "period_labels": [],
                "view_type": view_type
            }
        
        logger.info(f"Split date range into {len(periods)} {view_type} periods")
        
        # Helper function to build date filter
        def build_date_filter(period_start, period_end):
            start_str = period_start.strftime('%Y-%m-%d')
            end_str = period_end.strftime('%Y-%m-%d')
            if needs_cast:
                return f'CAST("{date_col}" AS DATE) >= \'{start_str}\' AND CAST("{date_col}" AS DATE) <= \'{end_str}\''
            else:
                return f'"{date_col}" >= \'{start_str}\' AND "{date_col}" <= \'{end_str}\''
        
        # Fetch volume for each period by SKU
        period_data = {}
        all_skus = set()
        
        logger.info(f"Fetching volume data for {len(periods)} periods")
        logger.debug(f"Using columns - SKU: {sku_col}, Date: {date_col}, Transaction: {txn_col}, Quantity: {quantity_col}")
        
        for i, period in enumerate(periods):
            date_filter = build_date_filter(period['start'], period['end'])
            
            if txn_col and quantity_col:
                sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({quantity_col}) ELSE 0 END), 0) as volume
                FROM sales
                WHERE {date_filter}
                GROUP BY "{sku_col}"
                HAVING volume > 0
                """
            elif quantity_col:
                sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(ABS({quantity_col})), 0) as volume
                FROM sales
                WHERE {quantity_col} > 0 AND {date_filter}
                GROUP BY "{sku_col}"
                HAVING volume > 0
                """
            else:
                # Fallback: count shipments
                if txn_col:
                    sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COUNT(*) as volume
                FROM sales
                WHERE "{txn_col}" = 'Shipment' AND {date_filter}
                GROUP BY "{sku_col}"
                """
                else:
                    sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COUNT(*) as volume
                FROM sales
                WHERE {date_filter}
                GROUP BY "{sku_col}"
                """
            
            period_df = execute_query(sql)
            if not period_df.empty:
                period_df['sku'] = period_df['sku'].astype(str).str.strip()
                period_data[i] = period_df.set_index('sku')['volume'].to_dict()
                all_skus.update(period_df['sku'].tolist())
                logger.debug(f"Period {i+1} ({period_labels[i]}): Found {len(period_df)} SKUs with data")
            else:
                logger.debug(f"Period {i+1} ({period_labels[i]}): No data found")
        
        logger.info(f"Total unique SKUs across all periods: {len(all_skus)}")
        
        # Calculate total volume for each SKU across all periods
        sku_totals = {}
        for sku in all_skus:
            total = sum(period_data[i].get(sku, 0) for i in range(len(periods)))
            if total > 0:
                sku_totals[sku] = total
        
        logger.info(f"SKUs with total volume > 0: {len(sku_totals)}")
        
        # Get top N products by total volume
        top_skus = sorted(sku_totals.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Build product performance data
        products = []
        for sku, total_volume in top_skus:
            periods_volumes = []
            growth_rates = []
            
            for i in range(len(periods)):
                volume = period_data[i].get(sku, 0)
                periods_volumes.append(int(volume))
                
                # Calculate growth rate (period-over-period)
                if i > 0:
                    prev_volume = periods_volumes[i - 1]
                    if prev_volume > 0:
                        growth = ((volume - prev_volume) / prev_volume) * 100
                        growth_rates.append(round(growth, 1))
                    else:
                        growth_rates.append(None if volume == 0 else 999.0)  # New product
                else:
                    growth_rates.append(None)  # First period has no previous
            
            products.append({
                'sku': sku,
                'periods': periods_volumes,
                'growth_rates': growth_rates,
                'total_volume': int(total_volume)
            })
        
        logger.info(f"Returning {len(products)} top products with {len(period_labels)} periods")
        
        return {
            "products": products,
            "period_labels": period_labels,
            "view_type": view_type
        }
    
    except ValueError as e:
        log_error(e, 'get_top_products_performance', {'start_date': start_date, 'end_date': end_date, 'view_type': view_type})
        return {
            "products": [],
            "period_labels": [],
            "view_type": view_type,
            "error": True,
            "message": f"Invalid input: {str(e)}",
            "status": 400
        }
    except Exception as e:
        log_error(e, 'get_top_products_performance', {'start_date': start_date, 'end_date': end_date, 'view_type': view_type})
        return {
            "products": [],
            "period_labels": [],
            "view_type": view_type,
            "error": True,
            "message": "Failed to fetch top products performance. Please try again later.",
            "status": 500
        }


def get_refunds_data(
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get refunds data for Product Quality Issues dashboard.
    
    Calculates refund percentage and lost revenue per SKU.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of products to return (default: 10)
    
    Returns:
        Dictionary with:
        - data: List of products with refund metrics
        Each product: {
            "sku": "...",
            "units_sold": 100,
            "refunds": 5,
            "refund_percentage": 5.0,
            "lost_revenue": 5000.0
        }
    """
    if not table_exists('sales'):
        return {"data": []}
    
    try:
        from datetime import datetime
        
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        if not sku_col or not txn_col or not date_col:
            logger.warning("Required columns not found for refunds data")
            return {"data": []}
        
        # Handle date column type
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        date_filter = f'CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\'' if needs_cast else f'"{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        # Query: Get shipments count per SKU
        shipments_sql = f"""
        SELECT 
            "{sku_col}" as sku,
            COUNT(*) as units_sold
            FROM sales
        WHERE "{txn_col}" = 'Shipment' AND {date_filter}
        GROUP BY "{sku_col}"
        """
        
        shipments_df = execute_query(shipments_sql)
        
        # Query: Get refunds count and lost revenue per SKU
        if revenue_col:
            refunds_sql = f"""
            SELECT 
                "{sku_col}" as sku,
                COUNT(*) as refunds,
                COALESCE(SUM(ABS({revenue_col})), 0) as lost_revenue
            FROM sales
            WHERE "{txn_col}" = 'Refund' AND {date_filter}
            GROUP BY "{sku_col}"
            """
        else:
            refunds_sql = f"""
            SELECT 
                "{sku_col}" as sku,
                COUNT(*) as refunds,
                0 as lost_revenue
            FROM sales
            WHERE "{txn_col}" = 'Refund' AND {date_filter}
            GROUP BY "{sku_col}"
            """
        
        refunds_df = execute_query(refunds_sql)
        
        # Merge dataframes
        if shipments_df.empty:
            return {"data": []}
        
        shipments_df['sku'] = shipments_df['sku'].astype(str).str.strip()
        if not refunds_df.empty:
            refunds_df['sku'] = refunds_df['sku'].astype(str).str.strip()
            merged_df = shipments_df.merge(refunds_df, on='sku', how='left')
        else:
            merged_df = shipments_df.copy()
            merged_df['refunds'] = 0
            merged_df['lost_revenue'] = 0.0
        
        merged_df['refunds'] = merged_df['refunds'].fillna(0).astype(int)
        merged_df['lost_revenue'] = merged_df['lost_revenue'].fillna(0.0).astype(float)
        
        # Calculate refund percentage
        merged_df['refund_percentage'] = merged_df.apply(
            lambda row: (row['refunds'] / row['units_sold'] * 100) if row['units_sold'] > 0 else 0.0,
            axis=1
        )
        
        # Filter products with refunds > 0 and sort by refund count (descending)
        result_df = merged_df[merged_df['refunds'] > 0].copy()
        result_df = result_df.sort_values('refunds', ascending=False).head(limit)
        
        # Format results
        data = []
        for _, row in result_df.iterrows():
            data.append({
                'sku': str(row['sku']),
                'units_sold': int(row['units_sold']),
                'refunds': int(row['refunds']),
                'refund_percentage': round(float(row['refund_percentage']), 2),
                'lost_revenue': round(float(row['lost_revenue']), 2)
            })
        
        logger.info(f"Returning {len(data)} products with refund data")
        
        return {"data": data}
    
    except ValueError as e:
        log_error(e, 'get_refunds_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": f"Invalid input: {str(e)}", "status": 400}
    except Exception as e:
        log_error(e, 'get_refunds_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": "Failed to fetch refunds data. Please try again later.", "status": 500}


def get_cancellations_data(
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get cancellations data for Product Quality Issues dashboard.
    
    Calculates cancellation percentage per SKU.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of products to return (default: 10)
    
    Returns:
        Dictionary with:
        - data: List of products with cancellation metrics
        Each product: {
            "sku": "...",
            "units_ordered": 100,
            "cancelled": 10,
            "cancel_percentage": 10.0
        }
    """
    if not table_exists('sales'):
        return {"data": []}
    
    try:
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not sku_col or not txn_col or not date_col:
            logger.warning("Required columns not found for cancellations data")
            return {"data": []}
        
        # Handle date column type
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        date_filter = f'CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\'' if needs_cast else f'"{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        # Query: Get shipments count per SKU
        shipments_sql = f"""
        SELECT 
            "{sku_col}" as sku,
            COUNT(*) as shipments
            FROM sales
        WHERE "{txn_col}" = 'Shipment' AND {date_filter}
        GROUP BY "{sku_col}"
        """
        
        shipments_df = execute_query(shipments_sql)
        
        # Query: Get cancellations count per SKU
        cancellations_sql = f"""
        SELECT 
            "{sku_col}" as sku,
            COUNT(*) as cancelled
            FROM sales
        WHERE "{txn_col}" = 'Cancel' AND {date_filter}
        GROUP BY "{sku_col}"
        """
        
        cancellations_df = execute_query(cancellations_sql)
        
        # Merge dataframes
        if shipments_df.empty and cancellations_df.empty:
            return {"data": []}
        
        # Combine shipments and cancellations to get units_ordered
        if not shipments_df.empty:
            shipments_df['sku'] = shipments_df['sku'].astype(str).str.strip()
            merged_df = shipments_df.copy()
        else:
            # If no shipments, create empty dataframe with SKU column
            if not cancellations_df.empty:
                cancellations_df['sku'] = cancellations_df['sku'].astype(str).str.strip()
                merged_df = pd.DataFrame({'sku': cancellations_df['sku'].unique(), 'shipments': 0})
            else:
                return {"data": []}
        
        if not cancellations_df.empty:
            cancellations_df['sku'] = cancellations_df['sku'].astype(str).str.strip()
            merged_df = merged_df.merge(cancellations_df, on='sku', how='outer')
        else:
            merged_df['cancelled'] = 0
        
        merged_df['shipments'] = merged_df['shipments'].fillna(0).astype(int)
        merged_df['cancelled'] = merged_df['cancelled'].fillna(0).astype(int)
        
        # Calculate units_ordered = shipments + cancellations
        merged_df['units_ordered'] = merged_df['shipments'] + merged_df['cancelled']
        
        # Calculate cancel percentage
        merged_df['cancel_percentage'] = merged_df.apply(
            lambda row: (row['cancelled'] / row['units_ordered'] * 100) if row['units_ordered'] > 0 else 0.0,
            axis=1
        )
        
        # Filter products with cancellations > 0 and sort by cancelled count (descending)
        result_df = merged_df[merged_df['cancelled'] > 0].copy()
        result_df = result_df.sort_values('cancelled', ascending=False).head(limit)
        
        # Format results
        data = []
        for _, row in result_df.iterrows():
            data.append({
                'sku': str(row['sku']),
                'units_ordered': int(row['units_ordered']),
                'cancelled': int(row['cancelled']),
                'cancel_percentage': round(float(row['cancel_percentage']), 2)
            })
        
        logger.info(f"Returning {len(data)} products with cancellation data")
        
        return {"data": data}
    
    except ValueError as e:
        log_error(e, 'get_cancellations_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": f"Invalid input: {str(e)}", "status": 400}
    except Exception as e:
        log_error(e, 'get_cancellations_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": "Failed to fetch cancellations data. Please try again later.", "status": 500}


def get_free_replacements_data(
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get free replacements data for Product Quality Issues dashboard.
    
    Calculates replacement loss per SKU based on ASIN lookup.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Number of products to return (default: 10)
    
    Returns:
        Dictionary with:
        - data: List of products with replacement metrics
        Each product: {
            "sku": "...",
            "replacements": 5,
            "total_loss": 10000.0
        }
    """
    if not table_exists('sales'):
        return {"data": []}
    
    try:
        # Detect column names
        column_info = execute_query("DESCRIBE sales")
        
        sku_columns = ['sku', 'Sku', 'SKU']
        asin_columns = ['asin', 'Asin', 'ASIN', 'Amazon ASIN']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        shipping_columns = ['shipping_loss_calc', 'shipping_amount', 'Shipping Amount']
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        asin_col = None
        for col in asin_columns:
            if col in column_info['column_name'].values:
                asin_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        shipping_col = None
        for col in shipping_columns:
            if col in column_info['column_name'].values:
                shipping_col = col
                break
        
        if not sku_col or not txn_col or not date_col:
            logger.warning("Required columns not found for free replacements data")
            return {"data": []}
        
        if not asin_col:
            logger.warning("ASIN column not found - cannot calculate replacement loss")
            return {"data": []}
        
        # Handle date column type
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        date_filter = f'CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\'' if needs_cast else f'"{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        # Query: Get free replacements with ASIN
        replacements_sql = f"""
        SELECT 
            "{sku_col}" as sku,
            "{asin_col}" as asin,
            COUNT(*) as replacement_count
        FROM sales
        WHERE "{txn_col}" = 'FreeReplacement' AND {date_filter}
            AND "{asin_col}" IS NOT NULL
            AND TRIM("{asin_col}") != ''
        GROUP BY "{sku_col}", "{asin_col}"
        """
        
        replacements_df = execute_query(replacements_sql)
        
        if replacements_df.empty:
            return {"data": []}
        
        # Query: Get shipment data with ASIN, invoice_amount, shipping_amount
        # We need to lookup the original shipment for each ASIN to get pricing
        shipment_select = f'COALESCE(AVG(ABS({revenue_col})), 0)' if revenue_col else '0'
        shipping_select = f'COALESCE(AVG(ABS({shipping_col})), 0)' if shipping_col else '0'
        
        shipments_sql = f"""
        SELECT 
            "{asin_col}" as asin,
            {shipment_select} as invoice_amount,
            {shipping_select} as shipping_amount
        FROM sales
        WHERE "{txn_col}" = 'Shipment' AND {date_filter}
            AND "{asin_col}" IS NOT NULL
            AND TRIM("{asin_col}") != ''
        GROUP BY "{asin_col}"
        """
        
        shipments_df = execute_query(shipments_sql)
        
        # Merge replacements with shipment pricing data
        replacements_df['asin'] = replacements_df['asin'].astype(str).str.strip()
        if not shipments_df.empty:
            shipments_df['asin'] = shipments_df['asin'].astype(str).str.strip()
            merged_df = replacements_df.merge(shipments_df, on='asin', how='left')
        else:
            merged_df = replacements_df.copy()
            merged_df['invoice_amount'] = 0.0
            merged_df['shipping_amount'] = 0.0
        
        merged_df['invoice_amount'] = merged_df['invoice_amount'].fillna(0.0).astype(float)
        merged_df['shipping_amount'] = merged_df['shipping_amount'].fillna(0.0).astype(float)
        
        # Calculate loss per unit: (2 × invoice_amount) + shipping_amount
        merged_df['loss_per_unit'] = (2 * merged_df['invoice_amount']) + merged_df['shipping_amount']
        
        # Calculate total loss per replacement
        merged_df['total_loss_per_replacement'] = merged_df['loss_per_unit'] * merged_df['replacement_count']
        
        # Group by SKU and sum
        sku_losses = merged_df.groupby('sku').agg({
            'replacement_count': 'sum',
            'total_loss_per_replacement': 'sum'
        }).reset_index()
        
        sku_losses.columns = ['sku', 'replacements', 'total_loss']
        
        # Sort by total_loss descending and get top N
        result_df = sku_losses.sort_values('total_loss', ascending=False).head(limit)
        
        # Format results
        data = []
        for _, row in result_df.iterrows():
            data.append({
                'sku': str(row['sku']),
                'replacements': int(row['replacements']),
                'total_loss': round(float(row['total_loss']), 2)
            })
        
        logger.info(f"Returning {len(data)} products with free replacement data")
        
        return {"data": data}
    
    except ValueError as e:
        log_error(e, 'get_free_replacements_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": f"Invalid input: {str(e)}", "status": 400}
    except Exception as e:
        log_error(e, 'get_free_replacements_data', {'start_date': start_date, 'end_date': end_date, 'limit': limit})
        return {"data": [], "error": True, "message": "Failed to fetch free replacements data. Please try again later.", "status": 500}


def get_skus_by_region(
    region: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Get top SKUs for a specific region/city
    
    Args:
        region: Region/city name (case-insensitive)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        limit: Number of SKUs to return (default: 10)
    
    Returns:
        DataFrame with columns: sku, asin, units, revenue
    """
    if not table_exists('sales'):
        return pd.DataFrame(columns=['sku', 'asin', 'units', 'revenue'])
    
    try:
        # Detect column names dynamically
        column_info = execute_query("DESCRIBE sales")
        
        region_columns = ['region', 'Region', 'city', 'City', 'location', 'Location']
        sku_columns = ['sku', 'Sku', 'SKU']
        asin_columns = ['asin', 'Asin', 'ASIN', 'Amazon ASIN']
        revenue_columns = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']
        quantity_columns = ['quantity', 'Quantity', 'units_sold']
        txn_columns = ['Transaction Type', 'transaction_type']
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
        
        region_col = None
        for col in region_columns:
            if col in column_info['column_name'].values:
                region_col = col
                break
        
        sku_col = None
        for col in sku_columns:
            if col in column_info['column_name'].values:
                sku_col = col
                break
        
        asin_col = None
        for col in asin_columns:
            if col in column_info['column_name'].values:
                asin_col = col
                break
        
        revenue_col = None
        for col in revenue_columns:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        quantity_col = None
        for col in quantity_columns:
            if col in column_info['column_name'].values:
                quantity_col = col
                break
        
        txn_col = None
        for col in txn_columns:
            if col in column_info['column_name'].values:
                txn_col = col
                break
        
        date_col = None
        for col in date_columns:
            if col in column_info['column_name'].values:
                date_col = col
                break
        
        if not region_col or not sku_col or not revenue_col:
            logger.warning(f"Required columns not found for region SKUs. region_col: {region_col}, sku_col: {sku_col}, revenue_col: {revenue_col}")
            return pd.DataFrame(columns=['sku', 'asin', 'units', 'revenue'])
        
        logger.info(f"Getting SKUs for region: {region}")
        logger.info(f"Using columns - Region: {region_col}, SKU: {sku_col}, ASIN: {asin_col}, Revenue: {revenue_col}, Quantity: {quantity_col}, Transaction: {txn_col}, Date: {date_col}")
        
        # Build date filter
        date_filter = ""
        if start_date and end_date and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                date_filter = f'AND CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\''
            else:
                date_filter = f'AND "{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
        
        # Build region filter (case-insensitive)
        region_filter = f'LOWER(TRIM("{region_col}")) = LOWER(TRIM(\'{region}\'))'
        
        # Build ASIN selection
        asin_select = f'COALESCE(MAX("{asin_col}"), \'\')' if asin_col else '\'\''
        
        # Build quantity selection
        quantity_select = f'COALESCE(SUM({quantity_col}), 0)' if quantity_col else 'COUNT(*)'
        
        # Build SQL query - include ALL transaction types (not just Shipments)
        # This matches the fix in get_revenue_by_region() - keep date filter, remove transaction type filter
        if revenue_col == 'revenue_calc':
            # revenue_calc is transaction-aware: positive = shipments, negative = refunds
            # Sum all values to get net revenue per SKU
            sql = f"""
            SELECT 
                "{sku_col}" as sku,
                {asin_select} as asin,
                {quantity_select} as units,
                COALESCE(SUM({revenue_col}), 0) as revenue
            FROM sales
            WHERE {region_filter} {date_filter}
            GROUP BY "{sku_col}"
            HAVING revenue > 0
            ORDER BY units DESC
            LIMIT {limit}
            """
        else:
            # For other revenue columns, use positive amounts only
            sql = f"""
            SELECT 
                "{sku_col}" as sku,
                {asin_select} as asin,
                {quantity_select} as units,
                COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as revenue
            FROM sales
            WHERE {region_filter} AND {revenue_col} > 0 {date_filter}
            GROUP BY "{sku_col}"
            HAVING revenue > 0
            ORDER BY units DESC
            LIMIT {limit}
            """
        
        logger.info(f"Executing SQL: {sql}")
        df = execute_query(sql)
        
        logger.info(f"Query returned {len(df)} SKUs for region {region}")
        
        # Ensure numeric columns
        if not df.empty:
            df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
            df['sku'] = df['sku'].astype(str).str.strip()
            df['asin'] = df['asin'].astype(str).str.strip()
        
        return df
    
    except ValueError as e:
        log_error(e, 'get_skus_by_region', {'region': region, 'limit': limit})
        return pd.DataFrame(columns=['sku', 'asin', 'units', 'revenue'])
    except Exception as e:
        log_error(e, 'get_skus_by_region', {'region': region, 'limit': limit})
        return pd.DataFrame(columns=['sku', 'asin', 'units', 'revenue'])
