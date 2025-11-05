"""
Metrics Service - Business KPI calculations

Extracted and refactored from legacy/app.py
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from core.database import execute_query, table_exists

logger = logging.getLogger(__name__)


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
        
    except Exception as e:
        logger.error(f"Error getting filtered data: {e}")
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
    if not table_exists('sales'):
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
        
        return results
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def get_daily_trends(
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Get daily trends with revenue, refunds, and orders grouped by date
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dictionary with:
        - data: List of daily records with date, revenue, refunds, orders
        - revenue_trend: List of {date, value} for revenue chart
        - refund_trend: List of {date, value} for refund chart
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
        date_columns = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']
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
        for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr']:
            if col in column_info['column_name'].values:
                revenue_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['Transaction Type', 'transaction_type']:
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
        
        # Build date filter
        if needs_cast:
            date_filter = f"CAST(\"{date_col}\" AS DATE) >= '{start_date}' AND CAST(\"{date_col}\" AS DATE) <= '{end_date}'"
            date_group = f"CAST(\"{date_col}\" AS DATE)"
        else:
            date_filter = f"\"{date_col}\" >= '{start_date}' AND \"{date_col}\" <= '{end_date}'"
            date_group = f"\"{date_col}\""
        
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
    
    except Exception as e:
        logger.error(f"Error getting daily trends: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'data': [],
            'revenue_trend': [],
            'refund_trend': [],
            'count': 0,
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
    
    except Exception as e:
        logger.error(f"Error getting top products: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['sku', 'asin', 'units_sold', 'revenue', 'refund_ratio', 'rating', 'trend'])
