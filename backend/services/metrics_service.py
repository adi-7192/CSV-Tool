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


def get_revenue_by_city(limit: int = 10) -> Dict[str, Any]:
    """
    Get top cities by total revenue (all-time data, no filters).
    
    Uses "Ship To City" column with case-insensitive normalization.
    Handles city name variations (Bangalore/Bengaluru, Delhi/New Delhi, etc.).
    
    Args:
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
        
        # DEBUG: Log all available columns
        logger.info(f"Available columns in sales table: {available_cols}")
        print(f"\n🔍 DEBUG: Available columns: {available_cols}\n")
        
        # Check if both columns exist
        has_ship_to_city = any(col.lower().strip() == 'ship to city' for col in available_cols)
        has_bill_from_city = any(col.lower().strip() == 'bill from city' for col in available_cols)
        
        if has_ship_to_city:
            print(f"✅ 'Ship To City' column EXISTS in database")
        else:
            print(f"⚠️  'Ship To City' column NOT FOUND in database")
        
        if has_bill_from_city:
            print(f"⚠️  'Bill From City' column EXISTS (will be EXCLUDED - wrong column)")
        
        print()
        
        # Find city/region column (try multiple variations)
        # PRIORITY: "Ship To City" is the correct column (customer delivery location)
        # EXCLUDE: "Bill From City" (this is wrong - it's the seller location)
        city_col = None
        
        # FIRST PRIORITY: "Ship To City" (exact case-insensitive match)
        for col in available_cols:
            col_lower = col.lower().strip()
            if col_lower == 'ship to city':
                city_col = col
                logger.info(f"✅ Found correct city column: '{city_col}' (Ship To City - customer location)")
                print(f"\n✅✅✅ USING CORRECT COLUMN: '{city_col}' (Ship To City - customer delivery location)\n")
                break
        
        # CRITICAL: If "Ship To City" not found, check if "Bill From City" exists
        if not city_col:
            if has_bill_from_city:
                logger.warning("⚠️  WARNING: 'Ship To City' column NOT FOUND, but 'Bill From City' exists!")
                logger.warning("⚠️  'Bill From City' will be EXCLUDED - this is seller location, not customer location")
                print(f"\n⚠️  WARNING: 'Ship To City' column NOT FOUND!")
                print(f"⚠️  'Bill From City' exists but will be EXCLUDED (wrong column)\n")
        
        # If not found, try "region" column (standardized name)
        if not city_col:
            for col in available_cols:
                if col.lower() == 'region':
                    city_col = col
                    logger.info(f"Found city column (region): '{city_col}'")
                    print(f"✅ Found column: '{city_col}' (region)")
                    break
        
        # If still not found, try other variations (but EXCLUDE "Bill From City")
        if not city_col:
            region_variations = ['city', 'City', 'location', 'Location', 'shipping_city', 'Shipping City', 
                               'delivery_city', 'Delivery City', 'ship_to_city', 'ship_to_state', 
                               'Ship To City', 'Ship To State', 'ship to state']
            for col in region_variations:
                if col in available_cols:
                    # EXCLUDE "Bill From City" - this is NOT what we want
                    if col.lower() != 'bill from city':
                        city_col = col
                        logger.info(f"Found city column (variation): '{city_col}'")
                        print(f"✅ Found column: '{city_col}' (variation)")
                        break
        
        # Last resort: Pattern matching (but EXCLUDE "Bill From City")
        if not city_col:
            for col in available_cols:
                col_lower = col.lower()
                # EXCLUDE "Bill From City" explicitly
                if col_lower == 'bill from city':
                    logger.info(f"Skipping '{col}' - this is Bill From City (wrong column)")
                    print(f"⚠️  Skipping '{col}' - this is Bill From City (wrong column)")
                    continue
                
                # Check if column name contains city/region/location keywords
                if any(keyword in col_lower for keyword in ['city', 'region', 'location', 'state', 'ship']):
                    # Verify it's not a date or other type
                    col_type = column_info[column_info['column_name'] == col]['column_type'].values[0]
                    if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                        city_col = col
                        logger.info(f"Found city column by pattern matching: '{city_col}'")
                        print(f"✅ Found column by pattern: '{city_col}'")
                        break
        
        if not city_col:
            logger.warning(f"'Ship To City' or 'region' column not found. Available columns: {available_cols}")
            print(f"\n⚠️  ERROR: City column not found! Available columns: {available_cols}\n")
            # Try to get sample data to help debug
            try:
                sample = execute_query("SELECT * FROM sales LIMIT 1")
                if not sample.empty:
                    print(f"🔍 DEBUG: Sample row columns: {list(sample.columns)}")
                    print(f"🔍 DEBUG: Sample data:")
                    for col in sample.columns:
                        val = sample[col].iloc[0]
                        print(f"  {col}: {val} (type: {type(val).__name__})")
            except Exception as e:
                print(f"🔍 DEBUG: Could not fetch sample: {e}")
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
            print(f"\n⚠️  ERROR: Revenue column not found! Available columns: {available_cols}\n")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # DEBUG: Check if we have any data
        count_sql = f'SELECT COUNT(*) as total FROM sales WHERE "{city_col}" IS NOT NULL AND TRIM("{city_col}") != \'\''
        count_df = execute_query(count_sql)
        total_records = int(count_df['total'].iloc[0]) if not count_df.empty else 0
        logger.info(f"Total records with city data: {total_records}")
        print(f"\n🔍 DEBUG: Total records with city data: {total_records:,}\n")
        
        if total_records == 0:
            logger.warning("No records found with city data")
            print(f"\n⚠️  WARNING: No records found with city data!\n")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # City normalization mapping
        city_mapping = {
            'bangalore': 'Bangalore',
            'bengaluru': 'Bangalore',
            'delhi': 'New Delhi',
            'new delhi': 'New Delhi',
            'delhi ncr': 'New Delhi',
            'bombay': 'Mumbai',
            'kolkata': 'Kolkata',
            'calcutta': 'Kolkata',
            'hyderabad': 'Hyderabad',
            'pune': 'Pune',
            'poona': 'Pune',
            'chennai': 'Chennai',
            'madras': 'Chennai',
            'gurugram': 'Gurugram',
            'gurgaon': 'Gurugram',
            'noida': 'Noida',
            'navi mumbai': 'Navi Mumbai',
            'thane': 'Thane',
        }
        
        # CRITICAL VERIFICATION: Ensure we're using "Ship To City" not "Bill From City"
        if city_col.lower() == 'bill from city':
            logger.error("❌ ERROR: Selected column is 'Bill From City' - this is WRONG!")
            logger.error("❌ Should use 'Ship To City' instead (customer location)")
            print(f"\n❌❌❌ ERROR: Selected column is 'Bill From City' - this is WRONG!")
            print(f"❌ Should use 'Ship To City' instead (customer location)\n")
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        logger.info(f"✅ Using city column: '{city_col}' for revenue by city calculation")
        print(f"\n✅ VERIFIED: Using column '{city_col}' for city revenue\n")
        
        # Fetch all data (no date/transaction filters)
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
        """
        
        logger.info(f"Fetching revenue by city - SQL: {sql}")
        print(f"\n🔍 DEBUG: Executing SQL:\n{sql}\n")
        print(f"🔍 DEBUG: Using column '{city_col}' (should be 'Ship To City')\n")
        raw_df = execute_query(sql)
        
        logger.info(f"Query returned {len(raw_df)} rows")
        print(f"\n🔍 DEBUG: Query returned {len(raw_df):,} rows\n")
        
        if raw_df.empty:
            logger.warning("No city data found after query")
            print(f"\n⚠️  WARNING: Query returned empty result!\n")
            # Try a simpler query to see if data exists
            test_sql = f'SELECT "{city_col}", {revenue_col} FROM sales LIMIT 5'
            test_df = execute_query(test_sql)
            print(f"🔍 DEBUG: Test query returned {len(test_df)} rows")
            if not test_df.empty:
                print(f"🔍 DEBUG: Sample data from test query:")
                print(test_df.head())
            return {
                "data": [],
                "count": 0,
                "total_revenue": 0.0
            }
        
        # DEBUG: Show sample raw data
        print(f"\n🔍 DEBUG: Sample raw data (first 10 rows):")
        print(raw_df.head(10))
        print(f"\n🔍 DEBUG: Unique cities in raw data: {raw_df['raw_city'].nunique()}")
        unique_raw_cities = raw_df['raw_city'].unique()
        print(f"🔍 DEBUG: All unique cities ({len(unique_raw_cities)}): {unique_raw_cities.tolist()}\n")
        
        # CRITICAL: Check if we're seeing Amethi and Mumbai (these might be from Bill From City)
        if 'Amethi' in unique_raw_cities or 'amethi' in [c.lower() for c in unique_raw_cities]:
            print(f"⚠️  WARNING: Found 'Amethi' in results - this might indicate wrong column!")
            print(f"⚠️  'Amethi' is typically a seller location (Bill From City), not customer location")
        
        # Show revenue distribution by city
        print(f"🔍 DEBUG: Revenue by raw city (before normalization):")
        city_revenue_raw = raw_df.groupby('raw_city')['revenue'].sum().sort_values(ascending=False)
        print(f"  Total cities with revenue: {len(city_revenue_raw)}")
        for city, rev in city_revenue_raw.head(20).items():
            print(f"  {city}: ₹{rev:,.2f}")
        print()
        
        # Verify the column being used
        print(f"🔍 DEBUG: VERIFICATION - Column being used: '{city_col}'")
        if city_col.lower() == 'bill from city':
            print(f"❌❌❌ CRITICAL ERROR: Using 'Bill From City' - this is WRONG!")
        elif city_col.lower() == 'ship to city':
            print(f"✅✅✅ CORRECT: Using 'Ship To City' - this is RIGHT!")
        print()
        
        # Normalize city names
        def normalize_city(city_name):
            """Normalize city name using mapping and title case"""
            if pd.isna(city_name) or not city_name:
                return None
            
            city_str = str(city_name).strip().lower()
            if not city_str:
                return None
            
            # Check mapping first
            if city_str in city_mapping:
                return city_mapping[city_str]
            
            # Otherwise, apply title case
            words = city_str.split()
            normalized_words = [word.capitalize() for word in words]
            return ' '.join(normalized_words)
        
        # Apply normalization
        raw_df['city'] = raw_df['raw_city'].apply(normalize_city)
        raw_df = raw_df[raw_df['city'].notna() & (raw_df['city'] != '')]
        
        # DEBUG: Show normalization results
        print(f"\n🔍 DEBUG: After normalization:")
        print(f"  Rows remaining: {len(raw_df):,}")
        print(f"  Unique normalized cities: {raw_df['city'].nunique()}")
        unique_normalized = raw_df['city'].unique()
        print(f"  All normalized cities ({len(unique_normalized)}): {unique_normalized.tolist()}\n")
        
        # Group by normalized city and sum revenue
        city_revenue = raw_df.groupby('city')['revenue'].sum().reset_index()
        
        # DEBUG: Show revenue by normalized city
        print(f"🔍 DEBUG: Revenue by normalized city (before filtering):")
        city_revenue_sorted = city_revenue.sort_values('revenue', ascending=False)
        for idx, row in city_revenue_sorted.head(20).iterrows():
            print(f"  {row['city']}: ₹{row['revenue']:,.2f}")
        print()
        
        # Filter out zero/negative revenue
        city_revenue = city_revenue[city_revenue['revenue'] > 0]
        
        print(f"🔍 DEBUG: After filtering revenue > 0:")
        print(f"  Cities remaining: {len(city_revenue)}")
        print()
        
        # Sort by revenue descending and get top N
        top_cities = city_revenue.sort_values('revenue', ascending=False).head(limit)
        
        # DEBUG: Show final results
        print(f"🔍 DEBUG: Final top {limit} cities:")
        final_cities = []
        for idx, row in top_cities.iterrows():
            city_name = row['city']
            city_rev = row['revenue']
            print(f"  {city_name}: ₹{city_rev:,.2f}")
            final_cities.append(city_name)
        
        # CRITICAL CHECK: Verify we're not returning Amethi/Mumbai (Bill From City cities)
        if 'Amethi' in final_cities or 'Mumbai' in final_cities:
            print(f"\n⚠️  WARNING: Final results contain 'Amethi' and/or 'Mumbai'")
            print(f"⚠️  These are typically seller locations (Bill From City)")
            print(f"⚠️  If you're seeing only these 2 cities, we might be using wrong column!")
            print(f"⚠️  Column being used: '{city_col}'")
            if city_col.lower() != 'ship to city':
                print(f"❌ ERROR: Column '{city_col}' is NOT 'Ship To City'!")
        
        print()
        
        # Convert to list of dicts
        cities_list = top_cities.to_dict('records')
        
        # Format revenue as float
        for city in cities_list:
            city['revenue'] = float(city['revenue'])
        
        total_revenue = float(top_cities['revenue'].sum())
        
        logger.info(f"Returning {len(cities_list)} cities, total revenue: ₹{total_revenue:,.2f}")
        logger.info(f"Final cities: {final_cities}")
        
        print(f"\n✅ FINAL RESULT: Returning {len(cities_list)} cities to frontend")
        print(f"✅ Cities: {final_cities}\n")
        
        return {
            "data": cities_list,
            "count": len(cities_list),
            "total_revenue": total_revenue
        }
    
    except Exception as e:
        logger.error(f"Error getting revenue by city: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "data": [],
            "count": 0,
            "total_revenue": 0.0
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
        
        # DEBUG: Log all available columns
        logger.info(f"Available columns in sales table (get_skus_by_city): {available_cols}")
        print(f"\n🔍 DEBUG (get_skus_by_city): Available columns: {available_cols}\n")
        
        # Find city/region column (try multiple variations)
        # PRIORITY: "Ship To City" is the correct column (customer delivery location)
        # EXCLUDE: "Bill From City" (this is wrong - it's the seller location)
        city_col = None
        
        # FIRST PRIORITY: "Ship To City" (exact case-insensitive match)
        for col in available_cols:
            if col.lower() == 'ship to city':
                city_col = col
                logger.info(f"Found city column (Ship To City): '{city_col}'")
                print(f"✅ Found correct column: '{city_col}' (Ship To City)")
                break
        
        # If not found, try "region" column (standardized name)
        if not city_col:
            for col in available_cols:
                if col.lower() == 'region':
                    city_col = col
                    logger.info(f"Found city column (region): '{city_col}'")
                    print(f"✅ Found column: '{city_col}' (region)")
                    break
        
        # If still not found, try other variations (but EXCLUDE "Bill From City")
        if not city_col:
            region_variations = ['city', 'City', 'location', 'Location', 'shipping_city', 'Shipping City', 
                               'delivery_city', 'Delivery City', 'ship_to_city', 'ship_to_state', 
                               'Ship To City', 'Ship To State', 'ship to state']
            for col in region_variations:
                if col in available_cols:
                    # EXCLUDE "Bill From City" - this is NOT what we want
                    if col.lower() != 'bill from city':
                        city_col = col
                        logger.info(f"Found city column (variation): '{city_col}'")
                        print(f"✅ Found column: '{city_col}' (variation)")
                        break
        
        # Last resort: Pattern matching (but EXCLUDE "Bill From City")
        if not city_col:
            for col in available_cols:
                col_lower = col.lower()
                # EXCLUDE "Bill From City" explicitly
                if col_lower == 'bill from city':
                    logger.info(f"Skipping '{col}' - this is Bill From City (wrong column)")
                    print(f"⚠️  Skipping '{col}' - this is Bill From City (wrong column)")
                    continue
                
                # Check if column name contains city/region/location keywords
                if any(keyword in col_lower for keyword in ['city', 'region', 'location', 'state', 'ship']):
                    # Verify it's not a date or other type
                    col_type = column_info[column_info['column_name'] == col]['column_type'].values[0]
                    if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                        city_col = col
                        logger.info(f"Found city column by pattern matching: '{city_col}'")
                        print(f"✅ Found column by pattern: '{city_col}'")
                        break
        
        if not city_col:
            logger.warning(f"'Ship To City' or 'region' column not found. Available columns: {available_cols}")
            print(f"\n⚠️  ERROR (get_skus_by_city): City column not found! Available columns: {available_cols}\n")
            # Try to get sample data to help debug
            try:
                sample = execute_query("SELECT * FROM sales LIMIT 1")
                if not sample.empty:
                    print(f"🔍 DEBUG: Sample row columns: {list(sample.columns)}")
                    print(f"🔍 DEBUG: Sample data:")
                    for col in sample.columns:
                        val = sample[col].iloc[0]
                        print(f"  {col}: {val} (type: {type(val).__name__})")
            except Exception as e:
                print(f"🔍 DEBUG: Could not fetch sample: {e}")
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
            print(f"\n⚠️  ERROR (get_skus_by_city): Required columns not found!")
            print(f"  city_col: {city_col}")
            print(f"  sku_col: {sku_col}")
            print(f"  revenue_col: {revenue_col}")
            print(f"  Available columns: {available_cols}\n")
            return {
                "city": city,
                "data": [],
                "count": 0
            }
        
        # City normalization mapping (same as get_revenue_by_city)
        city_mapping = {
            'bangalore': 'Bangalore',
            'bengaluru': 'Bangalore',
            'delhi': 'New Delhi',
            'new delhi': 'New Delhi',
            'delhi ncr': 'New Delhi',
            'bombay': 'Mumbai',
            'kolkata': 'Kolkata',
            'calcutta': 'Kolkata',
            'hyderabad': 'Hyderabad',
            'pune': 'Pune',
            'poona': 'Pune',
            'chennai': 'Chennai',
            'madras': 'Chennai',
            'gurugram': 'Gurugram',
            'gurgaon': 'Gurugram',
            'noida': 'Noida',
            'navi mumbai': 'Navi Mumbai',
            'thane': 'Thane',
        }
        
        # Normalize input city name (same normalization as get_revenue_by_city)
        # Use the EXACT same normalization function as get_revenue_by_city
        def normalize_city_input(city_name):
            """Normalize city name using mapping and title case - SAME as get_revenue_by_city"""
            if not city_name:
                return None
            city_str = str(city_name).strip().lower()
            if not city_str:
                return None
            # Check mapping first
            if city_str in city_mapping:
                return city_mapping[city_str]
            # Otherwise, apply title case
            words = city_str.split()
            normalized_words = [word.capitalize() for word in words]
            return ' '.join(normalized_words)
        
        city_normalized = normalize_city_input(city)
        
        logger.info(f"Getting SKUs for city: '{city}' -> normalized: '{city_normalized}'")
        print(f"\n🔍 DEBUG (get_skus_by_city): Input city: '{city}' -> Normalized: '{city_normalized}'")
        print(f"🔍 DEBUG (get_skus_by_city): City mapping check - '{city.lower().strip()}' in mapping: {city.lower().strip() in city_mapping}\n")
        
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
        
        logger.info(f"Fetching SKUs for city '{city_normalized}' - SQL: {sql}")
        print(f"🔍 DEBUG (get_skus_by_city): Executing SQL query\n")
        df = execute_query(sql)
        
        print(f"🔍 DEBUG (get_skus_by_city): Query returned {len(df)} rows\n")
        
        if df.empty:
            logger.warning(f"No SKU data found for city: {city_normalized}")
            return {
                "city": city_normalized,
                "data": [],
                "count": 0
            }
        
        # Normalize city names in the result (SAME function as get_revenue_by_city)
        def normalize_city(city_name):
            """Normalize city name using mapping and title case - SAME as get_revenue_by_city"""
            if pd.isna(city_name) or not city_name:
                return None
            city_str = str(city_name).strip().lower()
            if not city_str:
                return None
            # Check mapping first
            if city_str in city_mapping:
                return city_mapping[city_str]
            # Otherwise, apply title case
            words = city_str.split()
            normalized_words = [word.capitalize() for word in words]
            return ' '.join(normalized_words)
        
        # Apply normalization and filter for the selected city
        df['city_normalized'] = df['raw_city'].apply(normalize_city)
        
        # DEBUG: Show normalization results
        print(f"🔍 DEBUG (get_skus_by_city): After normalization:")
        print(f"  Total rows: {len(df)}")
        print(f"  Unique normalized cities: {df['city_normalized'].nunique()}")
        unique_cities_found = df['city_normalized'].unique()
        print(f"  Cities found: {unique_cities_found.tolist()}")
        print(f"  Looking for: '{city_normalized}'\n")
        
        city_data = df[df['city_normalized'] == city_normalized]
        
        if city_data.empty:
            logger.warning(f"No SKU data found for normalized city: {city_normalized}")
            print(f"\n⚠️  WARNING: No data found for normalized city '{city_normalized}'")
            print(f"⚠️  Input city was: '{city}'")
            print(f"⚠️  Normalized to: '{city_normalized}'")
            print(f"⚠️  Available cities in data: {unique_cities_found.tolist()}")
            
            # Try case-insensitive match as fallback
            city_data_fallback = df[df['city_normalized'].str.lower() == city_normalized.lower()] if 'city_normalized' in df.columns else pd.DataFrame()
            if not city_data_fallback.empty:
                print(f"⚠️  Found {len(city_data_fallback)} rows with case-insensitive match")
                city_data = city_data_fallback
            else:
                # Show sample raw cities to help debug
                print(f"⚠️  Sample raw cities from query (first 20):")
                raw_cities_sample = df['raw_city'].unique()[:20]
                for raw_city in raw_cities_sample:
                    normalized_sample = normalize_city(raw_city)
                    print(f"    Raw: '{raw_city}' -> Normalized: '{normalized_sample}'")
                print(f"⚠️  This might be a normalization mismatch issue\n")
                return {
                    "city": city_normalized,
                    "data": [],
                    "count": 0
                }
        
        print(f"✅ Found {len(city_data)} rows for city '{city_normalized}'\n")
        
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
    
    except Exception as e:
        logger.error(f"Error getting SKUs by city: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "city": city,
            "data": [],
            "count": 0
        }


def get_movers_decliners(
    start_date: str,
    end_date: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Get movers (fast growing) and decliners (declining) SKUs
    
    Compares current period vs previous period (same duration, 1 week before)
    - Decliners: growth % <= -30%
    - Fast Movers: growth % >= +30%
    
    Args:
        start_date: Start date of current period (YYYY-MM-DD)
        end_date: End date of current period (YYYY-MM-DD)
        limit: Number of SKUs to return per category (default: 10)
    
    Returns:
        Dictionary with "movers" and "decliners" lists
        Each item: {"sku": "...", "revenue": 100000, "wow_change": 35.5}
    """
    if not table_exists('sales'):
        return {"movers": [], "decliners": []}
    
    try:
        from datetime import datetime, timedelta
        
        # Parse dates
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
        current_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate period length
        period_length = (current_end - current_start).days + 1
        
        # Calculate previous period dates (same duration, 1 week before)
        prev_start = current_start - timedelta(days=7 + period_length - 1)
        prev_end = current_start - timedelta(days=7)
        
        prev_start_str = prev_start.strftime('%Y-%m-%d')
        prev_end_str = prev_end.strftime('%Y-%m-%d')
        
        logger.info(f"📊 Movers & Decliners: Current period {start_date} to {end_date} ({period_length} days)")
        logger.info(f"📊 Previous period: {prev_start_str} to {prev_end_str}")
        
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
            return {"movers": [], "decliners": []}
        
        logger.info(f"Using columns - SKU: {sku_col}, Revenue: {revenue_col}, Transaction: {txn_col}, Date: {date_col}")
        
        # Handle date column type (VARCHAR vs DATE)
        needs_cast = False
        if date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper() or 'TEXT' in str(col_type).upper():
                needs_cast = True
        
        # Build date filters
        if needs_cast:
            current_date_filter = f'CAST("{date_col}" AS DATE) >= \'{start_date}\' AND CAST("{date_col}" AS DATE) <= \'{end_date}\''
            prev_date_filter = f'CAST("{date_col}" AS DATE) >= \'{prev_start_str}\' AND CAST("{date_col}" AS DATE) <= \'{prev_end_str}\''
        else:
            current_date_filter = f'"{date_col}" >= \'{start_date}\' AND "{date_col}" <= \'{end_date}\''
            prev_date_filter = f'"{date_col}" >= \'{prev_start_str}\' AND "{date_col}" <= \'{prev_end_str}\''
        
        # Query current period revenue by SKU
        # Handle revenue_calc differently (it's already transaction-aware)
        if txn_col:
            if revenue_col == 'revenue_calc':
                # revenue_calc is already positive for shipments, negative for refunds
                current_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' AND {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as current_revenue
                FROM sales
                WHERE {current_date_filter}
                GROUP BY "{sku_col}"
                HAVING current_revenue > 0
                """
            else:
                current_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as current_revenue
                FROM sales
                WHERE {current_date_filter}
                GROUP BY "{sku_col}"
                HAVING current_revenue > 0
                """
        else:
            if revenue_col == 'revenue_calc':
                current_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as current_revenue
                FROM sales
                WHERE {revenue_col} > 0 AND {current_date_filter}
                GROUP BY "{sku_col}"
                HAVING current_revenue > 0
                """
            else:
                current_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as current_revenue
                FROM sales
                WHERE {revenue_col} > 0 AND {current_date_filter}
                GROUP BY "{sku_col}"
                HAVING current_revenue > 0
                """
        
        logger.info(f"Current period SQL: {current_sql}")
        current_df = execute_query(current_sql)
        logger.info(f"Current period query returned {len(current_df)} SKUs")
        if not current_df.empty:
            logger.info(f"Sample current period SKUs: {current_df['sku'].head(5).tolist()}")
            logger.info(f"Sample current period revenues: {current_df['current_revenue'].head(5).tolist()}")
            logger.info(f"Total current period revenue: {current_df['current_revenue'].sum():.2f}")
        
        # Query previous period revenue by SKU
        if txn_col:
            if revenue_col == 'revenue_calc':
                prev_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' AND {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as prev_revenue
                FROM sales
                WHERE {prev_date_filter}
                GROUP BY "{sku_col}"
                """
            else:
                prev_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                FROM sales
                WHERE {prev_date_filter}
                GROUP BY "{sku_col}"
                """
        else:
            if revenue_col == 'revenue_calc':
                prev_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END), 0) as prev_revenue
                FROM sales
                WHERE {revenue_col} > 0 AND {prev_date_filter}
                GROUP BY "{sku_col}"
                """
            else:
                prev_sql = f"""
                SELECT 
                    "{sku_col}" as sku,
                    COALESCE(SUM(CASE WHEN {revenue_col} > 0 THEN ABS({revenue_col}) ELSE 0 END), 0) as prev_revenue
                FROM sales
                WHERE {revenue_col} > 0 AND {prev_date_filter}
                GROUP BY "{sku_col}"
                """
        
        logger.info(f"Previous period SQL: {prev_sql}")
        prev_df = execute_query(prev_sql)
        logger.info(f"Previous period query returned {len(prev_df)} SKUs")
        if not prev_df.empty:
            logger.info(f"Sample previous period SKUs: {prev_df['sku'].head(5).tolist()}")
            logger.info(f"Sample previous period revenues: {prev_df['prev_revenue'].head(5).tolist()}")
            logger.info(f"Total previous period revenue: {prev_df['prev_revenue'].sum():.2f}")
        
        # CRITICAL DEBUG: Date ranges and record counts
        print("\n" + "="*80)
        print("🔍 CRITICAL DEBUG: DATE RANGES AND RECORD COUNTS")
        print("="*80)
        print(f"Current Period: {start_date} to {end_date}")
        print(f"  Start: {current_start.strftime('%Y-%m-%d')}")
        print(f"  End: {current_end.strftime('%Y-%m-%d')}")
        print(f"  Duration: {period_length} days")
        print(f"\nPrevious Period: {prev_start_str} to {prev_end_str}")
        print(f"  Start: {prev_start.strftime('%Y-%m-%d')}")
        print(f"  End: {prev_end.strftime('%Y-%m-%d')}")
        print(f"  Duration: {(prev_end - prev_start).days + 1} days")
        print(f"\nDate Range Summary:")
        print(f"  Current: {start_date} to {end_date} | Previous: {prev_start_str} to {prev_end_str}")
        print("-" * 80)
        
        # Check database record counts for BOTH periods
        print("\n📊 DATABASE RECORD COUNTS:")
        print("-" * 80)
        
        # Count shipments in current period
        if txn_col:
            current_count_sql = f"""
            SELECT COUNT(*) as total
            FROM sales
            WHERE "{txn_col}" = 'Shipment' AND {current_date_filter}
            """
        else:
            current_count_sql = f"""
            SELECT COUNT(*) as total
            FROM sales
            WHERE {revenue_col} > 0 AND {current_date_filter}
            """
        
        current_count_df = execute_query(current_count_sql)
        current_count = int(current_count_df['total'].iloc[0]) if not current_count_df.empty else 0
        print(f"Current Period Shipment Records: {current_count:,}")
        
        # Count shipments in previous period
        if txn_col:
            prev_count_sql = f"""
            SELECT COUNT(*) as total
            FROM sales
            WHERE "{txn_col}" = 'Shipment' AND {prev_date_filter}
            """
        else:
            prev_count_sql = f"""
            SELECT COUNT(*) as total
            FROM sales
            WHERE {revenue_col} > 0 AND {prev_date_filter}
            """
        
        prev_count_df = execute_query(prev_count_sql)
        prev_count = int(prev_count_df['total'].iloc[0]) if not prev_count_df.empty else 0
        print(f"Previous Period Shipment Records: {prev_count:,}")
        
        if prev_count == 0:
            print("\n⚠️  WARNING: Previous period has ZERO shipment records!")
            print("   This explains why all products show 999% growth (new products)")
            print("   Possible causes:")
            print("   1. Date calculation is wrong")
            print("   2. Date column type mismatch")
            print("   3. No data exists in previous period date range")
        
        print("-" * 80)
        
        # Show sample records from both periods
        print("\n📊 SAMPLE RECORDS FROM DATABASE:")
        print("-" * 80)
        
        # Sample from current period
        if txn_col:
            current_sample_sql = f"""
            SELECT "{sku_col}" as sku, {revenue_col} as revenue, "{date_col}" as date
            FROM sales
            WHERE "{txn_col}" = 'Shipment' AND {current_date_filter}
            LIMIT 5
            """
        else:
            current_sample_sql = f"""
            SELECT "{sku_col}" as sku, {revenue_col} as revenue, "{date_col}" as date
            FROM sales
            WHERE {revenue_col} > 0 AND {current_date_filter}
            LIMIT 5
            """
        
        current_sample_df = execute_query(current_sample_sql)
        print(f"\nCurrent Period Sample Records ({len(current_sample_df)} shown):")
        if not current_sample_df.empty:
            for idx, row in current_sample_df.iterrows():
                print(f"  SKU: {row.get('sku', 'N/A')} | Revenue: ₹{row.get('revenue', 0):,.2f} | Date: {row.get('date', 'N/A')}")
        else:
            print("  ⚠️  No records found!")
        
        # Sample from previous period
        if txn_col:
            prev_sample_sql = f"""
            SELECT "{sku_col}" as sku, {revenue_col} as revenue, "{date_col}" as date
            FROM sales
            WHERE "{txn_col}" = 'Shipment' AND {prev_date_filter}
            LIMIT 5
            """
        else:
            prev_sample_sql = f"""
            SELECT "{sku_col}" as sku, {revenue_col} as revenue, "{date_col}" as date
            FROM sales
            WHERE {revenue_col} > 0 AND {prev_date_filter}
            LIMIT 5
            """
        
        prev_sample_df = execute_query(prev_sample_sql)
        print(f"\nPrevious Period Sample Records ({len(prev_sample_df)} shown):")
        if not prev_sample_df.empty:
            for idx, row in prev_sample_df.iterrows():
                print(f"  SKU: {row.get('sku', 'N/A')} | Revenue: ₹{row.get('revenue', 0):,.2f} | Date: {row.get('date', 'N/A')}")
        else:
            print("  ⚠️  No records found!")
        
        print("-" * 80)
        
        # Show aggregated results
        print("\n📊 AGGREGATED RESULTS:")
        print("-" * 80)
        print(f"Current Period Aggregated:")
        print(f"  SKUs with revenue > 0: {len(current_df)}")
        if not current_df.empty:
            print(f"  Total revenue: ₹{current_df['current_revenue'].sum():,.2f}")
            print(f"  Sample SKUs: {current_df['sku'].head(5).tolist()}")
            print(f"  Sample revenues: {current_df['current_revenue'].head(5).tolist()}")
        else:
            print("  ⚠️  No SKUs found!")
        
        print(f"\nPrevious Period Aggregated:")
        print(f"  SKUs with revenue > 0: {len(prev_df)}")
        if not prev_df.empty:
            print(f"  Total revenue: ₹{prev_df['prev_revenue'].sum():,.2f}")
            print(f"  Sample SKUs: {prev_df['sku'].head(5).tolist()}")
            print(f"  Sample revenues: {prev_df['prev_revenue'].head(5).tolist()}")
        else:
            print("  ⚠️  No SKUs found!")
            print("  ⚠️  This is why all products show 999% growth!")
        
        print("="*80 + "\n")
        
        # Merge current and previous period data
        if current_df.empty:
            return {"movers": [], "decliners": []}
        
        # Ensure SKU values are strings for proper matching
        current_df['sku'] = current_df['sku'].astype(str).str.strip()
        if not prev_df.empty:
            prev_df['sku'] = prev_df['sku'].astype(str).str.strip()
            # Merge on SKU
            merged_df = current_df.merge(prev_df, on='sku', how='left')
        else:
            merged_df = current_df.copy()
            merged_df['prev_revenue'] = 0.0
        
        # Fill missing previous revenue with 0
        merged_df['prev_revenue'] = merged_df['prev_revenue'].fillna(0)
        
        logger.info(f"Merged dataframe has {len(merged_df)} SKUs")
        logger.info(f"SKUs with current revenue > 0: {len(merged_df[merged_df['current_revenue'] > 0])}")
        logger.info(f"SKUs with previous revenue > 0: {len(merged_df[merged_df['prev_revenue'] > 0])}")
        logger.info(f"SKUs with both periods > 0: {len(merged_df[(merged_df['current_revenue'] > 0) & (merged_df['prev_revenue'] > 0)])}")
        
        # Calculate growth percentage
        # Handle three cases:
        # 1. prev_revenue > 0: normal calculation ((current - prev) / prev * 100)
        # 2. prev_revenue = 0 and current_revenue > 0: new product, treat as 100% growth (mover)
        # 3. prev_revenue = 0 and current_revenue = 0: skip (shouldn't happen due to HAVING clause)
        def calculate_growth(row):
            current = float(row['current_revenue'])
            previous = float(row['prev_revenue'])
            
            if previous > 0:
                # Normal case: calculate percentage change
                return round(((current - previous) / previous) * 100, 1)
            elif current > 0:
                # New product: treat as infinite growth (1000% as placeholder for sorting)
                # We'll filter these separately
                return 1000.0
            else:
                # Shouldn't happen, but return 0
                return 0.0
        
        # DEBUG: Log BEFORE growth calculation - first 5 SKUs with full details
        print("\n" + "="*80)
        print("🔍 DEBUG: MOVERS & DECLINERS - BEFORE GROWTH CALCULATION")
        print("="*80)
        for idx, row in merged_df.head(5).iterrows():
            sku = str(row['sku'])
            current = float(row['current_revenue'])
            previous = float(row['prev_revenue'])
            # Calculate growth manually for logging
            if previous > 0:
                growth = ((current - previous) / previous) * 100
            elif current > 0:
                growth = 1000.0  # New product
            else:
                growth = 0.0
            print(f"SKU: {sku}")
            print(f"  Current Period Revenue: ₹{current:,.2f}")
            print(f"  Previous Period Revenue: ₹{previous:,.2f}")
            print(f"  Calculated Growth %: {growth:.1f}%")
            print("-" * 80)
        print(f"Total SKUs in merged dataframe: {len(merged_df)}")
        print("="*80 + "\n")
        
        merged_df['wow_change'] = merged_df.apply(calculate_growth, axis=1)
        
        # Log sample calculations
        sample_calc = merged_df[['sku', 'current_revenue', 'prev_revenue', 'wow_change']].head(10)
        logger.info(f"Sample growth calculations:\n{sample_calc.to_string()}")
        
        # DEBUG: Log AFTER growth calculation - first 5 SKUs
        print("\n" + "="*80)
        print("🔍 DEBUG: MOVERS & DECLINERS - AFTER GROWTH CALCULATION")
        print("="*80)
        for idx, row in merged_df.head(5).iterrows():
            print(f"SKU: {row['sku']} | Current: ₹{row['current_revenue']:,.2f} | Previous: ₹{row['prev_revenue']:,.2f} | Growth: {row['wow_change']:.1f}%")
        print("="*80 + "\n")
        
        # Filter decliners (growth <= -30%)
        # Exclude new products (wow_change = 1000)
        decliners_df = merged_df[(merged_df['wow_change'] <= -30) & (merged_df['wow_change'] < 1000)].copy()
        decliners_df = decliners_df.sort_values('wow_change', ascending=True).head(limit)
        logger.info(f"Found {len(decliners_df)} decliners (growth <= -30%)")
        if not decliners_df.empty:
            logger.info(f"Sample decliners: {decliners_df[['sku', 'current_revenue', 'prev_revenue', 'wow_change']].head(5).to_dict('records')}")
        
        # Filter movers (growth >= +30% OR new products)
        # Include both high growth products and new products
        movers_df = merged_df[merged_df['wow_change'] >= 30].copy()
        movers_df = movers_df.sort_values('wow_change', ascending=False).head(limit)
        logger.info(f"Found {len(movers_df)} movers (growth >= +30% or new products)")
        if not movers_df.empty:
            logger.info(f"Sample movers: {movers_df[['sku', 'current_revenue', 'prev_revenue', 'wow_change']].head(5).to_dict('records')}")
        
        # If we don't have enough results, relax the threshold
        if len(decliners_df) < 5 and len(merged_df) > 0:
            logger.info("Not enough decliners found, checking for any negative growth...")
            any_decliners = merged_df[(merged_df['wow_change'] < 0) & (merged_df['wow_change'] < 1000)].copy()
            if len(any_decliners) > 0:
                logger.info(f"Found {len(any_decliners)} SKUs with negative growth")
                decliners_df = any_decliners.sort_values('wow_change', ascending=True).head(limit)
        
        if len(movers_df) < 5 and len(merged_df) > 0:
            logger.info("Not enough movers found, checking for any positive growth...")
            any_movers = merged_df[(merged_df['wow_change'] > 0) & (merged_df['wow_change'] < 1000)].copy()
            if len(any_movers) > 0:
                logger.info(f"Found {len(any_movers)} SKUs with positive growth")
                # Combine with new products
                new_products = merged_df[merged_df['wow_change'] >= 1000].copy()
                combined = pd.concat([any_movers.sort_values('wow_change', ascending=False), new_products]).head(limit)
                movers_df = combined
        
        # Format results
        decliners = []
        for _, row in decliners_df.iterrows():
            wow_change = float(row['wow_change'])
            # Cap at -100% for display (can't go below -100%)
            wow_change = max(wow_change, -100.0)
            decliners.append({
                'sku': str(row['sku']),
                'revenue': float(row['current_revenue']),
                'wow_change': wow_change,
            })
        
        movers = []
        for _, row in movers_df.iterrows():
            wow_change = float(row['wow_change'])
            # For new products (1000%), show as "New" or cap at reasonable value
            # We'll handle this in frontend, but for now cap at 999%
            if wow_change >= 1000:
                wow_change = 999.0  # Indicates new product
            movers.append({
                'sku': str(row['sku']),
                'revenue': float(row['current_revenue']),
                'wow_change': wow_change,
            })
        
        logger.info(f"📊 Final results: {len(movers)} movers and {len(decliners)} decliners")
        if decliners:
            logger.info(f"Decliners range: {min(d['wow_change'] for d in decliners):.1f}% to {max(d['wow_change'] for d in decliners):.1f}%")
        if movers:
            logger.info(f"Movers range: {min(m['wow_change'] for m in movers):.1f}% to {max(m['wow_change'] for m in movers):.1f}%")
        
        return {
            "movers": movers,
            "decliners": decliners,
        }
    
    except Exception as e:
        logger.error(f"Error getting movers and decliners: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"movers": [], "decliners": []}


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
        
        # DEBUG: Log ALL SKUs for selected region
        print("\n" + "="*80)
        print(f"🔍 DEBUG: REGION SKUs - Region: {region}")
        print("="*80)
        if df.empty:
            print(f"⚠️  No SKUs found for region: {region}")
        else:
            print(f"Total SKUs found: {len(df)}")
            print(f"\nAll SKUs for region '{region}':")
            for idx, row in df.iterrows():
                sku = row.get('sku', 'N/A')
                asin = row.get('asin', 'N/A')
                units = row.get('units', 0)
                revenue = row.get('revenue', 0)
                print(f"  {idx+1}. SKU: {sku}")
                print(f"     ASIN: {asin}")
                print(f"     Units Sold: {units:,}")
                print(f"     Revenue: ₹{revenue:,.2f}")
                print("-" * 80)
            print(f"\nTotal Units: {df['units'].sum():,}")
            print(f"Total Revenue: ₹{df['revenue'].sum():,.2f}")
            logger.info(f"Sample SKUs: {df['sku'].head(5).tolist()}")
            logger.info(f"Total units: {df['units'].sum()}, Total revenue: {df['revenue'].sum():.2f}")
        print("="*80 + "\n")
        
        # Ensure numeric columns
        if not df.empty:
            df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
            df['sku'] = df['sku'].astype(str).str.strip()
            df['asin'] = df['asin'].astype(str).str.strip()
        
        return df
    
    except Exception as e:
        logger.error(f"Error getting SKUs by region: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame(columns=['sku', 'asin', 'units', 'revenue'])
