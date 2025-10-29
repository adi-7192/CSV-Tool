"""
Reconciliation Script - Validates Dashboard Metrics Against Source CSVs

Purpose: Ensure dashboard revenue, orders, and other metrics match raw CSV data.
Run this after uploads to verify data integrity.
"""

import pandas as pd
import duckdb
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = "data/analytics.duckdb"
RAW_DATA_PATH = "data/raw/"


def format_indian_currency(amount: float) -> str:
    """Format currency in Indian numbering system"""
    if pd.isna(amount) or amount == 0:
        return "₹0"
    
    abs_amount = abs(amount)
    
    # Convert to crores if >= 1 crore
    if abs_amount >= 10000000:
        crores = abs_amount / 10000000
        return f"₹{crores:.2f} Cr" if amount >= 0 else f"-₹{crores:.2f} Cr"
    # Convert to lakhs if >= 1 lakh
    elif abs_amount >= 100000:
        lakhs = abs_amount / 100000
        return f"₹{lakhs:.2f} L" if amount >= 0 else f"-₹{lakhs:.2f} L"
    else:
        return f"₹{abs_amount:,.2f}" if amount >= 0 else f"-₹{abs_amount:,.2f}"


def load_raw_csv_totals(csv_path: str) -> Dict:
    """
    Calculate totals directly from raw CSV file.
    
    Args:
        csv_path: Path to raw CSV file
        
    Returns:
        Dictionary with calculated totals
    """
    logger.info(f"📄 Loading raw CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Auto-detect column names (handle different naming conventions)
    amount_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['amount', 'revenue', 'total', 'invoice amount', 'order amount']):
            # Prefer "invoice amount" if available
            if 'invoice' in col_lower:
                amount_col = col
                break
            elif not amount_col:  # Take first match as fallback
                amount_col = col
    
    type_col = None
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['type', 'transaction', 'order type']):
            if 'transaction' in col_lower:
                type_col = col
                break
            elif not type_col:
                type_col = col
    
    if not amount_col:
        # Try numeric columns as fallback
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            amount_col = numeric_cols[0]
            logger.warning(f"   ⚠️ Auto-selected numeric column as amount: {amount_col}")
        else:
            raise ValueError(f"Cannot find amount/revenue column in {csv_path}. Available columns: {list(df.columns)}")
    
    logger.info(f"   Using amount column: {amount_col}")
    if type_col:
        logger.info(f"   Using transaction type column: {type_col}")
    
    # Convert amount column to numeric (handle currency symbols, commas)
    df[amount_col] = df[amount_col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    
    # Calculate totals
    totals = {
        'file': os.path.basename(csv_path),
        'file_path': csv_path,
        'total_rows': len(df),
        'total_amount_raw': df[amount_col].sum(),
        'amount_column': amount_col
    }
    
    if type_col:
        # Normalize transaction type column
        df[type_col] = df[type_col].astype(str).str.strip()
        
        # Calculate by transaction type
        shipments = df[df[type_col].str.contains('Shipment|shipment', case=False, na=False)]
        refunds = df[df[type_col].str.contains('Refund|refund', case=False, na=False)]
        cancels = df[df[type_col].str.contains('Cancel|cancel', case=False, na=False)]
        free_replacements = df[df[type_col].str.contains('Free|free', case=False, na=False)]
        
        totals['shipment_revenue'] = shipments[amount_col].sum() if len(shipments) > 0 else 0
        totals['refund_amount'] = abs(refunds[amount_col].sum()) if len(refunds) > 0 else 0
        totals['shipment_count'] = len(shipments)
        totals['refund_count'] = len(refunds)
        totals['cancel_count'] = len(cancels)
        totals['free_replacement_count'] = len(free_replacements)
        totals['transaction_type_column'] = type_col
        
        logger.info(f"   Shipments: {len(shipments)} rows, Revenue: {format_indian_currency(totals['shipment_revenue'])}")
        logger.info(f"   Refunds: {len(refunds)} rows, Amount: {format_indian_currency(totals['refund_amount'])}")
    else:
        logger.warning("   ⚠️ No transaction type column found - using total amounts only")
        totals['shipment_revenue'] = totals['total_amount_raw']  # Assume all are shipments if no type
        totals['refund_amount'] = 0
        totals['shipment_count'] = len(df)
        totals['refund_count'] = 0
    
    return totals


def load_database_totals(db_path: str = DB_PATH) -> Dict:
    """
    Calculate totals from DuckDB database.
    
    Args:
        db_path: Path to DuckDB database file
        
    Returns:
        Dictionary with database totals
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    logger.info(f"📊 Querying database: {db_path}")
    
    try:
        conn = duckdb.connect(db_path, read_only=True)
        
        # Check if sales table exists
        table_exists = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'sales'").fetchone()[0]
        if not table_exists:
            raise ValueError("Sales table does not exist in database")
        
        # Total rows
        total_rows_result = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
        total_rows = total_rows_result[0] if total_rows_result else 0
        
        # Get column names to handle different schemas
        columns_info = conn.execute("DESCRIBE sales").df()
        column_names = columns_info['column_name'].tolist()
        
        # Find revenue column (could be Invoice Amount, revenue_calc, etc.)
        revenue_col = None
        for col in column_names:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['revenue_calc', 'invoice amount', 'revenue_amount']):
                if 'revenue_calc' in col_lower or 'revenue_amount' in col_lower:
                    revenue_col = col
                    break
                elif not revenue_col:
                    revenue_col = col
        
        # Fallback to Invoice Amount if revenue_calc not found
        if not revenue_col:
            for col in column_names:
                if 'amount' in col.lower():
                    revenue_col = col
                    break
        
        if not revenue_col:
            raise ValueError(f"Cannot find revenue column in database. Available columns: {column_names}")
        
        logger.info(f"   Using revenue column: {revenue_col}")
        
        # Check if transaction_type column exists
        has_transaction_type = 'transaction_type' in column_names or 'Transaction Type' in column_names
        txn_col = 'transaction_type' if 'transaction_type' in column_names else 'Transaction Type' if 'Transaction Type' in column_names else None
        
        # Revenue metrics (shipments only)
        if has_transaction_type:
            shipment_sql = f"""
            SELECT
                COUNT(*) as count,
                SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END) as total_revenue
            FROM sales
            WHERE ({txn_col} = 'Shipment' OR {txn_col} LIKE '%Shipment%')
            """
            shipment_result = conn.execute(shipment_sql).fetchone()
            shipment_count = shipment_result[0] if shipment_result else 0
            shipment_revenue = shipment_result[1] if shipment_result and shipment_result[1] else 0
            
            # Refund metrics
            refund_sql = f"""
            SELECT
                COUNT(*) as count,
                ABS(SUM(CASE WHEN {revenue_col} < 0 THEN {revenue_col} ELSE 0 END)) as total_refunds
            FROM sales
            WHERE ({txn_col} = 'Refund' OR {txn_col} LIKE '%Refund%')
            """
            refund_result = conn.execute(refund_sql).fetchone()
            refund_count = refund_result[0] if refund_result else 0
            refund_amount = refund_result[1] if refund_result and refund_result[1] else 0
            
            # All revenue (positive amounts)
            all_revenue_sql = f"""
            SELECT SUM({revenue_col}) as total
            FROM sales
            WHERE {revenue_col} > 0
            """
            all_revenue_result = conn.execute(all_revenue_sql).fetchone()
            all_revenue = all_revenue_result[0] if all_revenue_result and all_revenue_result[0] else 0
        else:
            # No transaction type - use positive amounts as shipments
            shipment_sql = f"""
            SELECT
                COUNT(*) as count,
                SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END) as total_revenue
            FROM sales
            WHERE {revenue_col} > 0
            """
            shipment_result = conn.execute(shipment_sql).fetchone()
            shipment_count = shipment_result[0] if shipment_result else 0
            shipment_revenue = shipment_result[1] if shipment_result and shipment_result[1] else 0
            
            refund_sql = f"""
            SELECT
                COUNT(*) as count,
                ABS(SUM(CASE WHEN {revenue_col} < 0 THEN {revenue_col} ELSE 0 END)) as total_refunds
            FROM sales
            WHERE {revenue_col} < 0
            """
            refund_result = conn.execute(refund_sql).fetchone()
            refund_count = refund_result[0] if refund_result else 0
            refund_amount = refund_result[1] if refund_result and refund_result[1] else 0
            
            all_revenue_sql = f"""
            SELECT SUM(CASE WHEN {revenue_col} > 0 THEN {revenue_col} ELSE 0 END) as total
            FROM sales
            """
            all_revenue_result = conn.execute(all_revenue_sql).fetchone()
            all_revenue = all_revenue_result[0] if all_revenue_result and all_revenue_result[0] else 0
        
        conn.close()
        
        return {
            'total_rows': total_rows,
            'shipment_count': shipment_count,
            'shipment_revenue': shipment_revenue,
            'refund_count': refund_count,
            'refund_amount': refund_amount,
            'all_revenue': all_revenue,
            'revenue_column': revenue_col
        }
        
    except Exception as e:
        logger.error(f"❌ Error querying database: {e}")
        raise


def compare_totals(csv_totals: Dict, db_totals: Dict, tolerance: float = 1.0) -> Tuple[bool, List[Dict]]:
    """
    Compare CSV totals vs database totals.
    
    Args:
        csv_totals: Totals calculated from CSV file
        db_totals: Totals calculated from database
        tolerance: Acceptable percentage difference (default 1%)
        
    Returns:
        Tuple of (passed: bool, discrepancies: List[Dict])
    """
    discrepancies = []
    passed = True
    
    # Compare shipment revenue (most important metric)
    if 'shipment_revenue' in csv_totals and 'shipment_revenue' in db_totals:
        csv_rev = csv_totals['shipment_revenue']
        db_rev = db_totals['shipment_revenue']
        
        if csv_rev == 0 and db_rev == 0:
            # Both zero - pass
            discrepancies.append({
                'metric': 'Shipment Revenue',
                'csv_value': csv_rev,
                'db_value': db_rev,
                'difference': 0,
                'pct_difference': 0.0,
                'status': 'PASS',
                'csv_display': format_indian_currency(csv_rev),
                'db_display': format_indian_currency(db_rev)
            })
        elif csv_rev == 0:
            passed = False
            discrepancies.append({
                'metric': 'Shipment Revenue',
                'csv_value': csv_rev,
                'db_value': db_rev,
                'difference': db_rev,
                'pct_difference': 100.0,
                'status': 'FAIL - CSV has no revenue but DB has revenue',
                'csv_display': format_indian_currency(csv_rev),
                'db_display': format_indian_currency(db_rev)
            })
        else:
            diff = abs(csv_rev - db_rev)
            pct_diff = (diff / abs(csv_rev)) * 100
            
            if pct_diff > tolerance:
                passed = False
                discrepancies.append({
                    'metric': 'Shipment Revenue',
                    'csv_value': csv_rev,
                    'db_value': db_rev,
                    'difference': diff,
                    'pct_difference': pct_diff,
                    'status': f'FAIL - Difference exceeds {tolerance}% tolerance',
                    'csv_display': format_indian_currency(csv_rev),
                    'db_display': format_indian_currency(db_rev)
                })
            else:
                discrepancies.append({
                    'metric': 'Shipment Revenue',
                    'csv_value': csv_rev,
                    'db_value': db_rev,
                    'difference': diff,
                    'pct_difference': pct_diff,
                    'status': 'PASS',
                    'csv_display': format_indian_currency(csv_rev),
                    'db_display': format_indian_currency(db_rev)
                })
    
    # Compare row counts
    if 'total_rows' in csv_totals and 'total_rows' in db_totals:
        csv_rows = csv_totals['total_rows']
        db_rows = db_totals['total_rows']
        
        # Allow for deduplication (DB can have fewer rows)
        if db_rows > csv_rows * 1.1:  # More than 10% extra rows is suspicious
            passed = False
            discrepancies.append({
                'metric': 'Row Count',
                'csv_value': csv_rows,
                'db_value': db_rows,
                'difference': db_rows - csv_rows,
                'pct_difference': ((db_rows - csv_rows) / csv_rows * 100) if csv_rows > 0 else 0,
                'status': f'FAIL - Database has {db_rows - csv_rows} MORE rows than CSV (possible duplicate inserts)',
                'csv_display': f"{csv_rows:,}",
                'db_display': f"{db_rows:,}"
            })
        elif csv_rows == db_rows:
            discrepancies.append({
                'metric': 'Row Count',
                'csv_value': csv_rows,
                'db_value': db_rows,
                'difference': 0,
                'pct_difference': 0.0,
                'status': 'PASS - Exact match',
                'csv_display': f"{csv_rows:,}",
                'db_display': f"{db_rows:,}"
            })
        else:
            # DB has fewer rows (deduplication expected)
            discrepancies.append({
                'metric': 'Row Count',
                'csv_value': csv_rows,
                'db_value': db_rows,
                'difference': csv_rows - db_rows,
                'pct_difference': ((csv_rows - db_rows) / csv_rows * 100) if csv_rows > 0 else 0,
                'status': f'PASS - {csv_rows - db_rows} duplicates removed (expected)',
                'csv_display': f"{csv_rows:,}",
                'db_display': f"{db_rows:,}"
            })
    
    # Compare refund amounts (if available)
    if 'refund_amount' in csv_totals and 'refund_amount' in db_totals:
        csv_refund = csv_totals['refund_amount']
        db_refund = db_totals['refund_amount']
        
        if csv_refund == 0 and db_refund == 0:
            discrepancies.append({
                'metric': 'Refund Amount',
                'csv_value': csv_refund,
                'db_value': db_refund,
                'difference': 0,
                'pct_difference': 0.0,
                'status': 'PASS',
                'csv_display': format_indian_currency(csv_refund),
                'db_display': format_indian_currency(db_refund)
            })
        elif csv_refund == 0 or db_refund == 0:
            # One has refunds, other doesn't - might be OK if CSV had no refunds
            if db_refund > 0:
                discrepancies.append({
                    'metric': 'Refund Amount',
                    'csv_value': csv_refund,
                    'db_value': db_refund,
                    'difference': db_refund,
                    'status': 'INFO - DB has refunds but CSV had none (may be from other uploads)',
                    'csv_display': format_indian_currency(csv_refund),
                    'db_display': format_indian_currency(db_refund)
                })
        else:
            diff = abs(csv_refund - db_refund)
            pct_diff = (diff / csv_refund) * 100 if csv_refund > 0 else 0
            
            if pct_diff > tolerance:
                passed = False
                discrepancies.append({
                    'metric': 'Refund Amount',
                    'csv_value': csv_refund,
                    'db_value': db_refund,
                    'difference': diff,
                    'pct_difference': pct_diff,
                    'status': f'FAIL - Difference exceeds {tolerance}% tolerance',
                    'csv_display': format_indian_currency(csv_refund),
                    'db_display': format_indian_currency(db_refund)
                })
            else:
                discrepancies.append({
                    'metric': 'Refund Amount',
                    'csv_value': csv_refund,
                    'db_value': db_refund,
                    'difference': diff,
                    'pct_difference': pct_diff,
                    'status': 'PASS',
                    'csv_display': format_indian_currency(csv_refund),
                    'db_display': format_indian_currency(db_refund)
                })
    
    return passed, discrepancies


def print_reconciliation_report(csv_files: List[str], tolerance: float = 1.0, db_path: str = DB_PATH):
    """
    Generate and print reconciliation report.
    
    Args:
        csv_files: List of CSV filenames to reconcile
        tolerance: Acceptable percentage difference
        db_path: Path to DuckDB database
    """
    print("\n" + "="*80)
    print("📊 DATA RECONCILIATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Load database totals
    try:
        db_totals = load_database_totals(db_path)
        print(f"📊 **Database Totals:**")
        print(f"   Total Rows: {db_totals['total_rows']:,}")
        print(f"   Shipment Revenue: {format_indian_currency(db_totals['shipment_revenue'])}")
        print(f"   Shipment Count: {db_totals['shipment_count']:,}")
        print(f"   Refund Amount: {format_indian_currency(db_totals['refund_amount'])}")
        print(f"   Refund Count: {db_totals['refund_count']:,}")
        print()
    except Exception as e:
        print(f"❌ Error loading database totals: {e}")
        return
    
    # Process each CSV file
    all_passed = True
    csv_file_path = Path(RAW_DATA_PATH)
    
    for csv_file in csv_files:
        csv_path = csv_file_path / csv_file
        if not csv_path.exists():
            print(f"⚠️ File not found: {csv_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"📄 **Checking: {csv_file}**")
        print("-" * 80)
        
        try:
            # Load CSV totals
            csv_totals = load_raw_csv_totals(str(csv_path))
            print(f"\n📄 **CSV Totals:**")
            print(f"   Total Rows: {csv_totals['total_rows']:,}")
            print(f"   Shipment Revenue: {format_indian_currency(csv_totals.get('shipment_revenue', 0))}")
            print(f"   Shipment Count: {csv_totals.get('shipment_count', 0):,}")
            if 'refund_amount' in csv_totals:
                print(f"   Refund Amount: {format_indian_currency(csv_totals['refund_amount'])}")
                print(f"   Refund Count: {csv_totals.get('refund_count', 0):,}")
            
            # Compare
            passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance)
            
            print(f"\n{'='*80}")
            if passed:
                print("✅ **RECONCILIATION PASSED**")
            else:
                print("❌ **RECONCILIATION FAILED**")
                all_passed = False
            
            # Print details
            for item in discrepancies:
                status_icon = "✅" if "PASS" in item['status'] else "❌" if "FAIL" in item['status'] else "ℹ️"
                print(f"\n{status_icon} **{item['metric']}:**")
                print(f"   CSV Value:  {item['csv_display']}")
                print(f"   DB Value:   {item['db_display']}")
                if 'difference' in item:
                    print(f"   Difference: {item['difference']:,.2f}")
                if 'pct_difference' in item:
                    print(f"   % Difference: {item['pct_difference']:.4f}%")
                print(f"   Status:     {item['status']}")
                
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            logger.exception("Full error details:")
            all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ **OVERALL STATUS: ALL RECONCILIATIONS PASSED**")
    else:
        print("❌ **OVERALL STATUS: RECONCILIATION FAILURES DETECTED**")
        print("⚠️  Please review discrepancies above and investigate data processing issues.")
    print("="*80 + "\n")
    
    return all_passed


def reconcile_single_file(csv_file_path: str, tolerance: float = 1.0, db_path: str = DB_PATH) -> Tuple[bool, List[Dict]]:
    """
    Reconcile a single CSV file against database.
    Useful for integration into upload workflow.
    
    Args:
        csv_file_path: Path to CSV file
        tolerance: Acceptable percentage difference
        db_path: Path to DuckDB database
        
    Returns:
        Tuple of (passed: bool, discrepancies: List[Dict])
    """
    try:
        csv_totals = load_raw_csv_totals(csv_file_path)
        db_totals = load_database_totals(db_path)
        passed, discrepancies = compare_totals(csv_totals, db_totals, tolerance)
        return passed, discrepancies
    except Exception as e:
        logger.error(f"Reconciliation error: {e}")
        return False, [{'metric': 'Error', 'status': f'FAIL - {str(e)}'}]


if __name__ == "__main__":
    import sys
    
    # List all CSV files in raw data folder
    csv_file_path = Path(RAW_DATA_PATH)
    
    if not csv_file_path.exists():
        print(f"⚠️ Directory not found: {RAW_DATA_PATH}")
        print("Please create data/raw/ directory and upload CSV files first")
        sys.exit(1)
    
    csv_files = list(csv_file_path.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in {RAW_DATA_PATH}")
        print("Please upload CSV files to data/raw/ first")
        sys.exit(1)
    else:
        # Run reconciliation with 1% tolerance
        csv_filenames = [f.name for f in csv_files]
        success = print_reconciliation_report(csv_filenames, tolerance=1.0)
        sys.exit(0 if success else 1)

