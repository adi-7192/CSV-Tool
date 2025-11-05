"""
Verification Script - Check Data Quality and Net Revenue Calculation

Run: python verify_fixes.py

Verifies:
- No duplicate records
- Net revenue calculated correctly
- All data quality checks pass
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

# Add backend to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import get_connection, table_exists

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def check_duplicates(conn) -> Tuple[bool, int]:
    """Check for duplicate business keys (order_id + transaction_type)"""
    dup_count = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT order_id, transaction_type, COUNT(*) AS cnt
            FROM sales
            GROUP BY order_id, transaction_type
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    return dup_count == 0, int(dup_count)


def check_net_revenue(conn) -> Tuple[bool, Dict[str, Any]]:
    """Check net revenue calculation using metrics service"""
    # Get metrics from the service (which handles all the complex logic)
    from backend.services.metrics_service import calculate_metrics
    metrics = calculate_metrics()
    
    revenue = metrics.get('revenue', 0)
    refunds = metrics.get('refunds', 0)
    shipping_loss = metrics.get('shipping_loss', 0)
    free_replacement_cost = metrics.get('free_replacement_cost', 0)
    actual_net = metrics.get('net_revenue', 0)
    
    # Calculate expected net revenue from components
    expected_net = revenue - refunds - shipping_loss - free_replacement_cost
    
    # Check if they match (allow small floating point differences)
    diff = abs(expected_net - actual_net)
    matches = diff < 0.01  # Allow 1 paisa tolerance
    
    return matches, {
        'revenue': revenue,
        'refunds': refunds,
        'shipping_loss': shipping_loss,
        'free_replacement_cost': free_replacement_cost,
        'expected_net': expected_net,
        'actual_net': actual_net,
        'difference': diff,
    }


def check_data_quality(conn) -> Dict[str, Any]:
    """Run comprehensive data quality checks"""
    checks = {}
    
    # Check for null order_ids
    null_order_ids = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE order_id IS NULL OR TRIM(order_id) = ''"
    ).fetchone()[0]
    checks['null_order_ids'] = int(null_order_ids)
    
    # Check for lineage columns
    missing_lineage = conn.execute(
        "SELECT COUNT(*) FROM sales WHERE source_file IS NULL OR ingestion_id IS NULL"
    ).fetchone()[0]
    checks['missing_lineage'] = int(missing_lineage)
    
    # Check transaction types
    txn_types = conn.execute(
        "SELECT transaction_type, COUNT(*) as cnt FROM sales GROUP BY transaction_type"
    ).fetchall()
    checks['transaction_types'] = {t: int(c) for t, c in txn_types if t}
    
    # Check total rows
    total_rows = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    checks['total_rows'] = int(total_rows)
    
    return checks


def main():
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}VERIFICATION SCRIPT - Data Quality & Net Revenue Check{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    # Check if database exists
    if not table_exists('sales'):
        print(f"{RED}❌ ERROR: Sales table not found in database{RESET}")
        sys.exit(1)
    
    # Get connection
    conn = get_connection()
    
    # Test 1: Check duplicates
    print(f"{CYAN}Test 1: Checking for duplicate records...{RESET}")
    no_duplicates, dup_count = check_duplicates(conn)
    if no_duplicates:
        print(f"{GREEN}✓ Duplicates: 0{RESET}\n")
    else:
        print(f"{RED}✗ Duplicates: {dup_count} duplicate groups found{RESET}\n")
    
    # Test 2: Check net revenue calculation
    print(f"{CYAN}Test 2: Verifying net revenue calculation...{RESET}")
    net_revenue_ok, net_details = check_net_revenue(conn)
    
    print(f"  Revenue: ₹{net_details['revenue']:,.2f}")
    print(f"  Refunds: ₹{net_details['refunds']:,.2f}")
    print(f"  Shipping Loss: ₹{net_details['shipping_loss']:,.2f}")
    print(f"  Free Replacement Cost: ₹{net_details['free_replacement_cost']:,.2f}")
    print(f"  Expected Net Revenue: ₹{net_details['expected_net']:,.2f}")
    print(f"  Actual Net Revenue: ₹{net_details['actual_net']:,.2f}")
    
    if net_revenue_ok:
        print(f"{GREEN}✓ Net Revenue calculated correctly{RESET}\n")
    else:
        print(f"{RED}✗ Net Revenue mismatch: Difference = ₹{net_details['difference']:,.2f}{RESET}\n")
    
    # Test 3: Data quality checks
    print(f"{CYAN}Test 3: Running data quality checks...{RESET}")
    quality = check_data_quality(conn)
    
    all_quality_ok = True
    
    if quality['null_order_ids'] == 0:
        print(f"{GREEN}✓ No null order_ids{RESET}")
    else:
        print(f"{RED}✗ Found {quality['null_order_ids']} rows with null order_id{RESET}")
        all_quality_ok = False
    
    if quality['missing_lineage'] == 0:
        print(f"{GREEN}✓ All rows have lineage tracking (source_file, ingestion_id){RESET}")
    else:
        print(f"{RED}✗ Found {quality['missing_lineage']} rows missing lineage{RESET}")
        all_quality_ok = False
    
    print(f"\n  Total Rows: {quality['total_rows']:,}")
    print(f"  Transaction Types:")
    for txn_type, count in quality['transaction_types'].items():
        print(f"    - {txn_type}: {count:,}")
    
    # Final summary
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}SUMMARY{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    all_tests_pass = no_duplicates and net_revenue_ok and all_quality_ok
    
    if all_tests_pass:
        print(f"{GREEN}✓ All tests pass!{RESET}")
        print(f"{GREEN}✓ Duplicates: 0{RESET}")
        print(f"{GREEN}✓ Net Revenue calculated correctly{RESET}")
        print(f"{GREEN}✓ Data quality checks passed{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}✗ Some tests failed{RESET}\n")
        if not no_duplicates:
            print(f"{RED}  - Duplicates found: {dup_count}{RESET}")
        if not net_revenue_ok:
            print(f"{RED}  - Net revenue calculation mismatch{RESET}")
        if not all_quality_ok:
            print(f"{RED}  - Data quality issues found{RESET}")
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()

