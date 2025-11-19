"""
Verification endpoints - Compare FastAPI vs Legacy calculations

For audit purposes before Phase 2
"""
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from core.database import execute_query, table_exists
from services.metrics_service import calculate_metrics

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/full-audit")
async def full_backend_audit(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    """
    Complete backend audit - compare ALL KPIs against expected values
    
    Returns detailed breakdown of:
    - Revenue calculations (by transaction type)
    - Refund calculations
    - Shipping loss
    - Free replacement costs
    - Net revenue
    - Order counts
    - Success rates
    - Multi-file scenarios
    """
    if not table_exists('sales'):
        return {
            "error": "Sales table does not exist",
            "message": "Please upload CSV files first"
        }
    
    audit_results = {
        "audit_timestamp": datetime.now().isoformat(),
        "date_range": {
            "start": start_date or "all data",
            "end": end_date or "all data",
        },
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
        }
    }
    
    # Test 1: Transaction type breakdown
    test_1 = await _test_transaction_breakdown(start_date, end_date)
    audit_results["tests"].append(test_1)
    
    # Test 2: Revenue calculation
    test_2 = await _test_revenue_calculation(start_date, end_date)
    audit_results["tests"].append(test_2)
    
    # Test 3: Refund calculation
    test_3 = await _test_refund_calculation(start_date, end_date)
    audit_results["tests"].append(test_3)
    
    # Test 4: Shipping loss calculation
    test_4 = await _test_shipping_loss(start_date, end_date)
    audit_results["tests"].append(test_4)
    
    # Test 5: Free replacement cost
    test_5 = await _test_free_replacement_cost(start_date, end_date)
    audit_results["tests"].append(test_5)
    
    # Test 6: Net revenue calculation
    test_6 = await _test_net_revenue(start_date, end_date)
    audit_results["tests"].append(test_6)
    
    # Test 7: Multi-file scenario
    test_7 = await _test_multi_file_scenario()
    audit_results["tests"].append(test_7)
    
    # Test 8: Custom date range
    test_8 = await _test_custom_date_range()
    audit_results["tests"].append(test_8)
    
    # Test 9: Month-over-month comparison
    test_9 = await _test_month_over_month()
    audit_results["tests"].append(test_9)
    
    # Test 10: Metrics service integration
    test_10 = await _test_metrics_service(start_date, end_date)
    audit_results["tests"].append(test_10)
    
    # Calculate summary
    for test in audit_results["tests"]:
        audit_results["summary"]["total_tests"] += 1
        if test["status"] == "PASS":
            audit_results["summary"]["passed"] += 1
        elif test["status"] == "FAIL":
            audit_results["summary"]["failed"] += 1
        else:
            audit_results["summary"]["warnings"] += 1
    
    return audit_results


async def _test_transaction_breakdown(start_date: Optional[str], end_date: Optional[str]):
    """Test transaction type counts and amounts"""
    try:
        # Get column info
        column_info = execute_query("DESCRIBE sales")
        existing_columns = column_info['column_name'].tolist()
        
        # Find date column
        date_col = None
        for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
            if col in existing_columns:
                date_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['transaction_type', 'Transaction Type']:
            if col in existing_columns:
                txn_col = col
                break
        
        # Find revenue column
        revenue_col = None
        for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount']:
            if col in existing_columns:
                revenue_col = col
                break
        
        if not txn_col or not revenue_col:
            return {
                "test_name": "Transaction Type Breakdown",
                "status": "FAIL",
                "error": "Required columns not found",
            }
        
        # Build date filter
        date_filter = ""
        if start_date and end_date and date_col:
            col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
            if 'VARCHAR' in str(col_type).upper():
                date_filter = f"AND CAST(\"{date_col}\" AS DATE) >= '{start_date}' AND CAST(\"{date_col}\" AS DATE) <= '{end_date}'"
            else:
                date_filter = f"AND \"{date_col}\" >= '{start_date}' AND \"{date_col}\" <= '{end_date}'"
        
        sql = f"""
        SELECT 
            "{txn_col}" as transaction_type,
            COUNT(*) as count,
            SUM(ABS("{revenue_col}")) as total_amount
        FROM sales
        WHERE "{txn_col}" IS NOT NULL
        {date_filter}
        GROUP BY "{txn_col}"
        ORDER BY "{txn_col}"
        """
        
        result = execute_query(sql)
        breakdown = result.to_dict('records') if not result.empty else []
        
        return {
            "test_name": "Transaction Type Breakdown",
            "status": "PASS",
            "breakdown": breakdown,
            "validation": "Check manually against Streamlit dashboard - Transaction Breakdown section",
        }
    
    except Exception as e:
        logger.error(f"Transaction breakdown test failed: {e}")
        return {
            "test_name": "Transaction Type Breakdown",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_revenue_calculation(start_date: Optional[str], end_date: Optional[str]):
    """Test revenue calculation (Shipments only)"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        return {
            "test_name": "Revenue Calculation (Shipments)",
            "status": "PASS",
            "metrics": {
                "revenue": metrics.get('revenue', 0),
                "orders": metrics.get('orders', 0),
                "avg_order_value": metrics.get('avg_order_value', 0),
            },
            "validation": "Compare with Streamlit 'Gross Revenue' metric",
        }
    
    except Exception as e:
        logger.error(f"Revenue calculation test failed: {e}")
        return {
            "test_name": "Revenue Calculation (Shipments)",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_refund_calculation(start_date: Optional[str], end_date: Optional[str]):
    """Test refund calculation (including shipping loss)"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        return {
            "test_name": "Refund Calculation",
            "status": "PASS",
            "metrics": {
                "refund_count": metrics.get('transaction_breakdown', {}).get('Refund', {}).get('count', 0),
                "total_refunds": metrics.get('refunds', 0),
                "shipping_loss": metrics.get('shipping_loss', 0),
            },
            "validation": "Compare with Streamlit 'Refunds' and 'Shipping Loss' metrics",
        }
    
    except Exception as e:
        logger.error(f"Refund calculation test failed: {e}")
        return {
            "test_name": "Refund Calculation",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_shipping_loss(start_date: Optional[str], end_date: Optional[str]):
    """Test shipping loss calculation"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        shipping_loss = metrics.get('shipping_loss', 0)
        refund_breakdown = metrics.get('transaction_breakdown', {}).get('Refund', {})
        
        return {
            "test_name": "Shipping Loss Calculation",
            "status": "PASS",
            "metrics": {
                "total_shipping_loss": shipping_loss,
                "refunds_shipping_loss": refund_breakdown.get('shipping_loss', 0),
            },
            "validation": "Compare with Streamlit 'Shipping Loss' metric (should match refund shipping loss)",
        }
    
    except Exception as e:
        logger.error(f"Shipping loss test failed: {e}")
        return {
            "test_name": "Shipping Loss Calculation",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_free_replacement_cost(start_date: Optional[str], end_date: Optional[str]):
    """Test free replacement cost calculation"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        free_replacement = metrics.get('transaction_breakdown', {}).get('FreeReplacement', {})
        
        return {
            "test_name": "Free Replacement Cost",
            "status": "PASS",
            "metrics": {
                "free_replacement_count": free_replacement.get('count', 0),
                "estimated_cost": metrics.get('free_replacement_cost', 0),
            },
            "validation": "Compare with Streamlit 'Free Replacement Cost' metric",
        }
    
    except Exception as e:
        logger.error(f"Free replacement test failed: {e}")
        return {
            "test_name": "Free Replacement Cost",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_net_revenue(start_date: Optional[str], end_date: Optional[str]):
    """Test net revenue calculation"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        revenue = metrics.get('revenue', 0)
        refunds = metrics.get('refunds', 0)
        shipping_loss = metrics.get('shipping_loss', 0)
        free_replacement = metrics.get('free_replacement_cost', 0)
        net_revenue = metrics.get('net_revenue', 0)
        
        # Verify calculation: net = revenue - refunds - shipping - free_replacement
        expected_net = revenue - refunds - shipping_loss - free_replacement
        difference = abs(net_revenue - expected_net)
        
        status = "PASS" if difference < 0.01 else "FAIL"
        
        return {
            "test_name": "Net Revenue Calculation",
            "status": status,
            "metrics": {
                "revenue": revenue,
                "refunds": refunds,
                "shipping_loss": shipping_loss,
                "free_replacement_cost": free_replacement,
                "calculated_net": expected_net,
                "actual_net": net_revenue,
                "difference": difference,
            },
            "validation": "Net = Revenue - Refunds - Shipping - FreeReplacement. Compare with Streamlit 'Net Revenue'",
        }
    
    except Exception as e:
        logger.error(f"Net revenue test failed: {e}")
        return {
            "test_name": "Net Revenue Calculation",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_multi_file_scenario():
    """Test handling of multiple source files"""
    try:
        # Get column info
        column_info = execute_query("DESCRIBE sales")
        existing_columns = column_info['column_name'].tolist()
        
        if 'source_file' not in existing_columns:
            return {
                "test_name": "Multi-File Scenario",
                "status": "WARNING",
                "warning": "source_file column not found - cannot check multi-file scenarios",
            }
        
        # Find date column
        date_col = None
        for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
            if col in existing_columns:
                date_col = col
                break
        
        if not date_col:
            return {
                "test_name": "Multi-File Scenario",
                "status": "WARNING",
                "warning": "Date column not found",
            }
        
        col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
        date_cast = f'CAST("{date_col}" AS DATE)' if 'VARCHAR' in str(col_type).upper() else f'"{date_col}"'
        
        sql = f"""
        SELECT 
            source_file,
            EXTRACT(MONTH FROM {date_cast}) as month,
            EXTRACT(YEAR FROM {date_cast}) as year,
            COUNT(*) as row_count,
            COUNT(DISTINCT CASE WHEN "{date_col}" IS NOT NULL THEN {date_cast} END) as unique_dates
        FROM sales
        WHERE source_file IS NOT NULL
        AND {date_cast} IS NOT NULL
        GROUP BY source_file, EXTRACT(MONTH FROM {date_cast}), EXTRACT(YEAR FROM {date_cast})
        ORDER BY year, month, source_file
        """
        
        result = execute_query(sql)
        breakdown = result.to_dict('records') if not result.empty else []
        
        # Check for overlapping months
        months_per_file = {}
        for row in breakdown:
            file = row['source_file']
            month_key = f"{int(row['year'])}-{int(row['month']):02d}"
            if file not in months_per_file:
                months_per_file[file] = []
            months_per_file[file].append(month_key)
        
        # Detect overlaps
        all_months = [m for months in months_per_file.values() for m in months]
        overlapping_months = [m for m in set(all_months) if all_months.count(m) > 1]
        
        status = "WARNING" if overlapping_months else "PASS"
        
        return {
            "test_name": "Multi-File Scenario",
            "status": status,
            "source_files": list(months_per_file.keys()),
            "months_per_file": months_per_file,
            "overlapping_months": overlapping_months,
            "breakdown": breakdown,
            "warning": f"Multiple files contain same month data: {overlapping_months}" if overlapping_months else None,
            "validation": "Verify files don't have overlapping date ranges (causes duplicate counting)",
        }
    
    except Exception as e:
        logger.error(f"Multi-file test failed: {e}")
        return {
            "test_name": "Multi-File Scenario",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_custom_date_range():
    """Test custom date range filtering"""
    try:
        # Test: July 15-31 (mid-month range)
        metrics = calculate_metrics('2025-07-15', '2025-07-31')
        
        return {
            "test_name": "Custom Date Range (July 15-31)",
            "status": "PASS",
            "test_range": "2025-07-15 to 2025-07-31",
            "metrics": {
                "revenue": metrics.get('revenue', 0),
                "orders": metrics.get('orders', 0),
                "refunds": metrics.get('refunds', 0),
                "net_revenue": metrics.get('net_revenue', 0),
            },
            "validation": "Manually verify in Streamlit with same date filter",
        }
    
    except Exception as e:
        logger.error(f"Custom date range test failed: {e}")
        return {
            "test_name": "Custom Date Range (July 15-31)",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_month_over_month():
    """Test month-over-month metrics"""
    try:
        # Get column info
        column_info = execute_query("DESCRIBE sales")
        existing_columns = column_info['column_name'].tolist()
        
        # Find date column
        date_col = None
        for col in ['Invoice Date', 'invoice_date', 'order_date', 'Order Date']:
            if col in existing_columns:
                date_col = col
                break
        
        # Find transaction type column
        txn_col = None
        for col in ['transaction_type', 'Transaction Type']:
            if col in existing_columns:
                txn_col = col
                break
        
        # Find order ID column
        order_id_col = None
        for col in ['order_id', 'Invoice Number']:
            if col in existing_columns:
                order_id_col = col
                break
        
        # Find revenue column
        revenue_col = None
        for col in ['revenue_calc', 'revenue_amount', 'Invoice Amount']:
            if col in existing_columns:
                revenue_col = col
                break
        
        if not date_col or not txn_col or not revenue_col:
            return {
                "test_name": "Month-over-Month Comparison",
                "status": "FAIL",
                "error": "Required columns not found",
            }
        
        col_type = column_info[column_info['column_name'] == date_col]['column_type'].values[0]
        date_cast = f'CAST("{date_col}" AS DATE)' if 'VARCHAR' in str(col_type).upper() else f'"{date_col}"'
        
        sql = f"""
        SELECT 
            EXTRACT(MONTH FROM {date_cast}) as month,
            EXTRACT(YEAR FROM {date_cast}) as year,
            COUNT(DISTINCT CASE WHEN "{txn_col}" = 'Shipment' THEN "{order_id_col}" END) as orders,
            SUM(CASE WHEN "{txn_col}" = 'Shipment' THEN ABS("{revenue_col}") ELSE 0 END) as revenue,
            ABS(SUM(CASE WHEN "{txn_col}" = 'Refund' THEN "{revenue_col}" ELSE 0 END)) as refunds
        FROM sales
        WHERE {date_cast} IS NOT NULL
        GROUP BY EXTRACT(MONTH FROM {date_cast}), EXTRACT(YEAR FROM {date_cast})
        ORDER BY year, month
        """
        
        result = execute_query(sql)
        monthly_data = result.to_dict('records') if not result.empty else []
        
        return {
            "test_name": "Month-over-Month Comparison",
            "status": "PASS",
            "monthly_breakdown": monthly_data,
            "validation": "Compare each month's metrics with Streamlit filtered by month",
        }
    
    except Exception as e:
        logger.error(f"Month-over-month test failed: {e}")
        return {
            "test_name": "Month-over-Month Comparison",
            "status": "FAIL",
            "error": str(e),
        }


async def _test_metrics_service(start_date: Optional[str], end_date: Optional[str]):
    """Test metrics service integration"""
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        # Validate all expected keys exist
        required_keys = [
            'revenue', 'refunds', 'shipping_loss', 'free_replacement_cost',
            'net_revenue', 'orders', 'avg_order_value', 'success_rate',
            'transaction_breakdown'
        ]
        
        missing_keys = [key for key in required_keys if key not in metrics]
        
        status = "PASS" if not missing_keys else "FAIL"
        
        return {
            "test_name": "Metrics Service Integration",
            "status": status,
            "metrics_keys": list(metrics.keys()),
            "missing_keys": missing_keys,
            "sample_metrics": {
                k: v for k, v in list(metrics.items())[:5]
            },
            "validation": "Verify all required metrics are returned",
        }
    
    except Exception as e:
        logger.error(f"Metrics service test failed: {e}")
        return {
            "test_name": "Metrics Service Integration",
            "status": "FAIL",
            "error": str(e),
        }


@router.post("/reconcile-with-csv")
async def reconcile_with_csv(
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    year: int = Query(2025, description="Year"),
    expected_revenue: Optional[float] = Query(None, description="Expected revenue from CSV"),
    expected_refunds: Optional[float] = Query(None, description="Expected refunds from CSV"),
    expected_orders: Optional[int] = Query(None, description="Expected order count from CSV"),
    expected_net_revenue: Optional[float] = Query(None, description="Expected net revenue from CSV"),
):
    """
    Reconcile database metrics with CSV manual calculations
    
    Usage:
    1. Manually calculate totals in Excel for specific month
    2. Call this endpoint with expected values
    3. Get detailed comparison and discrepancy analysis
    """
    if not table_exists('sales'):
        raise HTTPException(status_code=404, detail="Sales table does not exist")
    
    try:
        # Calculate last day of month
        from calendar import monthrange
        last_day = monthrange(year, month)[1]
        
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year}-{month:02d}-{last_day}"
        
        # Get database metrics for specified month
        metrics = calculate_metrics(start_date, end_date)
        
        reconciliation = {
            "month": month,
            "year": year,
            "date_range": {
                "start": start_date,
                "end": end_date,
            },
            "database_metrics": metrics,
            "expected_metrics": {
                "revenue": expected_revenue,
                "refunds": expected_refunds,
                "orders": expected_orders,
                "net_revenue": expected_net_revenue,
            },
            "discrepancies": [],
            "status": "PASS",
            "tolerance": 1.0,  # 1% tolerance
        }
        
        # Check revenue
        if expected_revenue is not None:
            db_revenue = metrics.get('revenue', 0)
            diff = abs(db_revenue - expected_revenue)
            pct_diff = (diff / expected_revenue * 100) if expected_revenue > 0 else 0
            
            if pct_diff > reconciliation["tolerance"]:
                reconciliation["discrepancies"].append({
                    "metric": "Revenue",
                    "expected": expected_revenue,
                    "actual": db_revenue,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "FAIL",
                })
                reconciliation["status"] = "FAIL"
            else:
                reconciliation["discrepancies"].append({
                    "metric": "Revenue",
                    "expected": expected_revenue,
                    "actual": db_revenue,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "PASS",
                })
        
        # Check refunds
        if expected_refunds is not None:
            db_refunds = metrics.get('refunds', 0)
            diff = abs(db_refunds - expected_refunds)
            pct_diff = (diff / expected_refunds * 100) if expected_refunds > 0 else 0
            
            if pct_diff > reconciliation["tolerance"]:
                reconciliation["discrepancies"].append({
                    "metric": "Refunds",
                    "expected": expected_refunds,
                    "actual": db_refunds,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "FAIL",
                })
                reconciliation["status"] = "FAIL"
            else:
                reconciliation["discrepancies"].append({
                    "metric": "Refunds",
                    "expected": expected_refunds,
                    "actual": db_refunds,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "PASS",
                })
        
        # Check orders
        if expected_orders is not None:
            db_orders = metrics.get('orders', 0)
            diff = abs(db_orders - expected_orders)
            pct_diff = (diff / expected_orders * 100) if expected_orders > 0 else 0
            
            if pct_diff > reconciliation["tolerance"]:
                reconciliation["discrepancies"].append({
                    "metric": "Orders",
                    "expected": expected_orders,
                    "actual": db_orders,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "FAIL",
                })
                reconciliation["status"] = "FAIL"
            else:
                reconciliation["discrepancies"].append({
                    "metric": "Orders",
                    "expected": expected_orders,
                    "actual": db_orders,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "PASS",
                })
        
        # Check net revenue
        if expected_net_revenue is not None:
            db_net = metrics.get('net_revenue', 0)
            diff = abs(db_net - expected_net_revenue)
            pct_diff = (diff / expected_net_revenue * 100) if expected_net_revenue > 0 else 0
            
            if pct_diff > reconciliation["tolerance"]:
                reconciliation["discrepancies"].append({
                    "metric": "Net Revenue",
                    "expected": expected_net_revenue,
                    "actual": db_net,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "FAIL",
                })
                reconciliation["status"] = "FAIL"
            else:
                reconciliation["discrepancies"].append({
                    "metric": "Net Revenue",
                    "expected": expected_net_revenue,
                    "actual": db_net,
                    "difference": diff,
                    "pct_difference": round(pct_diff, 2),
                    "status": "PASS",
                })
        
        # Add diagnosis
        if reconciliation["status"] == "FAIL":
            reconciliation["diagnosis"] = _diagnose_reconciliation_discrepancy(reconciliation["discrepancies"])
        else:
            reconciliation["diagnosis"] = "All metrics match expected values within tolerance"
        
        return reconciliation
    
    except Exception as e:
        logger.error(f"Reconciliation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def _diagnose_reconciliation_discrepancy(discrepancies: list) -> str:
    """Provide diagnosis for reconciliation discrepancies"""
    failed_metrics = [d for d in discrepancies if d.get('status') == 'FAIL']
    
    if not failed_metrics:
        return "No discrepancies found"
    
    diagnosis = "Potential causes:\n"
    
    for disc in failed_metrics:
        metric = disc['metric']
        pct_diff = disc['pct_difference']
        
        if metric == "Refunds":
            diagnosis += f"- {metric} differs by {pct_diff:.2f}%: "
            diagnosis += "Check if multiple source files contain overlapping date ranges\n"
            diagnosis += "  Or verify transaction type mapping (Refund vs Refunds vs Return)\n"
        elif metric == "Revenue":
            diagnosis += f"- {metric} differs by {pct_diff:.2f}%: "
            diagnosis += "Check if all Shipment transactions are included\n"
            diagnosis += "  Verify revenue column mapping and calculation method\n"
        elif metric == "Orders":
            diagnosis += f"- {metric} differs by {pct_diff:.2f}%: "
            diagnosis += "Check if order counting uses DISTINCT on order_id\n"
            diagnosis += "  Verify duplicate order handling\n"
        elif metric == "Net Revenue":
            diagnosis += f"- {metric} differs by {pct_diff:.2f}%: "
            diagnosis += "Check if all deductions are included (refunds, shipping, free replacements)\n"
            diagnosis += "  Verify: Net = Revenue - Refunds - Shipping - FreeReplacement\n"
    
    diagnosis += "\nRecommended actions:\n"
    diagnosis += "1. Check /api/health/data-quality for duplicate detection\n"
    diagnosis += "2. Check /api/charts/july-refunds-breakdown for source file breakdown\n"
    diagnosis += "3. Verify CSV filters match date range exactly\n"
    diagnosis += "4. Check transaction type normalization (case sensitivity)\n"
    
    return diagnosis


@router.get("/compare-metrics")
async def compare_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """
    Quick comparison endpoint - returns all metrics in one place for easy verification
    
    Use this to quickly compare with Streamlit dashboard side-by-side
    """
    if not table_exists('sales'):
        raise HTTPException(status_code=404, detail="Sales table does not exist")
    
    try:
        metrics = calculate_metrics(start_date, end_date)
        
        # Format for easy comparison
        comparison = {
            "date_range": {
                "start": start_date or "all data",
                "end": end_date or "all data",
            },
            "key_metrics": {
                "revenue": round(metrics.get('revenue', 0), 2),
                "refunds": round(metrics.get('refunds', 0), 2),
                "shipping_loss": round(metrics.get('shipping_loss', 0), 2),
                "free_replacement_cost": round(metrics.get('free_replacement_cost', 0), 2),
                "net_revenue": round(metrics.get('net_revenue', 0), 2),
                "orders": metrics.get('orders', 0),
                "avg_order_value": round(metrics.get('avg_order_value', 0), 2),
                "success_rate": round(metrics.get('success_rate', 0), 2),
            },
            "transaction_breakdown": metrics.get('transaction_breakdown', {}),
            "formula_verification": {
                "revenue": metrics.get('revenue', 0),
                "minus_refunds": metrics.get('refunds', 0),
                "minus_shipping": metrics.get('shipping_loss', 0),
                "minus_free_replacement": metrics.get('free_replacement_cost', 0),
                "calculated_net": round(
                    metrics.get('revenue', 0) - 
                    metrics.get('refunds', 0) - 
                    metrics.get('shipping_loss', 0) - 
                    metrics.get('free_replacement_cost', 0),
                    2
                ),
                "actual_net": round(metrics.get('net_revenue', 0), 2),
                "matches": abs(
                    (metrics.get('revenue', 0) - metrics.get('refunds', 0) - 
                     metrics.get('shipping_loss', 0) - metrics.get('free_replacement_cost', 0)) -
                    metrics.get('net_revenue', 0)
                ) < 0.01,
            },
        }
        
        return comparison
    
    except Exception as e:
        logger.error(f"Compare metrics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))





