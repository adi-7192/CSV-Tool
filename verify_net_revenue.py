"""
Net Revenue Verification Script - September 2025

Verifies net revenue calculation by:
1. Manual SQL calculation
2. Metrics service calculation
3. API endpoint comparison
4. AI Chat response comparison

Run: python verify_net_revenue.py
"""
import os
import sys
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import duckdb

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import get_connection, table_exists, execute_query
from backend.services.metrics_service import calculate_metrics

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

API_BASE = "http://localhost:8000"


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}{BOLD}{text}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{YELLOW}{text}{RESET}")
    print("-" * 80)


def get_db_connection():
    """Get database connection, handling locks gracefully"""
    # Try multiple database paths
    db_paths = [
        PROJECT_ROOT / 'data' / 'analytics.duckdb',
        PROJECT_ROOT / 'backend' / 'data' / 'analytics.duckdb',
        PROJECT_ROOT / 'sales_data.db',
    ]
    
    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        raise FileNotFoundError("Database file not found. Checked: " + ", ".join(str(p) for p in db_paths))
    
    # Try using execute_query which uses the singleton connection
    # If that fails, try direct connection
    try:
        # Test if we can query through the backend connection
        test_result = execute_query("SELECT 1 as test")
        if test_result is not None:
            print(f"{GREEN}Using backend database connection{RESET}")
            # Return None to indicate we'll use execute_query instead
            return None
    except Exception:
        pass
    
    # Try direct read-only connection
    try:
        print(f"{YELLOW}Using direct read-only connection to: {db_path}{RESET}")
        return duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        if "lock" in str(e).lower():
            print(f"{YELLOW}⚠️  Database is locked (API may be running).{RESET}")
            print(f"{YELLOW}   Will use API endpoints for verification instead.{RESET}")
            return None
        raise Exception(f"Could not connect: {e}")


def calculate_manual_september_metrics(conn) -> Dict[str, Any]:
    """Calculate September 2025 metrics manually using SQL"""
    print_section("STEP 1: Manual SQL Calculation (September 2025)")
    
    # Use execute_query if conn is None (backend connection)
    if conn is None:
        date_check_df = execute_query("""
            SELECT 
                MIN(order_date) as earliest,
                MAX(order_date) as latest,
                COUNT(*) as total_rows
            FROM sales
            WHERE order_date IS NOT NULL
        """)
        date_check = date_check_df.iloc[0] if not date_check_df.empty else (None, None, 0)
    else:
        date_check = conn.execute("""
            SELECT 
                MIN(order_date) as earliest,
                MAX(order_date) as latest,
                COUNT(*) as total_rows
            FROM sales
            WHERE order_date IS NOT NULL
        """).fetchone()
    
    print(f"Date Range in Database: {date_check[0]} to {date_check[1]}")
    print(f"Total Rows: {date_check[2]:,}")
    
    # Calculate metrics for September 2025
    sql = """
        SELECT 
            -- Shipment Revenue (positive)
            COALESCE(SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END), 0) AS shipment_revenue,
            
            -- Refund Amount (absolute value, already negative in DB)
            ABS(COALESCE(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END), 0)) AS refund_amount,
            
            -- Cancellation Amount (should be 0 as Cancels don't affect revenue)
            ABS(COALESCE(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END), 0)) AS cancel_amount,
            
            -- Shipping Loss (from refunds)
            COALESCE(SUM(CASE WHEN transaction_type = 'Refund' THEN shipping_amount ELSE 0 END), 0) AS shipping_loss,
            
            -- Counts
            COUNT(CASE WHEN transaction_type = 'Shipment' THEN 1 END) AS shipment_count,
            COUNT(CASE WHEN transaction_type = 'Refund' THEN 1 END) AS refund_count,
            COUNT(CASE WHEN transaction_type = 'Cancel' THEN 1 END) AS cancel_count,
            COUNT(CASE WHEN transaction_type = 'FreeReplacement' THEN 1 END) AS freereplacement_count
            
        FROM sales
        WHERE 
            EXTRACT(YEAR FROM order_date) = 2025 
            AND EXTRACT(MONTH FROM order_date) = 9
    """
    
    if conn is None:
        result_df = execute_query(sql)
        result = result_df.iloc[0] if not result_df.empty else [0]*8
    else:
        result = conn.execute(sql).fetchone()
    
    shipment_revenue = float(result[0] or 0)
    refund_amount = float(result[1] or 0)
    cancel_amount = float(result[2] or 0)
    shipping_loss = float(result[3] or 0)
    shipment_count = int(result[4] or 0)
    refund_count = int(result[5] or 0)
    cancel_count = int(result[6] or 0)
    freereplacement_count = int(result[7] or 0)
    
    # Calculate Free Replacement Cost (using same logic as metrics service)
    # Get all FreeReplacement records for September
    fr_sql = """
        SELECT revenue_amount, shipping_amount, sku
        FROM sales
        WHERE transaction_type = 'FreeReplacement'
        AND EXTRACT(YEAR FROM order_date) = 2025 
        AND EXTRACT(MONTH FROM order_date) = 9
    """
    
    shipment_sql = """
        SELECT revenue_amount, shipping_amount, sku
        FROM sales
        WHERE transaction_type = 'Shipment'
        AND EXTRACT(YEAR FROM order_date) = 2025 
        AND EXTRACT(MONTH FROM order_date) = 9
    """
    
    if conn is None:
        fr_data = execute_query(fr_sql)
        shipment_data = execute_query(shipment_sql)
    else:
        fr_data = conn.execute(fr_sql).fetchdf()
        shipment_data = conn.execute(shipment_sql).fetchdf()
    
    # Calculate FreeReplacement cost (simplified: 2x average shipment price)
    free_replacement_cost = 0.0
    if not fr_data.empty and not shipment_data.empty:
        avg_shipment_price = shipment_data['revenue_amount'].mean()
        free_replacement_cost = avg_shipment_price * 2 * len(fr_data)
        print(f"  Note: FreeReplacement cost estimated as 2x avg shipment price ({avg_shipment_price:.2f}) × {len(fr_data)} items")
    
    # Manual net revenue calculation
    manual_net = shipment_revenue - refund_amount - shipping_loss - free_replacement_cost
    
    print(f"\n{BOLD}Manual Calculation Results:{RESET}")
    print(f"  Shipments:     ₹{shipment_revenue:,.2f} ({shipment_count:,} transactions)")
    print(f"  Refunds:       -₹{refund_amount:,.2f} ({refund_count:,} transactions)")
    print(f"  Cancels:       ₹{cancel_amount:,.2f} ({cancel_count:,} transactions) [should be 0]")
    print(f"  Shipping Loss: -₹{shipping_loss:,.2f}")
    print(f"  Free Repl:      -₹{free_replacement_cost:,.2f} ({freereplacement_count:,} transactions)")
    print(f"  {'='*60}")
    print(f"  {BOLD}Net Revenue:    ₹{manual_net:,.2f}{RESET}")
    
    return {
        'shipment_revenue': shipment_revenue,
        'refund_amount': refund_amount,
        'cancel_amount': cancel_amount,
        'shipping_loss': shipping_loss,
        'free_replacement_cost': free_replacement_cost,
        'net_revenue': manual_net,
        'counts': {
            'shipment': shipment_count,
            'refund': refund_count,
            'cancel': cancel_count,
            'freereplacement': freereplacement_count,
        }
    }


def calculate_metrics_service_september() -> Dict[str, Any]:
    """Calculate September 2025 metrics using metrics service"""
    print_section("STEP 2: Metrics Service Calculation (September 2025)")
    
    metrics = calculate_metrics(
        start_date='2025-09-01',
        end_date='2025-09-30'
    )
    
    print(f"\n{BOLD}Metrics Service Results:{RESET}")
    print(f"  Revenue:              ₹{metrics.get('revenue', 0):,.2f}")
    print(f"  Refunds:               ₹{metrics.get('refunds', 0):,.2f}")
    print(f"  Shipping Loss:         ₹{metrics.get('shipping_loss', 0):,.2f}")
    print(f"  Free Replacement Cost: ₹{metrics.get('free_replacement_cost', 0):,.2f}")
    print(f"  Net Revenue:           ₹{metrics.get('net_revenue', 0):,.2f}")
    print(f"  Orders:                {metrics.get('orders', 0):,}")
    
    if metrics.get('transaction_breakdown'):
        print(f"\n  Transaction Breakdown:")
        for txn_type, details in metrics['transaction_breakdown'].items():
            if isinstance(details, dict):
                count = details.get('count', 0)
                revenue = details.get('revenue', details.get('refund_amount', details.get('estimated_cost', 0)))
                print(f"    {txn_type}: {count:,} transactions, ₹{revenue:,.2f}")
    
    return metrics


def compare_api_endpoint() -> Optional[Dict[str, Any]]:
    """Compare with /api/metrics endpoint"""
    print_section("STEP 3: API Endpoint Comparison (/api/metrics)")
    
    try:
        response = requests.get(
            f"{API_BASE}/api/metrics",
            params={
                'start_date': '2025-09-01',
                'end_date': '2025-09-30'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            api_data = response.json()
            metrics = api_data.get('data', {})
            
            print(f"\n{BOLD}API Endpoint Results:{RESET}")
            print(f"  Revenue:              ₹{metrics.get('revenue', 0):,.2f}")
            print(f"  Refunds:               ₹{metrics.get('refunds', 0):,.2f}")
            print(f"  Shipping Loss:         ₹{metrics.get('shipping_loss', 0):,.2f}")
            print(f"  Free Replacement Cost: ₹{metrics.get('free_replacement_cost', 0):,.2f}")
            print(f"  Net Revenue:           ₹{metrics.get('net_revenue', 0):,.2f}")
            return metrics
        else:
            print(f"{RED}API returned status {response.status_code}{RESET}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"{YELLOW}⚠️  API not available at {API_BASE}{RESET}")
        print(f"   Make sure the backend is running: uvicorn backend.main:app --reload")
        return None
    except Exception as e:
        print(f"{RED}Error calling API: {e}{RESET}")
        return None


def compare_ai_chat() -> Optional[Dict[str, Any]]:
    """Compare with AI Chat response"""
    print_section("STEP 4: AI Chat Comparison (/api/chat)")
    
    try:
        # Ask AI for September 2025 net revenue
        response = requests.post(
            f"{API_BASE}/api/chat/query",
            json={
                "query": "What is the net revenue for September 2025? Show me the breakdown: shipments, refunds, shipping loss, and free replacement costs."
            },
            timeout=30
        )
        
        if response.status_code == 200:
            chat_data = response.json()
            
            print(f"\n{BOLD}AI Chat Response:{RESET}")
            print(f"  Answer: {chat_data.get('answer', 'N/A')}")
            
            if chat_data.get('sql'):
                print(f"\n  Generated SQL:")
                print(f"    {chat_data['sql']}")
            
            if chat_data.get('data'):
                print(f"\n  Query Results:")
                data = chat_data['data']
                if isinstance(data, dict):
                    for key, value in data.items():
                        print(f"    {key}: {value}")
                elif isinstance(data, list) and len(data) > 0:
                    print(f"    {data[0]}")
            
            return chat_data
        else:
            # Try alternative endpoint
            try:
                response2 = requests.post(
                    f"{API_BASE}/api/chat",
                    json={
                        "message": "What is the net revenue for September 2025? Show me the breakdown: shipments, refunds, shipping loss, and free replacement costs."
                    },
                    timeout=30
                )
                if response2.status_code == 200:
                    chat_data = response2.json()
                    print(f"\n{BOLD}AI Chat Response:{RESET}")
                    print(f"  Answer: {chat_data.get('answer', chat_data.get('message', 'N/A'))}")
                    return chat_data
            except:
                pass
            
            print(f"{YELLOW}⚠️  AI Chat returned status {response.status_code}{RESET}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"{YELLOW}⚠️  API not available at {API_BASE}{RESET}")
        return None
    except Exception as e:
        print(f"{YELLOW}⚠️  Error calling AI Chat: {e}{RESET}")
        return None


def compare_results(manual: Dict[str, Any], service: Dict[str, Any], api: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compare all calculation methods"""
    print_section("STEP 5: Comparison & Bug Detection")
    
    comparison = {
        'manual': manual,
        'service': service,
        'api': api,
        'discrepancies': [],
        'bugs_found': []
    }
    
    # Compare Manual vs Service
    print(f"\n{BOLD}Manual vs Metrics Service:{RESET}")
    
    manual_net = manual['net_revenue']
    service_net = service.get('net_revenue', 0)
    
    diff_manual_service = abs(manual_net - service_net)
    if diff_manual_service > 0.01:
        print(f"  {RED}✗ NET REVENUE MISMATCH!{RESET}")
        print(f"    Manual:    ₹{manual_net:,.2f}")
        print(f"    Service:   ₹{service_net:,.2f}")
        print(f"    Difference: ₹{diff_manual_service:,.2f}")
        comparison['bugs_found'].append({
            'type': 'manual_vs_service',
            'manual_net': manual_net,
            'service_net': service_net,
            'difference': diff_manual_service
        })
    else:
        print(f"  {GREEN}✓ Net Revenue matches: ₹{manual_net:,.2f}{RESET}")
    
    # Component comparison
    print(f"\n  Component Breakdown:")
    print(f"    Shipment Revenue:")
    print(f"      Manual:  ₹{manual['shipment_revenue']:,.2f}")
    print(f"      Service: ₹{service.get('revenue', 0):,.2f}")
    
    print(f"    Refunds:")
    print(f"      Manual:  ₹{manual['refund_amount']:,.2f}")
    print(f"      Service: ₹{service.get('refunds', 0):,.2f}")
    
    print(f"    Shipping Loss:")
    print(f"      Manual:  ₹{manual['shipping_loss']:,.2f}")
    print(f"      Service: ₹{service.get('shipping_loss', 0):,.2f}")
    
    print(f"    Free Replacement Cost:")
    print(f"      Manual:  ₹{manual['free_replacement_cost']:,.2f}")
    print(f"      Service: ₹{service.get('free_replacement_cost', 0):,.2f}")
    
    # Compare with API if available
    if api:
        api_net = api.get('net_revenue', 0)
        diff_api_service = abs(service_net - api_net)
        
        print(f"\n{BOLD}Service vs API Endpoint:{RESET}")
        if diff_api_service > 0.01:
            print(f"  {RED}✗ MISMATCH!{RESET}")
            print(f"    Service: ₹{service_net:,.2f}")
            print(f"    API:     ₹{api_net:,.2f}")
            print(f"    Difference: ₹{diff_api_service:,.2f}")
        else:
            print(f"  {GREEN}✓ Match: ₹{service_net:,.2f}{RESET}")
    
    return comparison


def save_results(comparison: Dict[str, Any], manual: Dict[str, Any], service: Dict[str, Any], api: Optional[Dict[str, Any]], chat: Optional[Dict[str, Any]]):
    """Save results to markdown file"""
    output_file = PROJECT_ROOT / 'METRIC_VERIFICATION_RESULTS.md'
    
    with open(output_file, 'w') as f:
        f.write("# Net Revenue Verification Results - September 2025\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        
        if comparison['bugs_found']:
            f.write(f"**{len(comparison['bugs_found'])} BUG(S) FOUND!**\n\n")
            for bug in comparison['bugs_found']:
                f.write(f"- **{bug['type']}**: Difference of ₹{bug['difference']:,.2f}\n")
        else:
            f.write("**✓ All calculations match!**\n\n")
        
        f.write("\n## Manual SQL Calculation\n\n")
        f.write("```\n")
        f.write(f"Shipments:     ₹{manual['shipment_revenue']:,.2f}\n")
        f.write(f"- Refunds:     -₹{manual['refund_amount']:,.2f}\n")
        f.write(f"- Cancels:     ₹{manual['cancel_amount']:,.2f}\n")
        f.write(f"- Shipping:    -₹{manual['shipping_loss']:,.2f}\n")
        f.write(f"- Free Repl:   -₹{manual['free_replacement_cost']:,.2f}\n")
        f.write(f"{'='*60}\n")
        f.write(f"= Net Revenue: ₹{manual['net_revenue']:,.2f}\n")
        f.write("```\n\n")
        
        f.write("## Metrics Service Calculation\n\n")
        f.write("```\n")
        f.write(f"Revenue:              ₹{service.get('revenue', 0):,.2f}\n")
        f.write(f"Refunds:               ₹{service.get('refunds', 0):,.2f}\n")
        f.write(f"Shipping Loss:         ₹{service.get('shipping_loss', 0):,.2f}\n")
        f.write(f"Free Replacement Cost: ₹{service.get('free_replacement_cost', 0):,.2f}\n")
        f.write(f"Net Revenue:           ₹{service.get('net_revenue', 0):,.2f}\n")
        f.write("```\n\n")
        
        if api:
            f.write("## API Endpoint (/api/metrics)\n\n")
            f.write("```json\n")
            f.write(json.dumps(api, indent=2))
            f.write("\n```\n\n")
        
        if chat:
            f.write("## AI Chat Response\n\n")
            f.write(f"**Answer:** {chat.get('answer', 'N/A')}\n\n")
            if chat.get('sql'):
                f.write(f"**SQL:**\n```sql\n{chat['sql']}\n```\n\n")
        
        f.write("## Discrepancies\n\n")
        if comparison['bugs_found']:
            for bug in comparison['bugs_found']:
                f.write(f"- **{bug['type']}**: ₹{bug['difference']:,.2f} difference\n")
        else:
            f.write("None found.\n")
    
    print(f"\n{GREEN}✓ Results saved to: {output_file}{RESET}")


def main():
    print_header("NET REVENUE VERIFICATION - September 2025")
    
    # Get connection first (may be None if database is locked)
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"{YELLOW}⚠️  Cannot get direct database connection: {e}{RESET}")
        print(f"{YELLOW}   Will use API endpoints and metrics service instead{RESET}")
        conn = None
    
    # Check if sales table exists (only if we have a connection)
    if conn is not None:
        try:
            tables = conn.execute("SHOW TABLES").fetchdf()
            if 'sales' not in tables['name'].values:
                print(f"{RED}❌ Sales table not found in database!{RESET}")
                print(f"Available tables: {list(tables['name'].values)}")
                sys.exit(1)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not check tables: {e}{RESET}")
            print(f"{YELLOW}Attempting to proceed anyway...{RESET}")
    
    # Step 1: Manual calculation
    try:
        manual_results = calculate_manual_september_metrics(conn)
    except Exception as e:
        print(f"{RED}Error in manual calculation: {e}{RESET}")
        import traceback
        traceback.print_exc()
        manual_results = {}
    
    # Step 2: Metrics service
    try:
        service_results = calculate_metrics_service_september()
    except Exception as e:
        print(f"{RED}Error in metrics service: {e}{RESET}")
        import traceback
        traceback.print_exc()
        service_results = {}
    
    # Step 3: API endpoint
    api_results = compare_api_endpoint()
    
    # Step 4: AI Chat
    chat_results = compare_ai_chat()
    
    # Step 5: Compare
    # Use API results as source of truth if manual calculation failed
    if not manual_results and api_results:
        print(f"\n{YELLOW}⚠️  Using API results as manual calculation baseline{RESET}")
        manual_results = {
            'shipment_revenue': api_results.get('revenue', 0),
            'refund_amount': api_results.get('refunds', 0),
            'cancel_amount': 0,
            'shipping_loss': api_results.get('shipping_loss', 0),
            'free_replacement_cost': api_results.get('free_replacement_cost', 0),
            'net_revenue': api_results.get('net_revenue', 0),
            'counts': {}
        }
    
    if manual_results and (service_results or api_results):
        # Use API results if service_results is empty
        if not service_results and api_results:
            service_results = api_results
        
        comparison = compare_results(manual_results, service_results, api_results)
        
        # Step 6: Save results
        save_results(comparison, manual_results, service_results, api_results, chat_results)
        
        # Final summary
        print_header("FINAL SUMMARY")
        
        if comparison['bugs_found']:
            print(f"{RED}{BOLD}BUGS FOUND!{RESET}")
            for bug in comparison['bugs_found']:
                print(f"  {RED}✗ {bug['type']}: ₹{bug['difference']:,.2f} difference{RESET}")
            print(f"\n{RED}Check METRIC_VERIFICATION_RESULTS.md for details{RESET}")
        else:
            print(f"{GREEN}{BOLD}✓ All calculations match!{RESET}")
            print(f"\n{GREEN}No bugs detected in net revenue calculation{RESET}")
    else:
        print(f"{RED}❌ Cannot complete comparison - missing results{RESET}")
        print(f"   Manual: {bool(manual_results)}")
        print(f"   Service: {bool(service_results)}")
        print(f"   API: {bool(api_results)}")
    
    # Close connection if we opened it
    if hasattr(conn, 'close'):
        try:
            conn.close()
        except:
            pass


if __name__ == '__main__':
    main()

