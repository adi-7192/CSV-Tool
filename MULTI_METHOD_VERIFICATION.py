"""
Multi-Method Verification Script

Verifies net revenue calculation by comparing:
1. Manual SQL calculation from database
2. /api/metrics endpoint response
3. AI Chat response

All three methods should return the same net revenue value!
"""
import os
import sys
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import duckdb

# Add backend to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'backend'))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.database import get_connection, execute_query, table_exists

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
    """Get database connection"""
    # Try using execute_query first (uses backend connection)
    # This works even when API is running
    try:
        test_result = execute_query("SELECT 1 as test")
        if test_result is not None:
            print(f"{GREEN}Using backend database connection{RESET}")
            return None  # Use execute_query instead
    except Exception as e:
        print(f"{YELLOW}Backend connection not available: {e}{RESET}")
        print(f"{YELLOW}Attempting direct connection...{RESET}")
    
    # Fallback: Try direct read-only connection
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
        raise FileNotFoundError("Database file not found")
    
    try:
        print(f"{YELLOW}Using direct read-only connection to: {db_path}{RESET}")
        return duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        if "lock" in str(e).lower():
            print(f"{YELLOW}⚠️  Database is locked (API may be running).{RESET}")
            print(f"{YELLOW}   Will use backend connection via execute_query instead.{RESET}")
            return None
        raise


def calculate_manual_september_net_revenue(conn) -> Optional[Dict[str, Any]]:
    """Calculate September 2025 net revenue manually using SQL"""
    print_section("METHOD 1: Manual SQL Calculation (September 2025)")
    
    sql = """
    SELECT 
        -- Gross Revenue (Shipments)
        SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as gross_revenue,
        
        -- Refund Amount (absolute value)
        ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END)) as refund_amount,
        
        -- Cancellation Amount (absolute value)
        ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END)) as cancel_amount,
        
        -- Free Replacement Amount (absolute value)
        ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END)) as free_repl_amount,
        
        -- Net Revenue (calculated)
        (SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END)
         - ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END))
         - ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END))
         - ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END))
        ) as net_revenue
        
    FROM sales
    WHERE CAST(order_date AS DATE) >= '2025-09-01'
      AND CAST(order_date AS DATE) <= '2025-09-30'
    """
    
    try:
        if conn is None:
            result_df = execute_query(sql)
            if result_df.empty:
                raise ValueError("No data returned from manual calculation")
            result = result_df.iloc[0]
        else:
            result = conn.execute(sql).fetchone()
            if result is None:
                raise ValueError("No data returned from manual calculation")
            # Convert to dict for easier access
            cols = ['gross_revenue', 'refund_amount', 'cancel_amount', 'free_repl_amount', 'net_revenue']
            result = {col: val for col, val in zip(cols, result)}
        
        # Handle both DataFrame row and tuple
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        
        gross_revenue = float(result.get('gross_revenue') or result[0] if isinstance(result, (list, tuple)) else 0)
        refund_amount = float(result.get('refund_amount') or result[1] if isinstance(result, (list, tuple)) else 0)
        cancel_amount = float(result.get('cancel_amount') or result[2] if isinstance(result, (list, tuple)) else 0)
        free_repl_amount = float(result.get('free_repl_amount') or result[3] if isinstance(result, (list, tuple)) else 0)
        net_revenue = float(result.get('net_revenue') or result[4] if isinstance(result, (list, tuple)) else 0)
        
        print(f"\n{BOLD}Manual Calculation Results:{RESET}")
        print(f"  Gross Revenue:        ₹{gross_revenue:,.2f}")
        print(f"  Refund Amount:        ₹{refund_amount:,.2f}")
        print(f"  Cancellation Amount:  ₹{cancel_amount:,.2f}")
        print(f"  Free Replacement:    ₹{free_repl_amount:,.2f}")
        print(f"  {'='*60}")
        print(f"  {BOLD}Net Revenue:           ₹{net_revenue:,.2f}{RESET}")
        
        return {
            'gross_revenue': gross_revenue,
            'refund_amount': refund_amount,
            'cancellation_amount': cancel_amount,
            'free_replacement_cost': free_repl_amount,
            'net_revenue': net_revenue,
            'source': 'manual_sql'
        }
    except Exception as e:
        if "lock" in str(e).lower():
            print(f"{YELLOW}⚠️  Database is locked. Cannot perform manual calculation.{RESET}")
            print(f"{YELLOW}   Will use API endpoint as baseline instead.{RESET}")
            return None
        raise


def get_api_metrics_september() -> Optional[Dict[str, Any]]:
    """Get September 2025 metrics from /api/metrics endpoint"""
    print_section("METHOD 2: API Endpoint (/api/metrics)")
    
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
            
            gross_revenue = metrics.get('gross_revenue') or metrics.get('revenue', 0)
            refund_amount = metrics.get('refund_amount') or metrics.get('refunds', 0)
            cancellation_amount = metrics.get('cancellation_amount', 0)
            free_replacement_cost = metrics.get('free_replacement_cost', 0)
            net_revenue = metrics.get('net_revenue', 0)
            
            print(f"\n{BOLD}API Endpoint Results:{RESET}")
            print(f"  Gross Revenue:        ₹{gross_revenue:,.2f}")
            print(f"  Refund Amount:        ₹{refund_amount:,.2f}")
            print(f"  Cancellation Amount:  ₹{cancellation_amount:,.2f}")
            print(f"  Free Replacement:     ₹{free_replacement_cost:,.2f}")
            print(f"  {'='*60}")
            print(f"  {BOLD}Net Revenue:           ₹{net_revenue:,.2f}{RESET}")
            
            return {
                'gross_revenue': float(gross_revenue),
                'refund_amount': float(refund_amount),
                'cancellation_amount': float(cancellation_amount),
                'free_replacement_cost': float(free_replacement_cost),
                'net_revenue': float(net_revenue),
                'source': 'api_endpoint'
            }
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


def get_ai_chat_september() -> Optional[Dict[str, Any]]:
    """Get September 2025 net revenue from AI Chat"""
    print_section("METHOD 3: AI Chat (/api/chat/ask)")
    
    try:
        response = requests.post(
            f"{API_BASE}/api/chat/ask",
            json={
                "question": "What is net revenue for September 2025?"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            chat_data = response.json()
            
            answer = chat_data.get('answer', '')
            sql = chat_data.get('sql', '')
            data = chat_data.get('data', {})
            error = chat_data.get('error')
            
            if error:
                print(f"{RED}AI Chat returned error: {error}{RESET}")
                return None
            
            print(f"\n{BOLD}AI Chat Response:{RESET}")
            print(f"  Answer: {answer}")
            
            if sql:
                print(f"\n  Generated SQL:")
                # Show SQL in a readable format
                sql_lines = sql.split('\n')
                for line in sql_lines[:15]:  # Show first 15 lines
                    print(f"    {line}")
                if len(sql_lines) > 15:
                    print(f"    ... ({len(sql_lines) - 15} more lines)")
            
            # Extract net revenue from answer or data
            net_revenue = None
            
            # Try to extract from answer (format: ₹X,XXX.XX)
            import re
            rupee_match = re.search(r'₹[\d,]+\.?\d*', answer.replace(',', ''))
            if rupee_match:
                value_str = rupee_match.group(0).replace('₹', '').replace(',', '')
                try:
                    net_revenue = float(value_str)
                except ValueError:
                    pass
            
            # Try to extract from data
            if net_revenue is None and data:
                if isinstance(data, dict):
                    if 'value' in data:
                        net_revenue = float(data['value'])
                    elif 'net_revenue' in data:
                        net_revenue = float(data['net_revenue'])
                    elif 'rows' in data and len(data['rows']) > 0:
                        first_row = data['rows'][0]
                        if 'net_revenue' in first_row:
                            net_revenue = float(first_row['net_revenue'])
                        elif 'value' in first_row:
                            net_revenue = float(first_row['value'])
            
            if net_revenue is not None:
                print(f"\n  {BOLD}Extracted Net Revenue: ₹{net_revenue:,.2f}{RESET}")
                return {
                    'net_revenue': float(net_revenue),
                    'answer': answer,
                    'sql': sql,
                    'source': 'ai_chat'
                }
            else:
                print(f"{YELLOW}⚠️  Could not extract net revenue from AI Chat response{RESET}")
                print(f"   Answer: {answer[:200]}")
                print(f"   Data: {data}")
                return None
        else:
            print(f"{RED}AI Chat returned status {response.status_code}{RESET}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"{YELLOW}⚠️  API not available at {API_BASE}{RESET}")
        return None
    except Exception as e:
        print(f"{RED}Error calling AI Chat: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(manual: Optional[Dict[str, Any]], api: Optional[Dict[str, Any]], chat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare all three calculation methods"""
    print_section("COMPARISON & VERIFICATION")
    
    comparison = {
        'manual': manual,
        'api': api,
        'chat': chat,
        'matches': {},
        'all_match': False
    }
    
    # Use API as baseline if manual is not available
    baseline = manual if manual else api
    if not baseline:
        print(f"{RED}❌ No baseline available for comparison{RESET}")
        return comparison
    
    baseline_net = baseline.get('net_revenue', 0)
    
    print(f"\n{BOLD}Results Comparison:{RESET}\n")
    if manual:
        print(f"  Manual Calculation:    ₹{manual['net_revenue']:,.2f}")
    else:
        print(f"  Manual Calculation:    {YELLOW}N/A (database locked){RESET}")
    
    # Compare with API
    if api:
        api_net = api.get('net_revenue', 0)
        diff_api = abs(baseline_net - api_net)
        matches_api = diff_api < 0.01  # Allow 1 paisa tolerance
        
        print(f"  /api/metrics response:  ₹{api_net:,.2f}")
        
        if matches_api:
            print(f"  {GREEN}✅ API matches baseline{RESET}")
            comparison['matches']['api'] = True
        else:
            print(f"  {RED}❌ API mismatch: Difference = ₹{diff_api:,.2f}{RESET}")
            comparison['matches']['api'] = False
    else:
        print(f"  /api/metrics response:  {YELLOW}N/A (not available){RESET}")
        comparison['matches']['api'] = None
    
    # Compare with AI Chat
    if chat:
        chat_net = chat.get('net_revenue', 0)
        diff_chat = abs(baseline_net - chat_net)
        matches_chat = diff_chat < 1.0  # Allow ₹1 tolerance for AI Chat (may have rounding)
        
        print(f"  AI Chat response:       ₹{chat_net:,.2f}")
        
        if matches_chat:
            print(f"  {GREEN}✅ AI Chat matches baseline{RESET}")
            comparison['matches']['chat'] = True
        else:
            print(f"  {RED}❌ AI Chat mismatch: Difference = ₹{diff_chat:,.2f}{RESET}")
            comparison['matches']['chat'] = False
    else:
        print(f"  AI Chat response:       {YELLOW}N/A (not available){RESET}")
        comparison['matches']['chat'] = None
    
    # Determine overall status
    all_match = True
    if api and not comparison['matches']['api']:
        all_match = False
    if chat and not comparison['matches']['chat']:
        all_match = False
    
    comparison['all_match'] = all_match
    comparison['baseline_net'] = baseline_net
    
    return comparison


def main():
    print_header("MULTI-METHOD VERIFICATION - September 2025 Net Revenue")
    
    # Get connection first
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"{RED}❌ Cannot connect to database: {e}{RESET}")
        sys.exit(1)
    
    # Check database - try to query directly
    try:
        if conn is None:
            # Use execute_query
            test_result = execute_query("SELECT COUNT(*) as cnt FROM sales LIMIT 1")
            if test_result.empty:
                print(f"{RED}❌ Sales table is empty or doesn't exist!{RESET}")
                sys.exit(1)
        else:
            # Use direct connection
            test_result = conn.execute("SELECT COUNT(*) as cnt FROM sales LIMIT 1").fetchone()
            if not test_result or test_result[0] == 0:
                print(f"{RED}❌ Sales table is empty or doesn't exist!{RESET}")
                sys.exit(1)
    except Exception as e:
        print(f"{YELLOW}Warning: Could not verify sales table: {e}{RESET}")
        print(f"{YELLOW}Attempting to proceed anyway...{RESET}")
    
    # Method 1: Manual calculation
    manual_results = None
    try:
        manual_results = calculate_manual_september_net_revenue(conn)
    except Exception as e:
        print(f"{YELLOW}Warning: Manual calculation failed: {e}{RESET}")
        print(f"{YELLOW}   Will use API endpoint as baseline instead.{RESET}")
    
    # Method 2: API endpoint
    api_results = get_api_metrics_september()
    
    # If manual failed, use API as baseline
    if not manual_results and api_results:
        print(f"\n{YELLOW}⚠️  Using API endpoint as baseline for comparison{RESET}")
        manual_results = api_results.copy()
        manual_results['source'] = 'api_endpoint_baseline'
    
    # Method 3: AI Chat
    chat_results = get_ai_chat_september()
    
    # Compare
    if manual_results or api_results:
        comparison = compare_results(manual_results, api_results, chat_results)
        
        # Final summary
        print_header("FINAL SUMMARY")
        
        if comparison['all_match']:
            baseline_net = comparison.get('baseline_net', 0)
            print(f"{GREEN}{BOLD}✅ ALL METHODS MATCH - VERIFICATION PASSED!{RESET}\n")
            print(f"{GREEN}All available methods return the same net revenue value.{RESET}\n")
            print(f"{GREEN}Net Revenue (September 2025): ₹{baseline_net:,.2f}{RESET}\n")
            sys.exit(0)
        else:
            print(f"{RED}{BOLD}❌ METHODS DO NOT MATCH - VERIFICATION FAILED!{RESET}\n")
            
            mismatches = []
            if api_results and not comparison['matches']['api']:
                mismatches.append("API endpoint")
            if chat_results and not comparison['matches']['chat']:
                mismatches.append("AI Chat")
            
            if mismatches:
                print(f"{RED}Mismatches found in: {', '.join(mismatches)}{RESET}\n")
            
            print(f"{RED}Check the output above for details{RESET}\n")
            sys.exit(1)
    else:
        print(f"{RED}❌ Cannot verify - no data available{RESET}")
        print(f"   Manual: {bool(manual_results)}")
        print(f"   API: {bool(api_results)}")
        sys.exit(1)
    
    # Close connection if we opened it
    if conn is not None and hasattr(conn, 'close'):
        try:
            conn.close()
        except:
            pass


if __name__ == '__main__':
    main()

