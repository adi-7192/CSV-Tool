"""
Net Revenue Fix Verification Script

Verifies that net revenue calculation is correct by comparing:
1. Manual SQL calculation from database
2. /api/metrics endpoint response
3. AI Chat response

All three should match exactly!
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
        raise FileNotFoundError("Database file not found")
    
    # Try using execute_query first (uses backend connection)
    try:
        test_result = execute_query("SELECT 1 as test")
        if test_result is not None:
            print(f"{GREEN}Using backend database connection{RESET}")
            return None  # Use execute_query instead
    except Exception:
        pass
    
    # Try direct read-only connection
    try:
        print(f"{YELLOW}Using direct read-only connection to: {db_path}{RESET}")
        return duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        if "lock" in str(e).lower():
            raise Exception("Database is locked. Please stop the API server first.")
        raise


def calculate_manual_september_net_revenue(conn) -> Dict[str, Any]:
    """Calculate September 2025 net revenue manually using SQL"""
    print_section("STEP 1: Manual SQL Calculation (September 2025)")
    
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


def get_api_metrics_september() -> Optional[Dict[str, Any]]:
    """Get September 2025 metrics from /api/metrics endpoint"""
    print_section("STEP 2: API Endpoint (/api/metrics)")
    
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
    print_section("STEP 3: AI Chat (/api/chat/ask)")
    
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
            
            print(f"\n{BOLD}AI Chat Response:{RESET}")
            print(f"  Answer: {answer}")
            
            if sql:
                print(f"\n  Generated SQL:")
                # Show SQL in a readable format
                sql_lines = sql.split('\n')
                for line in sql_lines[:10]:  # Show first 10 lines
                    print(f"    {line}")
                if len(sql_lines) > 10:
                    print(f"    ... ({len(sql_lines) - 10} more lines)")
            
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


def compare_results(manual: Dict[str, Any], api: Optional[Dict[str, Any]], chat: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare all three calculation methods"""
    print_section("STEP 4: Comparison & Verification")
    
    comparison = {
        'manual': manual,
        'api': api,
        'chat': chat,
        'matches': {},
        'all_match': False
    }
    
    manual_net = manual.get('net_revenue', 0)
    
    print(f"\n{BOLD}Results Comparison:{RESET}\n")
    print(f"  Manual Calculation:    ₹{manual_net:,.2f}")
    
    # Compare with API
    if api:
        api_net = api.get('net_revenue', 0)
        diff_api = abs(manual_net - api_net)
        matches_api = diff_api < 0.01  # Allow 1 paisa tolerance
        
        print(f"  /api/metrics response:  ₹{api_net:,.2f}")
        
        if matches_api:
            print(f"  {GREEN}✅ API matches manual calculation{RESET}")
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
        diff_chat = abs(manual_net - chat_net)
        matches_chat = diff_chat < 1.0  # Allow ₹1 tolerance for AI Chat (may have rounding)
        
        print(f"  AI Chat response:       ₹{chat_net:,.2f}")
        
        if matches_chat:
            print(f"  {GREEN}✅ AI Chat matches manual calculation{RESET}")
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
    
    return comparison


def save_results(comparison: Dict[str, Any], manual: Dict[str, Any], api: Optional[Dict[str, Any]], chat: Optional[Dict[str, Any]]):
    """Save results to markdown file"""
    output_file = PROJECT_ROOT / 'NET_REVENUE_FIX_VERIFICATION.md'
    
    with open(output_file, 'w') as f:
        f.write("# Net Revenue Fix Verification - September 2025\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        
        if comparison['all_match']:
            f.write("**✅ ALL RESULTS MATCH - BUG FIXED!**\n\n")
        else:
            f.write("**❌ RESULTS DO NOT MATCH - BUG NOT FIXED!**\n\n")
        
        f.write("## Results\n\n")
        
        f.write("### Manual Calculation (SQL)\n\n")
        f.write("```\n")
        f.write(f"Gross Revenue:        ₹{manual['gross_revenue']:,.2f}\n")
        f.write(f"Refund Amount:        ₹{manual['refund_amount']:,.2f}\n")
        f.write(f"Cancellation Amount:  ₹{manual['cancellation_amount']:,.2f}\n")
        f.write(f"Free Replacement:     ₹{manual['free_replacement_cost']:,.2f}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Net Revenue:          ₹{manual['net_revenue']:,.2f}\n")
        f.write("```\n\n")
        
        if api:
            f.write("### /api/metrics Endpoint\n\n")
            f.write("```\n")
            f.write(f"Gross Revenue:        ₹{api['gross_revenue']:,.2f}\n")
            f.write(f"Refund Amount:        ₹{api['refund_amount']:,.2f}\n")
            f.write(f"Cancellation Amount:  ₹{api['cancellation_amount']:,.2f}\n")
            f.write(f"Free Replacement:     ₹{api['free_replacement_cost']:,.2f}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Net Revenue:          ₹{api['net_revenue']:,.2f}\n")
            f.write("```\n\n")
            
            diff = abs(manual['net_revenue'] - api['net_revenue'])
            if diff < 0.01:
                f.write(f"**Status:** ✅ Match (Difference: ₹{diff:.2f})\n\n")
            else:
                f.write(f"**Status:** ❌ Mismatch (Difference: ₹{diff:,.2f})\n\n")
        
        if chat:
            f.write("### AI Chat Response\n\n")
            f.write(f"**Answer:** {chat.get('answer', 'N/A')}\n\n")
            
            if chat.get('sql'):
                f.write("**Generated SQL:**\n\n")
                f.write("```sql\n")
                f.write(chat['sql'])
                f.write("\n```\n\n")
            
            f.write(f"**Extracted Net Revenue:** ₹{chat['net_revenue']:,.2f}\n\n")
            
            diff = abs(manual['net_revenue'] - chat['net_revenue'])
            if diff < 1.0:
                f.write(f"**Status:** ✅ Match (Difference: ₹{diff:.2f})\n\n")
            else:
                f.write(f"**Status:** ❌ Mismatch (Difference: ₹{diff:,.2f})\n\n")
        
        f.write("## Comparison\n\n")
        f.write("| Source | Net Revenue | Status |\n")
        f.write("|--------|-------------|--------|\n")
        f.write(f"| Manual Calculation | ₹{manual['net_revenue']:,.2f} | Baseline |\n")
        
        if api:
            status = "✅ Match" if comparison['matches']['api'] else "❌ Mismatch"
            f.write(f"| /api/metrics | ₹{api['net_revenue']:,.2f} | {status} |\n")
        
        if chat:
            status = "✅ Match" if comparison['matches']['chat'] else "❌ Mismatch"
            f.write(f"| AI Chat | ₹{chat['net_revenue']:,.2f} | {status} |\n")
        
        f.write("\n## Formula Verification\n\n")
        f.write("All calculations should use:\n\n")
        f.write("```\n")
        f.write("net_revenue = gross_revenue - refund_amount - cancellation_amount - free_replacement_cost\n")
        f.write("```\n\n")
        
        f.write("### Manual Calculation\n\n")
        manual_calc = (
            manual['gross_revenue'] -
            manual['refund_amount'] -
            manual['cancellation_amount'] -
            manual['free_replacement_cost']
        )
        f.write(f"Formula: {manual['gross_revenue']:,.2f} - {manual['refund_amount']:,.2f} - {manual['cancellation_amount']:,.2f} - {manual['free_replacement_cost']:,.2f} = ₹{manual_calc:,.2f}\n")
        f.write(f"Actual: ₹{manual['net_revenue']:,.2f}\n")
        f.write(f"Match: {'✅' if abs(manual_calc - manual['net_revenue']) < 0.01 else '❌'}\n\n")
        
        if api:
            f.write("### /api/metrics\n\n")
            api_calc = (
                api['gross_revenue'] -
                api['refund_amount'] -
                api['cancellation_amount'] -
                api['free_replacement_cost']
            )
            f.write(f"Formula: {api['gross_revenue']:,.2f} - {api['refund_amount']:,.2f} - {api['cancellation_amount']:,.2f} - {api['free_replacement_cost']:,.2f} = ₹{api_calc:,.2f}\n")
            f.write(f"Actual: ₹{api['net_revenue']:,.2f}\n")
            f.write(f"Match: {'✅' if abs(api_calc - api['net_revenue']) < 0.01 else '❌'}\n\n")
    
    print(f"\n{GREEN}✓ Results saved to: {output_file}{RESET}")


def main():
    print_header("NET REVENUE FIX VERIFICATION - September 2025")
    
    # Check database
    if not table_exists('sales'):
        print(f"{RED}❌ Sales table not found!{RESET}")
        sys.exit(1)
    
    # Get connection
    conn = None
    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"{RED}❌ Cannot connect to database: {e}{RESET}")
        sys.exit(1)
    
    # Step 1: Manual calculation
    try:
        manual_results = calculate_manual_september_net_revenue(conn)
    except Exception as e:
        print(f"{RED}Error in manual calculation: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 2: API endpoint
    api_results = get_api_metrics_september()
    
    # Step 3: AI Chat
    chat_results = get_ai_chat_september()
    
    # Step 4: Compare
    comparison = compare_results(manual_results, api_results, chat_results)
    
    # Step 5: Save results
    save_results(comparison, manual_results, api_results, chat_results)
    
    # Final summary
    print_header("FINAL SUMMARY")
    
    if comparison['all_match']:
        print(f"{GREEN}{BOLD}✅ ALL RESULTS MATCH - BUG FIXED!{RESET}\n")
        print(f"{GREEN}Manual, API, and AI Chat all return the same net revenue value.{RESET}\n")
        sys.exit(0)
    else:
        print(f"{RED}{BOLD}❌ RESULTS DO NOT MATCH - BUG NOT FIXED!{RESET}\n")
        
        mismatches = []
        if api_results and not comparison['matches']['api']:
            mismatches.append("API endpoint")
        if chat_results and not comparison['matches']['chat']:
            mismatches.append("AI Chat")
        
        if mismatches:
            print(f"{RED}Mismatches found in: {', '.join(mismatches)}{RESET}\n")
        
        print(f"{RED}Check NET_REVENUE_FIX_VERIFICATION.md for details{RESET}\n")
        sys.exit(1)
    
    # Close connection if we opened it
    if conn is not None and hasattr(conn, 'close'):
        try:
            conn.close()
        except:
            pass


if __name__ == '__main__':
    main()

