#!/usr/bin/env python3
"""
Debug script to diagnose why trend calculation is returning 0
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

from core.database import execute_query, table_exists
from datetime import datetime, timedelta

def debug_trend_calculation():
    """Debug trend calculation step by step"""
    
    if not table_exists('sales'):
        print("❌ Sales table does not exist")
        return
    
    print("=" * 80)
    print("TREND CALCULATION DEBUG")
    print("=" * 80)
    
    # Step 1: Check date range in database
    print("\n1. Checking date range in database...")
    date_range = execute_query("SELECT MIN(order_date) as earliest, MAX(order_date) as latest FROM sales WHERE order_date IS NOT NULL")
    if not date_range.empty:
        earliest = str(date_range['earliest'].iloc[0])
        latest = str(date_range['latest'].iloc[0])
        print(f"   ✅ Database date range: {earliest} to {latest}")
    else:
        print("   ❌ No date data found")
        return
    
    # Step 2: Check current period (July-Sept 2025)
    print("\n2. Checking current period (2025-07-01 to 2025-09-30)...")
    current_period = execute_query("""
        SELECT 
            COUNT(DISTINCT sku) as unique_skus,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as revenue
        FROM sales
        WHERE order_date >= '2025-07-01' AND order_date <= '2025-09-30'
    """)
    if not current_period.empty:
        print(f"   ✅ Current period: {current_period['unique_skus'].iloc[0]} SKUs, {current_period['total_transactions'].iloc[0]} transactions")
        print(f"   ✅ Current period revenue: ₹{current_period['revenue'].iloc[0]:,.2f}")
    else:
        print("   ❌ No current period data")
    
    # Step 3: Calculate previous period dates
    print("\n3. Calculating previous period dates...")
    current_start = datetime.strptime('2025-07-01', '%Y-%m-%d')
    current_end = datetime.strptime('2025-09-30', '%Y-%m-%d')
    period_length = (current_end - current_start).days + 1
    prev_end = current_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=period_length - 1)
    
    print(f"   Current period: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')} ({period_length} days)")
    print(f"   Previous period: {prev_start.strftime('%Y-%m-%d')} to {prev_end.strftime('%Y-%m-%d')} ({period_length} days)")
    
    # Step 4: Check previous period data
    print("\n4. Checking previous period data...")
    prev_period = execute_query(f"""
        SELECT 
            COUNT(DISTINCT sku) as unique_skus,
            COUNT(*) as total_transactions,
            SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as revenue
        FROM sales
        WHERE order_date >= '{prev_start.strftime('%Y-%m-%d')}' AND order_date <= '{prev_end.strftime('%Y-%m-%d')}'
    """)
    if not prev_period.empty:
        prev_count = prev_period['unique_skus'].iloc[0]
        prev_revenue = prev_period['revenue'].iloc[0]
        print(f"   ✅ Previous period: {prev_count} SKUs, {prev_period['total_transactions'].iloc[0]} transactions")
        print(f"   ✅ Previous period revenue: ₹{prev_revenue:,.2f}")
        
        if prev_count == 0:
            print("\n   ⚠️  WARNING: No SKUs found in previous period!")
            print("   This means trend calculation will return 0 for all SKUs.")
            print("   Reason: Cannot calculate growth without previous period data.")
    else:
        print("   ❌ No previous period data found")
        print("   ⚠️  This explains why trend is 0!")
    
    # Step 5: Get sample SKUs from current period
    print("\n5. Checking sample SKUs from current period...")
    current_skus = execute_query("""
        SELECT 
            sku,
            SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as revenue
        FROM sales
        WHERE order_date >= '2025-07-01' AND order_date <= '2025-09-30'
        GROUP BY sku
        ORDER BY revenue DESC
        LIMIT 5
    """)
    if not current_skus.empty:
        print("   Top 5 SKUs in current period:")
        for idx, row in current_skus.iterrows():
            print(f"   - {row['sku']}: ₹{row['revenue']:,.2f}")
    
    # Step 6: Check if these SKUs exist in previous period
    print("\n6. Checking if these SKUs exist in previous period...")
    if not current_skus.empty:
        sku_list = "', '".join(current_skus['sku'].tolist())
        prev_skus = execute_query(f"""
            SELECT 
                sku,
                SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as prev_revenue
            FROM sales
            WHERE order_date >= '{prev_start.strftime('%Y-%m-%d')}' AND order_date <= '{prev_end.strftime('%Y-%m-%d')}'
              AND sku IN ('{sku_list}')
            GROUP BY sku
        """)
        if not prev_skus.empty:
            print("   ✅ Found matching SKUs in previous period:")
            for idx, row in prev_skus.iterrows():
                print(f"   - {row['sku']}: ₹{row['prev_revenue']:,.2f}")
            
            # Calculate sample trend
            print("\n7. Sample trend calculation:")
            for idx, curr_row in current_skus.head(3).iterrows():
                sku = curr_row['sku']
                curr_rev = curr_row['revenue']
                prev_row = prev_skus[prev_skus['sku'] == sku]
                if not prev_row.empty:
                    prev_rev = prev_row['prev_revenue'].iloc[0]
                    if prev_rev > 0:
                        trend = ((curr_rev - prev_rev) / prev_rev) * 100
                        print(f"   {sku}: ({curr_rev:,.2f} - {prev_rev:,.2f}) / {prev_rev:,.2f} * 100 = {trend:.1f}%")
                    else:
                        print(f"   {sku}: Previous revenue is 0, trend = 0%")
                else:
                    print(f"   {sku}: Not found in previous period, trend = 0%")
        else:
            print("   ❌ None of these SKUs found in previous period!")
            print("   ⚠️  This explains why trend is 0!")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    debug_trend_calculation()


