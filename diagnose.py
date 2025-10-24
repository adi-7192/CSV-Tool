#!/usr/bin/env python3
"""
Diagnostic tool to check the entire data pipeline
"""
import sqlite3
import os
import pandas as pd

DB_FILE = "data.db"
TABLE_NAME = "sales"

def check_database():
    """Check if database exists and has data"""
    print("\n" + "="*70)
    print("🔍 STEP 1: CHECKING DATABASE")
    print("="*70)
    
    if not os.path.exists(DB_FILE):
        print(f"❌ Database file '{DB_FILE}' does NOT exist!")
        print("   You need to upload and store CSV data first.")
        return False
    
    print(f"✅ Database file '{DB_FILE}' exists")
    print(f"   Size: {os.path.getsize(DB_FILE)} bytes")
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
        if not cursor.fetchone():
            print(f"❌ Table '{TABLE_NAME}' does NOT exist in database!")
            conn.close()
            return False
        
        print(f"✅ Table '{TABLE_NAME}' exists")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        row_count = cursor.fetchone()[0]
        print(f"   Total rows: {row_count}")
        
        if row_count == 0:
            print("❌ Table is EMPTY! No data stored.")
            conn.close()
            return False
        
        # Check columns
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"\n📋 Table columns ({len(columns)} total):")
        for col in columns:
            print(f"   • {col}")
        
        # Check for derived fields
        derived_fields = ['revenue_calc', 'shipping_loss_calc', 'units_sold_calc', 'needs_estimation']
        missing_derived = [f for f in derived_fields if f not in columns]
        if missing_derived:
            print(f"\n⚠️  Missing derived fields: {missing_derived}")
            print("   Database schema needs updating!")
        else:
            print(f"\n✅ All derived fields present")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        return False


def check_data_content():
    """Check the actual data in the database"""
    print("\n" + "="*70)
    print("🔍 STEP 2: CHECKING DATA CONTENT")
    print("="*70)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get date range
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL")
        date_range = cursor.fetchone()
        print(f"\n📅 Date range in database:")
        print(f"   Min: {date_range[0]}")
        print(f"   Max: {date_range[1]}")
        
        # Check transaction types
        cursor.execute(f"SELECT DISTINCT transaction_type FROM {TABLE_NAME}")
        txn_types = [row[0] for row in cursor.fetchall()]
        print(f"\n🔄 Transaction types found:")
        for txn in txn_types:
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE transaction_type = ?", (txn,))
            count = cursor.fetchone()[0]
            print(f"   • {txn}: {count} rows")
        
        # Check derived fields
        print(f"\n📊 Derived fields summary:")
        cursor.execute(f"SELECT SUM(revenue_calc), SUM(shipping_loss_calc), SUM(units_sold_calc) FROM {TABLE_NAME}")
        totals = cursor.fetchone()
        print(f"   • Total revenue_calc: ₹{totals[0]:,.2f}" if totals[0] else "   • revenue_calc: NULL or 0")
        print(f"   • Total shipping_loss_calc: ₹{totals[1]:,.2f}" if totals[1] else "   • shipping_loss_calc: NULL or 0")
        print(f"   • Total units_sold_calc: {totals[2]}" if totals[2] else "   • units_sold_calc: NULL or 0")
        
        # Sample data
        print(f"\n📋 First 5 rows:")
        df = pd.read_sql_query(f"SELECT order_date, transaction_type, revenue_in_inr, revenue_calc, shipping_loss_calc, units_sold_calc FROM {TABLE_NAME} LIMIT 5", conn)
        print(df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking data content: {e}")
        import traceback
        traceback.print_exc()


def check_date_filter():
    """Test date filtering"""
    print("\n" + "="*70)
    print("🔍 STEP 3: TESTING DATE FILTER")
    print("="*70)
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get date range
        cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME}")
        date_range = cursor.fetchone()
        min_date, max_date = date_range
        
        print(f"\n🧪 Test 1: Query with full date range")
        print(f"   Dates: {min_date} to {max_date}")
        
        query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE order_date >= ? AND order_date <= ?"
        cursor.execute(query, (min_date, max_date))
        count = cursor.fetchone()[0]
        print(f"   ✅ Result: {count} rows")
        
        print(f"\n🧪 Test 2: Query each transaction type")
        for txn in ['Shipment', 'Refund', 'Cancel', 'FreeReplacement']:
            query = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE transaction_type = ? AND order_date >= ? AND order_date <= ?"
            cursor.execute(query, (txn, min_date, max_date))
            count = cursor.fetchone()[0]
            print(f"   {txn}: {count} rows")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error testing date filter: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "="*70)
    print("🔬 DATABASE DIAGNOSTIC TOOL")
    print("="*70)
    
    if not check_database():
        print("\n❌ Database check failed. Cannot continue.")
        return
    
    check_data_content()
    check_date_filter()
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nIf you see zero values in derived fields:")
    print("1. Delete data.db")
    print("2. Upload CSV again")
    print("3. Click 'Clean & Validate Data'")
    print("4. Click 'Store Data'")
    print("5. Check console output for cleaning debug messages")


if __name__ == "__main__":
    main()

