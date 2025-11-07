"""
Debug script to investigate Order ID column detection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.database import execute_query
from services.data_service import detect_order_id_column
import pandas as pd

print("\n" + "="*80)
print("ORDER ID COLUMN DETECTION DEBUG")
print("="*80 + "\n")

# Get all columns from sales table
print("1. Fetching column information from 'sales' table...")
column_info = execute_query("DESCRIBE sales")
print(f"   Found {len(column_info)} columns\n")

# Display all columns
print("2. All columns in 'sales' table:")
for idx, row in column_info.iterrows():
    col_name = row['column_name']
    col_type = row['column_type']
    print(f"   [{idx+1}] {col_name:30s} ({col_type})")

# Check for specific columns
print("\n3. Checking for specific columns:")
available_cols = list(column_info['column_name'].values)

if 'Order Id' in available_cols:
    print("   ✅ 'Order Id' EXISTS")
else:
    print("   ❌ 'Order Id' NOT FOUND")

if 'Order ID' in available_cols:
    print("   ✅ 'Order ID' EXISTS")
else:
    print("   ❌ 'Order ID' NOT FOUND")

if 'order_id' in available_cols:
    print("   ✅ 'order_id' EXISTS")
else:
    print("   ❌ 'order_id' NOT FOUND")

if 'Invoice Number' in available_cols:
    print("   ⚠️  'Invoice Number' EXISTS")
else:
    print("   ✅ 'Invoice Number' NOT FOUND")

# Sample data from Order Id and Invoice Number if they exist
print("\n4. Sampling data from columns:")
if 'Order Id' in available_cols:
    try:
        sample_sql = 'SELECT "Order Id" FROM sales LIMIT 5'
        sample_df = execute_query(sample_sql)
        if not sample_df.empty:
            print(f"   'Order Id' sample values:")
            for val in sample_df['Order Id'].head(5).tolist():
                print(f"      - {val}")
        else:
            print("   'Order Id' column exists but has no data")
    except Exception as e:
        print(f"   Error sampling 'Order Id': {e}")

if 'Invoice Number' in available_cols:
    try:
        sample_sql = 'SELECT "Invoice Number" FROM sales LIMIT 5'
        sample_df = execute_query(sample_sql)
        if not sample_df.empty:
            print(f"   'Invoice Number' sample values:")
            for val in sample_df['Invoice Number'].head(5).tolist():
                print(f"      - {val}")
        else:
            print("   'Invoice Number' column exists but has no data")
    except Exception as e:
        print(f"   Error sampling 'Invoice Number': {e}")

# Test detection function
print("\n5. Testing detect_order_id_column() function:")
selected_col = detect_order_id_column(column_info)
print(f"   Selected column: '{selected_col}'")

# Verify what data is returned
if selected_col:
    print(f"\n6. Verifying data from selected column '{selected_col}':")
    try:
        verify_sql = f'SELECT "{selected_col}" as order_id FROM sales LIMIT 5'
        verify_df = execute_query(verify_sql)
        if not verify_df.empty:
            print(f"   Sample values from '{selected_col}':")
            for val in verify_df['order_id'].head(5).tolist():
                print(f"      - {val}")
        else:
            print(f"   Column '{selected_col}' exists but has no data")
    except Exception as e:
        print(f"   Error verifying '{selected_col}': {e}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80 + "\n")

