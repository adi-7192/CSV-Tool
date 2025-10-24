"""
CSV Analytics Dashboard - Simple & Clean

Upload CSV → Auto-map columns → View KPIs → Chat about data
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import hashlib
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Constants
DB_FILE = 'data.db'
TABLE_NAME = 'sales'
STAGING_TABLE = 'sales_staging'
MAPPING_FILE = 'data/mapping.json'
RAW_DATA_FOLDER = 'data/raw'
CLEANED_DATA_FOLDER = 'data/cleaned'

# Database path logging
import os
DB_ABSOLUTE_PATH = os.path.abspath(DB_FILE)
print(f"🗄️ Database path: {DB_ABSOLUTE_PATH}")

# Create data folders
os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
os.makedirs(CLEANED_DATA_FOLDER, exist_ok=True)

# Column synonyms for auto-mapping
SYNONYMS = {
    "order_date": ["order date", "invoice date", "shipment date", "date"],
    "order_id": ["order id", "order number", "invoice id", "order no"],
    "sku": ["sku", "product id", "item id"],
    "asin": ["asin", "amazon asin"],
    "product_name": ["product name", "item description", "title", "product"],
    "quantity": ["quantity", "qty", "units"],
    "revenue_amount": ["invoice amount", "order amount", "total amount", "amount", "revenue"],
    "shipping_amount": ["shipping amount", "shipping cost", "shipping", "freight"],
    "transaction_type": ["transaction type", "type", "transaction", "order type", "transaction"],
    "shipment_item_id": ["shipment item id", "shipment id", "item id", "shipment item"],
    "region": ["region", "city", "market"],
    "status": ["status", "order status"]
}

def normalize_header(header: str) -> str:
    """Normalize header: strip, collapse spaces, lowercase"""
    return re.sub(r'\s+', ' ', header.strip().lower())

def normalize_key_fields(order_id: str, sku: str, order_date: str) -> Tuple[str, str, str]:
    """Normalize key fields for consistent duplicate detection"""
    # Normalize order_id and sku: trim whitespace, unify case
    norm_order_id = str(order_id).strip().upper() if pd.notna(order_id) else ""
    norm_sku = str(sku).strip().upper() if pd.notna(sku) else ""
    
    # Normalize order_date: con›
    if pd.notna(order_date) and str(order_date).strip():
        try:
            # Parse date and convert to YYYY-MM-DD format
            parsed_date = pd.to_datetime(order_date, errors='coerce')
            if pd.notna(parsed_date):
                norm_order_date = parsed_date.strftime('%Y-%m-%d')
            else:
                norm_order_date = ""
        except:
            norm_order_date = ""
    else:
        norm_order_date = ""
    
    return norm_order_id, norm_sku, norm_order_date

def clean_dataframe(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict]:
    """Comprehensive data cleaning with validation report"""
    print("🧹 Starting data cleaning process...")
    
    # Initialize validation report
    report = {
        'total_rows_read': len(df),
        'rows_kept': 0,
        'rows_dropped': 0,
        'duplicates_found': 0,
        'columns_with_missing': [],
        'invalid_revenue_rows': 0,
        'problematic_rows': [],
        'cleaning_steps': []
    }
    
    # Step 1: Create a copy for cleaning
    cleaned_df = df.copy()
    report['cleaning_steps'].append("📋 Created working copy of data")
    
    # Step 2: Trim all string columns
    string_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in string_columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    report['cleaning_steps'].append(f"✂️ Trimmed whitespace from {len(string_columns)} string columns")
    
    # Step 3: Normalize casing for key fields
    if mappings.get('order_id'):
        cleaned_df[mappings['order_id']] = cleaned_df[mappings['order_id']].str.upper()
    if mappings.get('sku'):
        cleaned_df[mappings['sku']] = cleaned_df[mappings['sku']].str.upper()
    report['cleaning_steps'].append("🔤 Normalized casing for order_id and sku")
    
    # Step 4: Parse dates to YYYY-MM-DD format
    if mappings.get('order_date'):
        date_col = mappings['order_date']
        try:
            cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
            report['cleaning_steps'].append("📅 Parsed and normalized dates to YYYY-MM-DD format")
        except Exception as e:
            report['cleaning_steps'].append(f"⚠️ Date parsing failed: {str(e)}")
    
    # Step 5: Clean currency fields
    if mappings.get('revenue_amount'):
        revenue_col = mappings['revenue_amount']
        # Strip currency symbols and commas, convert to numeric
        cleaned_df[revenue_col] = cleaned_df[revenue_col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        cleaned_df[revenue_col] = pd.to_numeric(cleaned_df[revenue_col], errors='coerce').fillna(0.0)
        report['cleaning_steps'].append("💰 Cleaned revenue fields (removed ₹, $, commas)")
    
    # Clean shipping amount fields
    if mappings.get('shipping_amount'):
        shipping_col = mappings['shipping_amount']
        cleaned_df[shipping_col] = cleaned_df[shipping_col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        cleaned_df[shipping_col] = pd.to_numeric(cleaned_df[shipping_col], errors='coerce').fillna(0.0)
        report['cleaning_steps'].append("🚚 Cleaned shipping amount fields")
    
    # Normalize transaction types
    if mappings.get('transaction_type'):
        transaction_col = mappings['transaction_type']
        cleaned_df[transaction_col] = cleaned_df[transaction_col].apply(normalize_transaction_type)
        report['cleaning_steps'].append("🔄 Normalized transaction types")
    
    # Step 6: Handle missing quantities with flag
    if mappings.get('quantity'):
        quantity_col = mappings['quantity']
        # Convert to numeric, keeping NaN for missing values
        cleaned_df[quantity_col] = pd.to_numeric(cleaned_df[quantity_col], errors='coerce')
        
        # Add missing quantity flag
        cleaned_df['missing_quantity_flag'] = cleaned_df[quantity_col].isna()
        
        # Fill missing quantities with 0 for calculations, but keep the flag
        missing_quantity_count = cleaned_df['missing_quantity_flag'].sum()
        cleaned_df[quantity_col] = cleaned_df[quantity_col].fillna(0).astype(int)
        
        report['cleaning_steps'].append(f"📦 Handled {missing_quantity_count} missing quantities (flagged, filled with 0)")
    
    # Step 7: Check for missing values in key columns
    key_columns = [mappings.get('order_id'), mappings.get('sku'), mappings.get('order_date')]
    key_columns = [col for col in key_columns if col is not None]
    
    for col in key_columns:
        missing_count = cleaned_df[col].isna().sum()
        if missing_count > 0:
            report['columns_with_missing'].append(f"{col}: {missing_count} missing values")
    
    # Step 8: Drop rows with missing key data
    before_drop = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=key_columns)
    after_drop = len(cleaned_df)
    dropped_missing = before_drop - after_drop
    report['rows_dropped'] += dropped_missing
    if dropped_missing > 0:
        report['cleaning_steps'].append(f"🗑️ Dropped {dropped_missing} rows with missing key data")
    
    # Step 9: Drop completely empty rows
    before_empty = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(how='all')
    after_empty = len(cleaned_df)
    dropped_empty = before_empty - after_empty
    report['rows_dropped'] += dropped_empty
    if dropped_empty > 0:
        report['cleaning_steps'].append(f"🗑️ Dropped {dropped_empty} completely empty rows")
    
    # Step 10: Detect and handle duplicates (consider all key columns)
    key_columns_for_dup = []
    if mappings.get('order_id'):
        key_columns_for_dup.append(mappings['order_id'])
    if mappings.get('sku'):
        key_columns_for_dup.append(mappings['sku'])
    if mappings.get('asin'):
        key_columns_for_dup.append(mappings['asin'])
    if mappings.get('transaction_type'):
        key_columns_for_dup.append(mappings['transaction_type'])
    if mappings.get('revenue_amount'):
        key_columns_for_dup.append(mappings['revenue_amount'])
    if mappings.get('order_date'):
        key_columns_for_dup.append(mappings['order_date'])
    
    if len(key_columns_for_dup) >= 3:  # Need at least 3 key columns
        # Find duplicates based on all key columns
        duplicate_mask = cleaned_df.duplicated(subset=key_columns_for_dup, keep='first')
        duplicates_count = duplicate_mask.sum()
        report['duplicates_found'] = duplicates_count
        
        if duplicates_count > 0:
            # Keep first occurrence, drop duplicates
            cleaned_df = cleaned_df[~duplicate_mask]
            report['cleaning_steps'].append(f"🔄 Removed {duplicates_count} duplicate rows (kept first occurrence)")
            report['cleaning_steps'].append(f"🔑 Duplicate detection used columns: {', '.join(key_columns_for_dup)}")
    
    # Step 11: Preserve negative revenue values (they represent refunds/adjustments)
    if mappings.get('revenue_amount'):
        revenue_col = mappings['revenue_amount']
        negative_revenue_count = (cleaned_df[revenue_col] < 0).sum()
        report['invalid_revenue_rows'] = negative_revenue_count
        if negative_revenue_count > 0:
            # Keep negative revenue values as they represent valid business transactions
            report['cleaning_steps'].append(f"💰 Preserved {negative_revenue_count} rows with negative revenue (refunds/adjustments)")
    
    # Step 12: Identify problematic rows (first 5)
    problematic_indicators = []
    if mappings.get('revenue_amount'):
        revenue_col = mappings['revenue_amount']
        zero_revenue = cleaned_df[cleaned_df[revenue_col] == 0].head(3)
        if not zero_revenue.empty:
            problematic_indicators.append("Zero revenue rows")
    
    if mappings.get('quantity'):
        quantity_col = mappings['quantity']
        zero_quantity = cleaned_df[cleaned_df[quantity_col] == 0].head(3)
        if not zero_quantity.empty:
            problematic_indicators.append("Zero quantity rows")
    
    # Sample problematic rows
    if problematic_indicators:
        sample_problematic = cleaned_df.head(5)
        for idx, row in sample_problematic.iterrows():
            issues = []
            if mappings.get('revenue_amount') and row[mappings['revenue_amount']] == 0:
                issues.append("Zero revenue")
            if mappings.get('quantity') and row[mappings['quantity']] == 0:
                issues.append("Zero quantity")
            
            if issues:
                report['problematic_rows'].append({
                    'row_index': idx,
                    'issues': issues,
                    'sample_data': {
                        'order_id': str(row.get(mappings.get('order_id'), '')),
                        'sku': str(row.get(mappings.get('sku'), '')),
                        'revenue': row.get(mappings.get('revenue_amount'), 0)
                    }
                })
    
    # Final statistics
    report['rows_kept'] = len(cleaned_df)
    report['cleaning_steps'].append(f"✅ Final result: {report['rows_kept']} rows kept")
    
    # Step 13: Transaction type summary
    if mappings.get('transaction_type') and mappings['transaction_type'] in cleaned_df.columns:
        transaction_col = mappings['transaction_type']
        transaction_summary = cleaned_df[transaction_col].value_counts()
        report['cleaning_steps'].append("📊 Transaction type summary:")
        for txn_type, count in transaction_summary.items():
            report['cleaning_steps'].append(f"  • {txn_type}: {count} rows")
        
        # Check for negative revenue in refunds
        if 'Refund' in transaction_summary.index:
            refund_rows = cleaned_df[cleaned_df[transaction_col] == 'Refund']
            if mappings.get('revenue_amount') and mappings['revenue_amount'] in cleaned_df.columns:
                revenue_col = mappings['revenue_amount']
                negative_refunds = (refund_rows[revenue_col] < 0).sum()
                report['cleaning_steps'].append(f"  • Refund rows with negative amounts: {negative_refunds}")
    
    print(f"🧹 Data cleaning completed: {report['rows_kept']} rows kept, {report['rows_dropped']} dropped")
    
    # Print transaction type summary to console
    if mappings.get('transaction_type') and mappings['transaction_type'] in cleaned_df.columns:
        print("📊 Transaction type summary after cleaning:")
        transaction_summary = cleaned_df[mappings['transaction_type']].value_counts()
        for txn_type, count in transaction_summary.items():
            print(f"  • {txn_type}: {count} rows")
    
    return cleaned_df, report

def save_raw_and_cleaned_data(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, filename: str) -> Tuple[str, str]:
    """Save raw and cleaned data to respective folders"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = os.path.splitext(filename)[0]
    
    # Save raw data
    raw_filename = f"{base_name}_raw_{timestamp}.csv"
    raw_path = os.path.join(RAW_DATA_FOLDER, raw_filename)
    raw_df.to_csv(raw_path, index=False)
    
    # Save cleaned data
    cleaned_filename = f"{base_name}_cleaned_{timestamp}.csv"
    cleaned_path = os.path.join(CLEANED_DATA_FOLDER, cleaned_filename)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    return raw_path, cleaned_path

def compute_signature(headers: List[str]) -> str:
    """Compute SHA1 signature of sorted headers"""
    return hashlib.sha1("|".join(sorted(headers)).encode()).hexdigest()

def is_date_column(series: pd.Series) -> bool:
    """Check if column contains dates (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        parsed = pd.to_datetime(sample, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def is_numeric_column(series: pd.Series) -> bool:
    """Check if column contains numbers (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        cleaned = sample.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        parsed = pd.to_numeric(cleaned, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def is_integer_column(series: pd.Series) -> bool:
    """Check if column contains integers (70% threshold)"""
    sample = series.dropna().head(200)
    if len(sample) == 0: return False
    try:
        parsed = pd.to_numeric(sample, errors='coerce')
        return (parsed.notna().sum() / len(sample)) >= 0.7
    except: return False

def auto_map_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Auto-map CSV columns to standard fields"""
    print(f"Auto-mapping {len(df.columns)} columns...")
    
    mappings = {field: None for field in SYNONYMS.keys()}
    used_headers = set()
    
    for field, synonyms in SYNONYMS.items():
        candidates = []
        
        # Find matching headers
        for header in df.columns:
            if header in used_headers: continue
            norm = normalize_header(header)
            for syn in synonyms:
                if syn in norm or norm in syn:
                    candidates.append(header)
                    break
        
        if len(candidates) == 1:
            mappings[field] = candidates[0]
            used_headers.add(candidates[0])
        elif len(candidates) > 1:
            # Use type inference to disambiguate
            best = None
            if field == "order_date":
                best = next((c for c in candidates if is_date_column(df[c])), None)
            elif field == "quantity":
                best = next((c for c in candidates if is_integer_column(df[c])), None)
            elif field == "revenue_amount":
                # Prioritize "invoice amount"
                invoice_candidates = [c for c in candidates if "invoice" in normalize_header(c)]
                if invoice_candidates:
                    best = next((c for c in invoice_candidates if is_numeric_column(df[c])), None)
                if not best:
                    best = next((c for c in candidates if is_numeric_column(df[c])), None)
            
            if best:
                mappings[field] = best
                used_headers.add(best)
    
    print(f"Mapped {sum(1 for v in mappings.values() if v)} fields")
    return mappings

def save_mapping(headers: List[str], mappings: Dict[str, str]) -> None:
    """Save mapping to file"""
    os.makedirs(os.path.dirname(MAPPING_FILE), exist_ok=True)
    signature = compute_signature(headers)
    
    saved = {}
    if os.path.exists(MAPPING_FILE):
        try:
            with open(MAPPING_FILE, 'r') as f:
                saved = json.load(f)
        except: pass
    
    saved[signature] = {"headers": headers, "mappings": mappings}
    
    with open(MAPPING_FILE, 'w') as f:
        json.dump(saved, f, indent=2)

def load_mapping(headers: List[str]) -> Optional[Dict[str, str]]:
    """Load saved mapping"""
    if not os.path.exists(MAPPING_FILE): return None
    signature = compute_signature(headers)
    
    try:
        with open(MAPPING_FILE, 'r') as f:
            saved = json.load(f)
        return saved.get(signature, {}).get("mappings")
    except: return None

def format_inr(amount: float) -> str:
    """Format amount as INR with thousand separators"""
    return f"₹{amount:,.2f}"

def create_product_identifier(sku: str, asin: str) -> str:
    """Create combined SKU/ASIN identifier for product display in format: SKU (ASIN)"""
    sku_clean = str(sku).strip() if pd.notna(sku) and str(sku).strip() != '' else None
    asin_clean = str(asin).strip() if pd.notna(asin) and str(asin).strip() != '' else None
    
    if sku_clean and asin_clean:
        return f"{sku_clean} ({asin_clean})"
    elif sku_clean:
        return sku_clean
    elif asin_clean:
        return f"Unknown ({asin_clean})"
    else:
        return "Unknown Product"

def parse_revenue(series: pd.Series) -> pd.Series:
    """Parse revenue column, stripping currency symbols"""
    cleaned = series.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def normalize_transaction_type(transaction_type: str) -> str:
    """Normalize transaction type to standard format"""
    if pd.isna(transaction_type):
        return None
    
    # Convert to string and clean
    txn_str = str(transaction_type).strip()
    
    # Handle 'nan' string or empty
    if txn_str == '' or txn_str.lower() == 'nan' or txn_str.lower() == 'none':
        return None
    
    normalized = txn_str.lower()
    
    print(f"  DEBUG: Normalizing '{transaction_type}' → '{normalized}'")
    
    if normalized in ['shipment', 'ship']:
        return 'Shipment'
    elif normalized in ['cancel', 'cancelled', 'cancellation']:
        return 'Cancel'
    elif normalized in ['refund', 'refunds']:
        return 'Refund'
    elif normalized in ['freereplacement', 'free replacement', 'free_replacement']:
        return 'FreeReplacement'
    else:
        # Return title case for unknown types
        result = normalized.title()
        print(f"  WARNING: Unknown transaction type '{transaction_type}' → '{result}'")
        return result

def clean_dataframe_transaction_aware(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict]:
    """Transaction-aware data cleaning - preserves all valid transactions"""
    
    print("\n" + "="*80)
    print("🧹 TRANSACTION-AWARE DATA CLEANING - NO LEGITIMATE DATA DROPPED")
    print("="*80)
    print(f"Starting with {len(df)} rows\n")
    
    # Step 1: Copy and trim whitespace
    cleaned_df = df.copy()
    for col in cleaned_df.select_dtypes(include=['object']).columns:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
    print("✂️ Trimmed whitespace from string columns")
    
    # Step 2: Normalize key fields (preserve original for comparison)
    if mappings.get('order_id'):
        cleaned_df[mappings['order_id']] = cleaned_df[mappings['order_id']].str.upper()
    if mappings.get('sku'):
        cleaned_df[mappings['sku']] = cleaned_df[mappings['sku']].str.upper()
    if mappings.get('asin'):
        cleaned_df[mappings['asin']] = cleaned_df[mappings['asin']].str.upper()
    print("🔤 Normalized Order Id, SKU, ASIN to uppercase")
    
    # Step 3: Parse Invoice Date to YYYY-MM-DD
    if mappings.get('order_date'):
        date_col = mappings['order_date']
        cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
        print(f"📅 Parsed {date_col} to YYYY-MM-DD")
    
    # Step 4: Clean Invoice Amount - PRESERVE negatives, KEEP NaN (no silent coercion!)
    rev_col = mappings.get('revenue_amount')
    if rev_col:
        # Strip currency symbols but preserve negatives
        cleaned_df[rev_col] = cleaned_df[rev_col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        cleaned_df[rev_col] = pd.to_numeric(cleaned_df[rev_col], errors='coerce')
        
        nan_count = cleaned_df[rev_col].isna().sum()
        neg_count = (cleaned_df[rev_col] < 0).sum()
        print(f"💰 Cleaned Invoice Amount: {nan_count} NaN values, {neg_count} negative values (preserved)")
    
    # Step 5: Clean Shipping Amount
    ship_col = mappings.get('shipping_amount')
    if ship_col:
        cleaned_df[ship_col] = cleaned_df[ship_col].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        cleaned_df[ship_col] = pd.to_numeric(cleaned_df[ship_col], errors='coerce').fillna(0.0)
        print(f"🚚 Cleaned Shipping Amount")
    
    # Step 6: Normalize Transaction Type to standard cases
    txn_col = mappings.get('transaction_type')
    if txn_col:
        print(f"📊 Transaction column name from mappings: '{txn_col}'")
        print(f"📊 Available columns: {list(cleaned_df.columns)}")
        
        if txn_col in cleaned_df.columns:
            original_values = cleaned_df[txn_col].unique()
            print(f"📊 Original transaction values: {list(original_values)}")
            
            # Apply normalization
            cleaned_df[txn_col] = cleaned_df[txn_col].apply(normalize_transaction_type)
            normalized_values = cleaned_df[txn_col].unique()
            
            print(f"🔄 Normalized Transaction: {list(original_values)} → {list(normalized_values)}")
            
            # Count None values
            none_count = cleaned_df[txn_col].isna().sum()
            if none_count > 0:
                print(f"⚠️ WARNING: {none_count} rows have None transaction_type after normalization!")
        else:
            print(f"❌ ERROR: Transaction column '{txn_col}' not found in DataFrame!")
    else:
        print(f"❌ ERROR: No transaction_type in mappings!")
    
    # Step 7: Clean Quantity
    qty_col = mappings.get('quantity')
    if qty_col:
        cleaned_df[qty_col] = pd.to_numeric(cleaned_df[qty_col], errors='coerce').fillna(0).astype(int)
        print(f"📦 Cleaned Quantity")
    
    # Step 7.5: Ensure transaction_type column exists with standard name
    if txn_col and txn_col in cleaned_df.columns and txn_col != 'transaction_type':
        # Rename to standard name
        cleaned_df['transaction_type'] = cleaned_df[txn_col]
        print(f"✅ Created standard 'transaction_type' column from '{txn_col}'")
    elif 'transaction_type' not in cleaned_df.columns:
        cleaned_df['transaction_type'] = None
        print(f"⚠️ Created empty 'transaction_type' column")
    
    # Step 8: Deduplication - only drop EXACT duplicates (all columns identical)
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(keep='first')
    duplicates_removed = initial_count - len(cleaned_df)
    print(f"🔄 Removed {duplicates_removed} EXACT duplicates (all columns identical)")
    
    # Step 9: Create derived transaction-aware columns
    print(f"\n📊 Creating derived columns based on Transaction logic...")
    
    cleaned_df['revenue_calc'] = 0.0
    cleaned_df['shipping_loss_calc'] = 0.0
    cleaned_df['units_sold_calc'] = 0
    cleaned_df['needs_estimation'] = False
    
    print(f"Transaction column: transaction_type (standard)")
    print(f"Revenue column: {rev_col}")
    print(f"Shipping column: {ship_col}")
    print(f"Quantity column: {qty_col}")
    
    # Use standard column name 'transaction_type'
    if 'transaction_type' in cleaned_df.columns and rev_col:
        print(f"Processing {len(cleaned_df)} rows for derived columns...")
        
        # Check if we have valid transaction types
        valid_txn_count = cleaned_df['transaction_type'].notna().sum()
        print(f"Rows with valid transaction_type: {valid_txn_count} / {len(cleaned_df)}")
        
        for idx, row in cleaned_df.iterrows():
            txn = row['transaction_type']
            amt = row[rev_col]
            ship = row[ship_col] if ship_col else 0.0
            qty = row[qty_col] if qty_col else 0
            
            if txn == 'Shipment':
                if pd.notna(amt):
                    # Shipment: Invoice Amount should be positive
                    cleaned_df.at[idx, 'revenue_calc'] = abs(amt)
                    cleaned_df.at[idx, 'units_sold_calc'] = int(qty)
                else:
                    print(f"⚠️ Warning: Shipment at row {idx} has NaN Invoice Amount")
                    
            elif txn == 'Refund':
                if pd.notna(amt):
                    # Refund: Invoice Amount should be negative (flip if positive)
                    if amt > 0:
                        print(f"⚠️ Warning: Flipping positive Refund amount {amt} to negative at row {idx}")
                        cleaned_df.at[idx, 'revenue_calc'] = -abs(amt)
                    else:
                        cleaned_df.at[idx, 'revenue_calc'] = amt
                    cleaned_df.at[idx, 'shipping_loss_calc'] = abs(ship)
                    
            elif txn == 'Cancel':
                # Cancel: zero revenue
                cleaned_df.at[idx, 'revenue_calc'] = 0.0
                cleaned_df.at[idx, 'units_sold_calc'] = 0
                
            elif txn == 'FreeReplacement':
                # FreeReplacement: mark for estimation, don't drop
                cleaned_df.at[idx, 'revenue_calc'] = 0.0
                cleaned_df.at[idx, 'needs_estimation'] = True
                cleaned_df.at[idx, 'shipping_loss_calc'] = 0.0  # Placeholder
        
        # Summary after processing
        total_revenue_calc = cleaned_df['revenue_calc'].sum()
        positive_revenue = (cleaned_df['revenue_calc'] > 0).sum()
        negative_revenue = (cleaned_df['revenue_calc'] < 0).sum()
        zero_revenue = (cleaned_df['revenue_calc'] == 0).sum()
        
        print(f"\n✅ Derived columns created:")
        print(f"   Total revenue_calc: ₹{total_revenue_calc:,.2f}")
        print(f"   Rows with positive revenue_calc: {positive_revenue}")
        print(f"   Rows with negative revenue_calc: {negative_revenue}")
        print(f"   Rows with zero revenue_calc: {zero_revenue}")
    else:
        print(f"⚠️ WARNING: Could not create derived columns - missing transaction_type or revenue_amount columns!")
    
    # Step 10: Comprehensive DEBUG SUMMARY
    print("\n" + "="*80)
    print("📊 DEBUG SUMMARY AFTER CLEANING")
    print("="*80)
    
    if 'transaction_type' in cleaned_df.columns:
        print("\n1️⃣ ROWS PER TRANSACTION TYPE:")
        txn_counts = cleaned_df['transaction_type'].value_counts(dropna=False)
        for txn, count in txn_counts.items():
            print(f"   {txn}: {count} rows")
        
        print("\n2️⃣ SUM OF INVOICE AMOUNT PER TRANSACTION TYPE:")
        if rev_col:
            for txn in txn_counts.index:
                txn_data = cleaned_df[cleaned_df['transaction_type'] == txn]
                invoice_sum = txn_data[rev_col].sum()
                revenue_calc_sum = txn_data['revenue_calc'].sum()
                print(f"   {txn}:")
                print(f"      Invoice Amount (original): ₹{invoice_sum:,.2f}")
                print(f"      revenue_calc (derived):    ₹{revenue_calc_sum:,.2f}")
        
        print("\n3️⃣ REVENUE_CALC DISTRIBUTION:")
        positive_count = (cleaned_df['revenue_calc'] > 0).sum()
        negative_count = (cleaned_df['revenue_calc'] < 0).sum()
        zero_count = (cleaned_df['revenue_calc'] == 0).sum()
        print(f"   Positive revenue_calc: {positive_count} rows")
        print(f"   Negative revenue_calc: {negative_count} rows")
        print(f"   Zero revenue_calc:     {zero_count} rows")
        
        print("\n4️⃣ SAMPLE ROWS:")
        for txn in ['Shipment', 'Refund', 'FreeReplacement', 'Cancel']:
            sample = cleaned_df[cleaned_df['transaction_type'] == txn]
            if not sample.empty:
                row = sample.iloc[0]
                print(f"\n   {txn} example (row {sample.index[0]}):")
                if rev_col:
                    print(f"      Invoice Amount: {row[rev_col]}")
                print(f"      revenue_calc: {row['revenue_calc']}")
                if ship_col:
                    print(f"      Shipping Amount: {row[ship_col]}")
                print(f"      shipping_loss_calc: {row['shipping_loss_calc']}")
                if qty_col:
                    print(f"      Quantity: {row[qty_col]}")
                print(f"      units_sold_calc: {row['units_sold_calc']}")
    
    # Warning if all derived values are zero
    if cleaned_df['revenue_calc'].abs().sum() == 0:
        print("\n⚠️⚠️⚠️ WARNING: ALL revenue_calc VALUES ARE ZERO! ⚠️⚠️⚠️")
        print("First 5 rows for inspection:")
        display_cols = [rev_col, txn_col, 'revenue_calc'] if rev_col and txn_col else cleaned_df.columns[:5]
        print(cleaned_df[display_cols].head())
    
    print("\n" + "="*80)
    print(f"✅ CLEANING COMPLETE: {len(cleaned_df)} rows kept, {duplicates_removed} exact duplicates removed")
    print("="*80 + "\n")
    
    # Build report for UI
    report = {
        'total_rows_read': len(df),
        'rows_kept': len(cleaned_df),
        'rows_dropped': duplicates_removed,
        'duplicates_found': duplicates_removed,
        'columns_with_missing': [],
        'invalid_revenue_rows': 0,
        'problematic_rows': [],
        'cleaning_steps': [
            "📋 Transaction-aware cleaning (no legitimate data dropped)",
            f"💰 Preserved {(cleaned_df[rev_col] < 0).sum() if rev_col else 0} negative Invoice Amounts",
            f"🔄 Removed {duplicates_removed} exact duplicates only",
            f"✅ Created derived columns with proper transaction logic"
        ]
    }
    
    if 'transaction_type' in cleaned_df.columns:
        report['cleaning_steps'].append("📊 Transaction type breakdown:")
        for txn, count in cleaned_df['transaction_type'].value_counts(dropna=False).items():
            report['cleaning_steps'].append(f"  • {txn}: {count} rows")
    
    return cleaned_df, report

def calculate_transaction_revenue(df: pd.DataFrame) -> Dict:
    """Calculate revenue based on transaction types - uses derived fields if available"""
    print(f"\n🔄 calculate_transaction_revenue called with {len(df)} rows")
    
    if df.empty:
        print("❌ DataFrame is empty, returning zero values")
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
    
    print(f"Available columns: {list(df.columns)}")
    
    # Check if we have derived fields from cleaning
    if 'revenue_calc' in df.columns and 'shipping_loss_calc' in df.columns:
        print("✅ Using derived fields (revenue_calc, shipping_loss_calc, units_sold_calc)")
        
        # Calculate directly from derived fields
        gross_revenue = df[df['revenue_calc'] > 0]['revenue_calc'].sum()
        refunds = abs(df[df['revenue_calc'] < 0]['revenue_calc'].sum())
        shipping_loss = df['shipping_loss_calc'].sum()
        units_sold = df['units_sold_calc'].sum() if 'units_sold_calc' in df.columns else 0
        orders = df['order_id'].nunique() if 'order_id' in df.columns else 0
        
        # Calculate FreeReplacement cost estimation
        free_replacement_cost = 0.0
        if 'transaction_type' in df.columns:
            freereplacement_data = df[df['transaction_type'] == 'FreeReplacement']
            shipment_data = df[df['transaction_type'] == 'Shipment']
            
            # Find ASIN column (handle different cases)
            asin_col = None
            for col in df.columns:
                if col.lower() == 'asin':
                    asin_col = col
                    break
            
            if asin_col and len(freereplacement_data) > 0:
                for _, row in freereplacement_data.iterrows():
                    asin = row[asin_col]
                    if pd.notna(asin) and asin != '':
                        asin_shipments = shipment_data[shipment_data[asin_col] == asin]
                        
                        if not asin_shipments.empty:
                            # Product cost loss: average revenue_calc for same ASIN
                            avg_revenue = asin_shipments['revenue_calc'].mean()
                            free_replacement_cost += avg_revenue
                            
                            # Shipping loss: 2x average shipping for same ASIN
                            avg_shipping = asin_shipments['Shipping Amount'].mean() if 'Shipping Amount' in asin_shipments.columns else 0.0
                            shipping_loss += avg_shipping * 2
            else:
                # If no ASIN column or no FreeReplacement data, skip estimation
                print(f"⚠️ Warning: Cannot estimate FreeReplacement cost - ASIN column not found or no FreeReplacement data")
        
        net_revenue = gross_revenue - refunds - shipping_loss - free_replacement_cost
        aov = net_revenue / orders if orders > 0 else 0.0
        
        # Build transaction breakdown
        breakdown = {}
        if 'transaction_type' in df.columns:
            for txn in df['transaction_type'].unique():
                if pd.notna(txn):
                    txn_data = df[df['transaction_type'] == txn]
                    breakdown[txn] = {
                        'count': len(txn_data),
                        'revenue': txn_data['revenue_calc'].sum(),
                        'units': txn_data['units_sold_calc'].sum() if 'units_sold_calc' in txn_data.columns else 0,
                        'orders': txn_data['order_id'].nunique() if 'order_id' in txn_data.columns else 0,
                        'description': f'{txn} transactions'
                    }
        
        print(f"  Gross Revenue: ₹{gross_revenue:,.2f}")
        print(f"  Refunds: ₹{refunds:,.2f}")
        print(f"  Shipping Loss: ₹{shipping_loss:,.2f}")
        print(f"  Net Revenue: ₹{net_revenue:,.2f}")
        print(f"  Units Sold: {units_sold}")
        print(f"  Orders: {orders}")
        
        return {
            'gross_revenue': gross_revenue,
            'refunds': refunds,
            'free_replacement_cost': free_replacement_cost,
            'shipping_cost_loss': shipping_loss,
            'net_revenue': net_revenue,
            'units_sold': units_sold,
            'orders': orders,
            'aov': aov,
            'transaction_breakdown': breakdown
        }
    
    # Check if transaction_type column exists, if not use legacy calculation
    if 'transaction_type' not in df.columns:
        print("⚠️ transaction_type column not found, using legacy calculation")
        # Legacy calculation - treat all as shipments
        gross_revenue = df['revenue_in_inr'].sum() if 'revenue_in_inr' in df.columns else 0.0
        units_sold = df['quantity'].sum() if 'quantity' in df.columns else 0
        orders = df['order_id'].nunique() if 'order_id' in df.columns else 0
        aov = gross_revenue / orders if orders > 0 else 0.0
        
        return {
            'gross_revenue': gross_revenue,
            'refunds': 0.0,
            'free_replacement_cost': 0.0,
            'shipping_cost_loss': 0.0,
            'net_revenue': gross_revenue,
            'units_sold': units_sold,
            'orders': orders,
            'aov': aov,
            'transaction_breakdown': {
                'Shipment': {
                    'count': len(df),
                    'revenue': gross_revenue,
                    'description': 'Legacy data (all treated as shipments)'
                }
            }
        }
    
    # Normalize transaction types and handle missing values
    df['normalized_transaction_type'] = df['transaction_type'].apply(normalize_transaction_type)
    
    # Ensure numeric columns have default values
    if 'revenue_in_inr' not in df.columns:
        df['revenue_in_inr'] = 0.0
    if 'shipping_amount' not in df.columns:
        df['shipping_amount'] = 0.0
    if 'quantity' not in df.columns:
        df['quantity'] = 0
    
    # Fill missing values with 0
    df['revenue_in_inr'] = pd.to_numeric(df['revenue_in_inr'], errors='coerce').fillna(0.0)
    df['shipping_amount'] = pd.to_numeric(df['shipping_amount'], errors='coerce').fillna(0.0)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    
    # Initialize results
    results = {
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
    
    # Get shipment data for FreeReplacement calculations
    shipment_data = df[df['normalized_transaction_type'] == 'Shipment']
    
    # Process each transaction type
    for transaction_type in df['normalized_transaction_type'].unique():
        if pd.isna(transaction_type):
            continue
            
        type_data = df[df['normalized_transaction_type'] == transaction_type]
        count = len(type_data)
        
        if transaction_type == 'Shipment':
            # Shipment: positive revenue contribution
            revenue = type_data['revenue_in_inr'].sum()
            units = type_data['quantity'].sum()
            orders = type_data['order_id'].nunique() if 'order_id' in type_data.columns else 0
            
            results['gross_revenue'] += revenue
            results['units_sold'] += units
            results['orders'] += orders
            
            results['transaction_breakdown'][transaction_type] = {
                'count': count,
                'revenue': revenue,
                'units': units,
                'orders': orders,
                'description': 'Successful sales'
            }
            
        elif transaction_type == 'Cancel':
            # Cancel: no revenue contribution
            results['transaction_breakdown'][transaction_type] = {
                'count': count,
                'revenue': 0.0,
                'units': 0,
                'orders': 0,
                'description': 'Cancelled orders (no revenue impact)'
            }
            
        elif transaction_type == 'Refund':
            # Refund: deduct invoice amount (already negative) and shipping loss
            revenue = abs(type_data['revenue_in_inr'].sum())  # Take absolute value
            shipping_loss = type_data['shipping_amount'].sum()
            
            results['refunds'] += revenue
            results['shipping_cost_loss'] += shipping_loss
            
            results['transaction_breakdown'][transaction_type] = {
                'count': count,
                'revenue': -revenue,
                'shipping_loss': shipping_loss,
                'description': 'Customer returns with shipping loss'
            }
            
        elif transaction_type == 'FreeReplacement':
            # FreeReplacement: calculate estimated loss
            replacement_cost = 0.0
            shipping_loss = 0.0
            
            for _, row in type_data.iterrows():
                asin = row['Asin']  # Use capitalized column name
                if pd.notna(asin) and asin != '':
                    # Find average shipment value for this ASIN
                    asin_shipments = shipment_data[shipment_data['Asin'] == asin]  # Use capitalized column name
                    
                    if not asin_shipments.empty:
                        # Product cost loss: average revenue_calc for same ASIN
                        avg_revenue = asin_shipments['revenue_calc'].mean()  # Use revenue_calc instead of revenue_in_inr
                        replacement_cost += avg_revenue
                        
                        # Shipping loss: 2x average shipping for same ASIN
                        avg_shipping = asin_shipments['Shipping Amount'].mean() if 'Shipping Amount' in asin_shipments.columns else 0.0  # Use Shipping Amount
                        shipping_loss += avg_shipping * 2
                    else:
                        # If no shipment data for this ASIN, use current row values
                        current_shipping = row['Shipping Amount'] if pd.notna(row['Shipping Amount']) else 0.0  # Use Shipping Amount
                        shipping_loss += current_shipping * 2
            
            results['free_replacement_cost'] += replacement_cost
            results['shipping_cost_loss'] += shipping_loss
            
            results['transaction_breakdown'][transaction_type] = {
                'count': count,
                'revenue': -replacement_cost,
                'shipping_loss': shipping_loss,
                'description': 'Free replacements with estimated cost'
            }
    
    # Calculate net revenue: Gross Revenue - (Refunds + Shipping Loss + Free Replacement cost)
    results['net_revenue'] = results['gross_revenue'] - results['refunds'] - results['shipping_cost_loss'] - results['free_replacement_cost']
    
    # Calculate AOV
    results['aov'] = results['net_revenue'] / results['orders'] if results['orders'] > 0 else 0.0
    
    # Safe logging: print counts by transaction type
    print("Transaction type counts:")
    for txn_type, data in results['transaction_breakdown'].items():
        print(f"  {txn_type}: {data['count']} records")
    
    return results

def create_sales_table(conn: sqlite3.Connection) -> None:
    """Create sales table with standard schema including derived fields (idempotent)"""
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT,
            order_id TEXT,
            sku TEXT,
            asin TEXT,
            product_name TEXT,
            quantity INTEGER,
            missing_quantity_flag BOOLEAN,
            revenue_in_inr REAL,
            shipping_amount REAL,
            transaction_type TEXT,
            shipment_item_id TEXT,
            region TEXT,
            status TEXT,
            data_source TEXT,
            upload_date TEXT,
            month_tag TEXT,
            normalized_order_id TEXT,
            normalized_sku TEXT,
            normalized_order_date TEXT,
            revenue_calc REAL,
            shipping_loss_calc REAL,
            units_sold_calc INTEGER,
            needs_estimation BOOLEAN,
            UNIQUE(normalized_order_id, normalized_sku, normalized_order_date)
        )
    ''')
    conn.commit()

def create_staging_table(conn: sqlite3.Connection) -> None:
    """Create staging table for CSV ingestion including derived fields"""
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {STAGING_TABLE}')
    cursor.execute(f'''
        CREATE TABLE {STAGING_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT,
            order_id TEXT,
            sku TEXT,
            asin TEXT,
            product_name TEXT,
            quantity INTEGER,
            missing_quantity_flag BOOLEAN,
            revenue_in_inr REAL,
            shipping_amount REAL,
            transaction_type TEXT,
            shipment_item_id TEXT,
            region TEXT,
            status TEXT,
            data_source TEXT,
            upload_date TEXT,
            month_tag TEXT,
            normalized_order_id TEXT,
            normalized_sku TEXT,
            normalized_order_date TEXT,
            revenue_calc REAL,
            shipping_loss_calc REAL,
            units_sold_calc INTEGER,
            needs_estimation BOOLEAN
        )
    ''')
    conn.commit()

def create_dataset_registry_table(conn: sqlite3.Connection) -> None:
    """Create dataset registry table to track uploaded files"""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            date_range_start TEXT,
            date_range_end TEXT,
            column_mappings TEXT,
            file_hash TEXT,
            status TEXT DEFAULT 'active'
        )
    ''')
    conn.commit()

def robust_ingest_csv(df: pd.DataFrame, mappings: Dict[str, Optional[str]], filename: str = "uploaded_file.csv", mode: str = "replace") -> str:
    """Robust CSV ingestion with staging and upsert logic"""
    try:
        print(f"🔄 Starting robust ingestion: {len(df)} rows in {mode} mode")
        print(f"📋 Mappings: {mappings}")
        print(f"📊 DataFrame columns: {list(df.columns)[:10]}...")  # First 10 columns
        
        # Check if DataFrame already has derived fields
        if 'revenue_calc' in df.columns:
            print(f"✅ DataFrame has derived fields already")
            revenue_calc_sum = df['revenue_calc'].sum()
            print(f"   Total revenue_calc in input: ₹{revenue_calc_sum:,.2f}")
        else:
            print(f"⚠️  DataFrame missing derived fields")
        
        # Check transaction_type
        if 'transaction_type' in df.columns:
            txn_values = df['transaction_type'].value_counts()
            print(f"📊 Transaction types in DataFrame:")
            for txn, count in txn_values.items():
                print(f"   {txn}: {count} rows")
        elif mappings.get('transaction_type'):
            mapped_col = mappings['transaction_type']
            if mapped_col in df.columns:
                txn_values = df[mapped_col].value_counts()
                print(f"📊 Transaction types in mapped column '{mapped_col}':")
                for txn, count in txn_values.items():
                    print(f"   {txn}: {count} rows")
        else:
            print(f"⚠️  No transaction_type found in DataFrame or mappings")
        
        # Create clean DataFrame with normalized keys
        clean_df = pd.DataFrame()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Map each field
        if mappings.get("order_date"):
            try:
                clean_df["order_date"] = pd.to_datetime(df[mappings["order_date"]], errors='coerce').dt.strftime('%Y-%m-%d')
            except: 
                clean_df["order_date"] = None
        else: 
            clean_df["order_date"] = None
        
        clean_df["order_id"] = df[mappings["order_id"]].astype(str) if mappings.get("order_id") else None
        clean_df["sku"] = df[mappings["sku"]].astype(str) if mappings.get("sku") else None
        clean_df["asin"] = df[mappings["asin"]].astype(str) if mappings.get("asin") else None
        clean_df["product_name"] = df[mappings["product_name"]].astype(str) if mappings.get("product_name") else None
        
        if mappings.get("quantity"):
            # Handle missing quantities with flag
            clean_df["quantity"] = pd.to_numeric(df[mappings["quantity"]], errors='coerce')
            clean_df["missing_quantity_flag"] = clean_df["quantity"].isna()
            clean_df["quantity"] = clean_df["quantity"].fillna(0).astype(int)
        else: 
            clean_df["quantity"] = 0
            clean_df["missing_quantity_flag"] = False
        
        if mappings.get("revenue_amount"):
            clean_df["revenue_in_inr"] = parse_revenue(df[mappings["revenue_amount"]])
        else: 
            clean_df["revenue_in_inr"] = 0.0
        
        if mappings.get("shipping_amount"):
            clean_df["shipping_amount"] = parse_revenue(df[mappings["shipping_amount"]])
        else: 
            clean_df["shipping_amount"] = 0.0
        
        # Preserve cleaned transaction_type if it exists, otherwise map from source
        if 'transaction_type' in df.columns:
            clean_df["transaction_type"] = df['transaction_type']  # Already cleaned
            print(f"✅ Using cleaned transaction_type column from DataFrame")
        elif mappings.get("transaction_type"):
            clean_df["transaction_type"] = df[mappings["transaction_type"]].astype(str)
            print(f"⚠️  Mapping transaction_type from '{mappings['transaction_type']}'")
        else:
            clean_df["transaction_type"] = None
            print(f"❌ No transaction_type found!")
        
        clean_df["shipment_item_id"] = df[mappings["shipment_item_id"]].astype(str) if mappings.get("shipment_item_id") else None
        clean_df["region"] = df[mappings["region"]].astype(str) if mappings.get("region") else None
        clean_df["status"] = df[mappings["status"]].astype(str) if mappings.get("status") else None
        
        # Add metadata columns
        clean_df["data_source"] = filename
        clean_df["upload_date"] = current_time
        
        # Add derived columns if they exist in the DataFrame (from cleaning)
        clean_df["revenue_calc"] = df["revenue_calc"] if "revenue_calc" in df.columns else 0.0
        clean_df["shipping_loss_calc"] = df["shipping_loss_calc"] if "shipping_loss_calc" in df.columns else 0.0
        clean_df["units_sold_calc"] = df["units_sold_calc"] if "units_sold_calc" in df.columns else 0
        clean_df["needs_estimation"] = df["needs_estimation"] if "needs_estimation" in df.columns else False
        
        # Add month tag for easy filtering
        if clean_df["order_date"].notna().any():
            clean_df["month_tag"] = pd.to_datetime(clean_df["order_date"], errors='coerce').dt.to_period('M').astype(str)
        else:
            clean_df["month_tag"] = None
        
        # Add normalized key fields
        clean_df["normalized_order_id"] = ""
        clean_df["normalized_sku"] = ""
        clean_df["normalized_order_date"] = ""
        
        for idx, row in clean_df.iterrows():
            norm_order_id, norm_sku, norm_order_date = normalize_key_fields(
                row['order_id'], row['sku'], row['order_date']
            )
            clean_df.at[idx, 'normalized_order_id'] = norm_order_id
            clean_df.at[idx, 'normalized_sku'] = norm_sku
            clean_df.at[idx, 'normalized_order_date'] = norm_order_date
        
        # Filter out rows with invalid normalized keys
        valid_rows = clean_df[
            (clean_df['normalized_order_id'] != "") & 
            (clean_df['normalized_sku'] != "") & 
            (clean_df['normalized_order_date'] != "")
        ].copy()
        
        dropped_invalid = len(clean_df) - len(valid_rows)
        print(f"📊 Valid rows: {len(valid_rows)}, Invalid rows: {dropped_invalid}")
        
        if len(valid_rows) == 0:
            return f"❌ No valid rows to process (all rows had missing key data)"
        
        # Start transaction
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        try:
            # Create tables
            create_sales_table(conn)
            create_staging_table(conn)
            create_dataset_registry_table(conn)
            
            if mode == "replace":
                # Clear existing data
                cursor.execute(f'DELETE FROM {TABLE_NAME}')
                print("🗑️ Cleared existing data")
            
            # Load data into staging
            for _, row in valid_rows.iterrows():
                cursor.execute(f'''
                    INSERT INTO {STAGING_TABLE} 
                    (order_date, order_id, sku, asin, product_name, quantity, missing_quantity_flag, 
                     revenue_in_inr, shipping_amount, transaction_type, shipment_item_id, region, status, 
                     data_source, upload_date, month_tag, normalized_order_id, normalized_sku, normalized_order_date,
                     revenue_calc, shipping_loss_calc, units_sold_calc, needs_estimation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['order_date'], row['order_id'], row['sku'], row['asin'], row['product_name'], 
                      row['quantity'], row.get('missing_quantity_flag', False), row['revenue_in_inr'], 
                      row['shipping_amount'], row['transaction_type'], row['shipment_item_id'], 
                      row['region'], row['status'], row['data_source'], row['upload_date'], row['month_tag'],
                      row['normalized_order_id'], row['normalized_sku'], row['normalized_order_date'],
                      row.get('revenue_calc', 0.0), row.get('shipping_loss_calc', 0.0), 
                      row.get('units_sold_calc', 0), row.get('needs_estimation', False)))
            
            # Upsert from staging to main table
            cursor.execute(f'''
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (order_date, order_id, sku, asin, product_name, quantity, missing_quantity_flag, 
                 revenue_in_inr, shipping_amount, transaction_type, shipment_item_id, region, status, 
                 data_source, upload_date, month_tag, normalized_order_id, normalized_sku, normalized_order_date,
                 revenue_calc, shipping_loss_calc, units_sold_calc, needs_estimation)
                SELECT order_date, order_id, sku, asin, product_name, quantity, missing_quantity_flag, 
                       revenue_in_inr, shipping_amount, transaction_type, shipment_item_id, region, status, 
                       data_source, upload_date, month_tag, normalized_order_id, normalized_sku, normalized_order_date,
                       revenue_calc, shipping_loss_calc, units_sold_calc, needs_estimation
                FROM {STAGING_TABLE}
            ''')
            
            # Get statistics
            cursor.execute(f'SELECT COUNT(*) FROM {STAGING_TABLE}')
            total_processed = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
            total_in_main = cursor.fetchone()[0]
            
            # DEBUG: Check if derived fields were stored correctly
            cursor.execute(f'SELECT transaction_type, revenue_calc, shipping_loss_calc, units_sold_calc FROM {TABLE_NAME} LIMIT 5')
            debug_rows = cursor.fetchall()
            print("\n📊 DEBUG: First 5 rows in database with derived fields:")
            for i, row in enumerate(debug_rows):
                print(f"  Row {i+1}: Transaction={row[0]}, revenue_calc={row[1]}, shipping_loss_calc={row[2]}, units_sold_calc={row[3]}")
            
            # Register dataset in registry
            date_range_start = valid_rows["order_date"].min() if valid_rows["order_date"].notna().any() else None
            date_range_end = valid_rows["order_date"].max() if valid_rows["order_date"].notna().any() else None
            
            cursor.execute('''
                INSERT INTO dataset_registry 
                (filename, upload_date, row_count, date_range_start, date_range_end, column_mappings, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, current_time, len(valid_rows), date_range_start, date_range_end, 
                  json.dumps(mappings), hashlib.md5(str(df.values).encode()).hexdigest()))
            
            # Clean up staging
            cursor.execute(f'DROP TABLE {STAGING_TABLE}')
            
            conn.commit()
            
            # Data health report
            cursor.execute(f'SELECT COUNT(DISTINCT normalized_order_id || "|" || normalized_sku || "|" || normalized_order_date) FROM {TABLE_NAME}')
            distinct_keys = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL')
            date_range = cursor.fetchone()
            
            conn.close()
            
            # Build result message
            message = f"✅ Processed {total_processed} rows from {filename}"
            if dropped_invalid > 0:
                message += f" ({dropped_invalid} invalid rows dropped)"
            
            message += f"\n📊 Data Health: {total_in_main:,} total rows, {distinct_keys:,} unique keys"
            if date_range[0] and date_range[1]:
                message += f", Date range: {date_range[0]} to {date_range[1]}"
            
            return message
            
        except Exception as e:
            conn.rollback()
            conn.close()
            raise e
            
    except Exception as e:
        return f"❌ Error during ingestion: {str(e)}"

def store_sales_data(df: pd.DataFrame, mappings: Dict[str, Optional[str]], filename: str = "uploaded_file.csv", mode: str = "replace") -> str:
    """Store CSV data to sales table with persistent storage support"""
    try:
        print(f"DEBUG: Storing {len(df)} rows in {mode} mode")
        print(f"DEBUG: Key columns - order_id: {mappings.get('order_id')}, sku: {mappings.get('sku')}, order_date: {mappings.get('order_date')}")
        # Create clean DataFrame
        clean_df = pd.DataFrame()
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Map each field
        if mappings.get("order_date"):
            try:
                clean_df["order_date"] = pd.to_datetime(df[mappings["order_date"]], errors='coerce').dt.strftime('%Y-%m-%d')
            except: clean_df["order_date"] = None
        else: clean_df["order_date"] = None
        
        clean_df["order_id"] = df[mappings["order_id"]].astype(str) if mappings.get("order_id") else None
        clean_df["sku"] = df[mappings["sku"]].astype(str) if mappings.get("sku") else None
        clean_df["asin"] = df[mappings["asin"]].astype(str) if mappings.get("asin") else None
        clean_df["product_name"] = df[mappings["product_name"]].astype(str) if mappings.get("product_name") else None
        
        if mappings.get("quantity"):
            clean_df["quantity"] = pd.to_numeric(df[mappings["quantity"]], errors='coerce').fillna(0).astype(int)
        else: clean_df["quantity"] = 0
        
        if mappings.get("revenue_amount"):
            clean_df["revenue_in_inr"] = parse_revenue(df[mappings["revenue_amount"]])
        else: clean_df["revenue_in_inr"] = 0.0
        
        if mappings.get("shipping_amount"):
            clean_df["shipping_amount"] = parse_revenue(df[mappings["shipping_amount"]])
        else: clean_df["shipping_amount"] = 0.0
        
        clean_df["transaction_type"] = df[mappings["transaction_type"]].astype(str) if mappings.get("transaction_type") else None
        clean_df["shipment_item_id"] = df[mappings["shipment_item_id"]].astype(str) if mappings.get("shipment_item_id") else None
        clean_df["region"] = df[mappings["region"]].astype(str) if mappings.get("region") else None
        clean_df["status"] = df[mappings["status"]].astype(str) if mappings.get("status") else None
        
        # Add metadata columns
        clean_df["data_source"] = filename
        clean_df["upload_date"] = current_time
        
        # Add month tag for easy filtering
        if clean_df["order_date"].notna().any():
            clean_df["month_tag"] = pd.to_datetime(clean_df["order_date"], errors='coerce').dt.to_period('M').astype(str)
        else:
            clean_df["month_tag"] = None
        
        print(f"DEBUG: Clean data shape: {clean_df.shape}")
        print(f"DEBUG: Sample data - order_id: {clean_df['order_id'].head(3).tolist()}")
        print(f"DEBUG: Sample data - sku: {clean_df['sku'].head(3).tolist()}")
        print(f"DEBUG: Sample data - order_date: {clean_df['order_date'].head(3).tolist()}")
        
        # Store to database
        conn = sqlite3.connect(DB_FILE)
        create_sales_table(conn)
        create_dataset_registry_table(conn)
        
        cursor = conn.cursor()
        
        if mode == "replace":
            # Clear existing data and replace
            cursor.execute(f'DELETE FROM {TABLE_NAME}')
            # Insert new data row by row to handle the unique constraint properly
            for _, row in clean_df.iterrows():
                cursor.execute(f'''
                    INSERT INTO {TABLE_NAME} 
                    (order_date, order_id, sku, asin, product_name, quantity, revenue_in_inr, 
                     shipping_amount, transaction_type, shipment_item_id, region, status, 
                     data_source, upload_date, month_tag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['order_date'], row['order_id'], row['sku'], row['asin'], row['product_name'], 
                      row['quantity'], row['revenue_in_inr'], row['shipping_amount'], 
                      row['transaction_type'], row['shipment_item_id'], row['region'], row['status'],
                      row['data_source'], row['upload_date'], row['month_tag']))
            conn.commit()
            message = f"✅ Replaced data with {len(clean_df)} rows from {filename}"
        else:
            # Append mode - handle duplicates using INSERT OR IGNORE
            initial_count = cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}').fetchone()[0]
            
            # Filter out rows with NULL values in key fields
            valid_rows = clean_df.dropna(subset=['order_id', 'sku', 'order_date'])
            skipped_nulls = len(clean_df) - len(valid_rows)
            
            new_rows = 0
            duplicates_removed = 0
            
            for i, (_, row) in enumerate(valid_rows.iterrows()):
                try:
                    print(f"DEBUG: Processing row {i+1}/{len(valid_rows)}: {row['order_id']}, {row['sku']}, {row['order_date']}")
                    
                    # First check if the record already exists
                    cursor.execute(f'''
                        SELECT COUNT(*) FROM {TABLE_NAME} 
                        WHERE order_id = ? AND sku = ? AND order_date = ?
                    ''', (row['order_id'], row['sku'], row['order_date']))
                    
                    existing_count = cursor.fetchone()[0]
                    if existing_count > 0:
                        print(f"DEBUG: Duplicate found, skipping")
                        duplicates_removed += 1
                        continue
                    
                    # Insert the new record
                    cursor.execute(f'''
                        INSERT INTO {TABLE_NAME} 
                        (order_date, order_id, sku, asin, product_name, quantity, revenue_in_inr, 
                         shipping_amount, transaction_type, shipment_item_id, region, status, 
                         data_source, upload_date, month_tag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['order_date'], row['order_id'], row['sku'], row['asin'], row['product_name'], 
                          row['quantity'], row['revenue_in_inr'], row['shipping_amount'], 
                          row['transaction_type'], row['shipment_item_id'], row['region'], row['status'],
                          row['data_source'], row['upload_date'], row['month_tag']))
                    new_rows += 1
                    print(f"DEBUG: Successfully inserted row {i+1}")
                    
                except sqlite3.IntegrityError as e:
                    # Handle any remaining constraint violations
                    print(f"DEBUG: Integrity error on row {i+1}: {e}")
                    if "UNIQUE constraint failed" in str(e):
                        duplicates_removed += 1
                    else:
                        print(f"Warning: Skipped row due to error: {e}")
                        continue
            
            conn.commit()
            
            message = f"✅ Appended {new_rows} new rows from {filename}"
            if duplicates_removed > 0:
                message += f" ({duplicates_removed} duplicates skipped)"
            if skipped_nulls > 0:
                message += f" ({skipped_nulls} rows with missing key data skipped)"
        
        # Register dataset in registry
        date_range_start = clean_df["order_date"].min() if clean_df["order_date"].notna().any() else None
        date_range_end = clean_df["order_date"].max() if clean_df["order_date"].notna().any() else None
        
        cursor.execute('''
            INSERT INTO dataset_registry 
            (filename, upload_date, row_count, date_range_start, date_range_end, column_mappings, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, current_time, len(clean_df), date_range_start, date_range_end, 
              json.dumps(mappings), hashlib.md5(str(df.values).encode()).hexdigest()))
        
        conn.commit()
        conn.close()
        
        return message
        
    except Exception as e:
        return f"❌ Error storing data: {str(e)}"

def check_existing_data() -> bool:
    """Check if there's existing data in the database"""
    if not os.path.exists(DB_FILE):
        return False
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except:
        return False

def get_dataset_registry() -> List[Dict]:
    """Get list of all uploaded datasets"""
    if not os.path.exists(DB_FILE):
        return []
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT filename, upload_date, row_count, date_range_start, date_range_end, status
            FROM dataset_registry 
            WHERE status = 'active'
            ORDER BY upload_date DESC
        ''')
        results = cursor.fetchall()
        conn.close()
        
        return [{
            'filename': row[0],
            'upload_date': row[1],
            'row_count': row[2],
            'date_range_start': row[3],
            'date_range_end': row[4],
            'status': row[5]
        } for row in results]
    except:
        return []

def get_total_data_summary() -> Dict:
    """Get summary of all stored data"""
    if not os.path.exists(DB_FILE):
        return {}
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Total rows
        cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
        total_rows = cursor.fetchone()[0]
        
        # Date range
        cursor.execute(f'SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL')
        date_range = cursor.fetchone()
        
        # Data sources
        cursor.execute(f'SELECT COUNT(DISTINCT data_source) FROM {TABLE_NAME}')
        data_sources = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_rows': total_rows,
            'date_range_start': date_range[0],
            'date_range_end': date_range[1],
            'data_sources': data_sources
        }
    except:
        return {}

def debug_db_info() -> str:
    """Debug function to show active DB path and row counts"""
    try:
        if not os.path.exists(DB_FILE):
            return f"❌ Database file not found at: {DB_ABSOLUTE_PATH}"
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if sales table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLE_NAME,))
        if not cursor.fetchone():
            conn.close()
            return f"❌ Sales table not found in database at: {DB_ABSOLUTE_PATH}"
        
        # Get row counts
        cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
        total_rows = cursor.fetchone()[0]
        
        cursor.execute(f'SELECT COUNT(DISTINCT normalized_order_id || "|" || normalized_sku || "|" || normalized_order_date) FROM {TABLE_NAME}')
        unique_keys = cursor.fetchone()[0]
        
        cursor.execute(f'SELECT COUNT(DISTINCT data_source) FROM {TABLE_NAME}')
        data_sources = cursor.fetchone()[0]
        
        conn.close()
        
        return f"""📊 Database Info:
🗄️ Path: {DB_ABSOLUTE_PATH}
📈 Total rows: {total_rows:,}
🔑 Unique composite keys: {unique_keys:,}
📁 Data sources: {data_sources}"""
        
    except Exception as e:
        return f"❌ Error checking database: {e}"

def debug_check_duplicates(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> str:
    """Debug function to check potential duplicates in incoming file"""
    try:
        if not os.path.exists(DB_FILE):
            return "❌ No existing database to check against"
        
        # Create normalized keys for incoming data
        incoming_keys = []
        for _, row in df.iterrows():
            order_id = str(row[mappings.get('order_id', '')]) if mappings.get('order_id') else ""
            sku = str(row[mappings.get('sku', '')]) if mappings.get('sku') else ""
            order_date = str(row[mappings.get('order_date', '')]) if mappings.get('order_date') else ""
            
            norm_order_id, norm_sku, norm_order_date = normalize_key_fields(order_id, sku, order_date)
            if norm_order_id and norm_sku and norm_order_date:
                incoming_keys.append((norm_order_id, norm_sku, norm_order_date))
        
        if not incoming_keys:
            return "❌ No valid keys found in incoming data"
        
        # Check against existing database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        duplicates_found = []
        for i, (norm_order_id, norm_sku, norm_order_date) in enumerate(incoming_keys[:5]):  # Check first 5
            cursor.execute(f'''
                SELECT COUNT(*) FROM {TABLE_NAME} 
                WHERE normalized_order_id = ? AND normalized_sku = ? AND normalized_order_date = ?
            ''', (norm_order_id, norm_sku, norm_order_date))
            
            if cursor.fetchone()[0] > 0:
                duplicates_found.append(f"Row {i+1}: {norm_order_id}|{norm_sku}|{norm_order_date}")
        
        conn.close()
        
        if duplicates_found:
            return f"⚠️ Found {len(duplicates_found)} potential duplicates:\n" + "\n".join(duplicates_found)
        else:
            return "✅ No duplicates found in first 5 rows"
            
    except Exception as e:
        return f"❌ Error checking duplicates: {e}"

def get_date_filtered_data(start_date: str, end_date: str, transaction_type: str = None) -> Optional[pd.DataFrame]:
    """Get filtered data for date range and optional transaction type"""
    if not os.path.exists(DB_FILE): 
        print(f"Database file not found: {DB_FILE}")
        return None
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # First, check if table exists and has data
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total_rows = cursor.fetchone()[0]
        print(f"✅ Total rows in database: {total_rows}")
        
        if total_rows == 0:
            print("❌ No data in database")
            conn.close()
            return None
        
        # Check date range in database
        cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL")
        db_date_range = cursor.fetchone()
        print(f"📅 Database date range: {db_date_range[0]} to {db_date_range[1]}")
        print(f"📅 Requested date range: {start_date} to {end_date}")
        
        # Build query with optional transaction type filter
        base_query = f"""
            SELECT * FROM {TABLE_NAME} 
            WHERE order_date >= ? AND order_date <= ?
        """
        params = [start_date, end_date]
        
        if transaction_type and transaction_type != "All":
            base_query += " AND transaction_type = ?"
            params.append(transaction_type)
        
        base_query += " ORDER BY order_date"
        
        print(f"🔍 Executing query with params: {params}")
        
        df = pd.read_sql_query(base_query, conn, params=params)
        conn.close()
        
        print(f"✅ Query returned {len(df)} rows")
        
        if df.empty: 
            print("❌ Query returned empty result")
            return None
        
        # DEBUG SUMMARY
        print("\n📊 DEBUG SUMMARY AFTER LOAD:")
        print(f"Row count: {len(df)}")
        
        # Check column names
        print(f"\nColumn names in DataFrame: {list(df.columns)}")
        
        # Check unique transaction types
        if 'transaction_type' in df.columns:
            unique_transactions = df['transaction_type'].unique()
            print(f"\nUnique Transaction values: {unique_transactions}")
            
            # Sum of revenue per transaction type
            if 'revenue_in_inr' in df.columns:
                print("\nSum of revenue_in_inr per Transaction type:")
                for txn in unique_transactions:
                    txn_sum = df[df['transaction_type'] == txn]['revenue_in_inr'].sum()
                    print(f"  {txn}: ₹{txn_sum:,.2f}")
        
        # Check data types for numeric columns
        print("\nData types for numeric columns:")
        numeric_cols = ['revenue_in_inr', 'shipping_amount', 'quantity']
        for col in numeric_cols:
            if col in df.columns:
                print(f"  {col}: {df[col].dtype}")
                print(f"    - Non-null count: {df[col].notna().sum()}")
                print(f"    - Sum: {df[col].sum()}")
        
        print("\n" + "="*50 + "\n")
        
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        # Ensure numeric columns are properly typed
        if 'revenue_in_inr' in df.columns:
            df['revenue_in_inr'] = pd.to_numeric(df['revenue_in_inr'], errors='coerce').fillna(0.0)
        if 'shipping_amount' in df.columns:
            df['shipping_amount'] = pd.to_numeric(df['shipping_amount'], errors='coerce').fillna(0.0)
        if 'quantity' in df.columns:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        
        print(f"✅ After date conversion and type casting: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"❌ Error getting filtered data: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_business_kpis(df: pd.DataFrame) -> Dict:
    """Compute business-grade KPIs from filtered data using transaction-based logic"""
    if df is None or df.empty:
        print("❌ KPI calculation skipped: DataFrame is None or empty")
        return {}
    
    print(f"\n🧮 Computing KPIs for {len(df)} rows")
    print(f"Columns available: {list(df.columns)}")
    
    kpis = {}
    
    # Calculate transaction-based revenue metrics
    print("\n📊 Calling calculate_transaction_revenue...")
    transaction_revenue = calculate_transaction_revenue(df)
    print(f"✅ Transaction revenue calculated: {transaction_revenue}")
    
    # Use transaction-based metrics
    kpis['gross_revenue'] = transaction_revenue['gross_revenue']
    kpis['refunds'] = transaction_revenue['refunds']
    kpis['free_replacement_cost'] = transaction_revenue['free_replacement_cost']
    kpis['shipping_cost_loss'] = transaction_revenue['shipping_cost_loss']
    kpis['net_revenue'] = transaction_revenue['net_revenue']
    kpis['units_sold'] = transaction_revenue['units_sold']
    kpis['total_orders'] = transaction_revenue['orders']
    kpis['aov'] = transaction_revenue['aov']
    kpis['transaction_breakdown'] = transaction_revenue['transaction_breakdown']
    
    # Legacy compatibility
    kpis['total_revenue'] = kpis['net_revenue']
    
    # WoW comparison (if we have enough data and order_date column)
    if len(df) > 0 and 'order_date' in df.columns:
        current_week_start = df['order_date'].max() - timedelta(days=7)
        previous_week_start = current_week_start - timedelta(days=7)
        
        current_week_data = df[df['order_date'] >= current_week_start]
        previous_week_data = df[(df['order_date'] >= previous_week_start) & (df['order_date'] < current_week_start)]
        
        if len(previous_week_data) > 0:
            prev_revenue = previous_week_data['revenue_in_inr'].sum() if 'revenue_in_inr' in previous_week_data.columns else previous_week_data['revenue_calc'].sum()
            curr_revenue = current_week_data['revenue_in_inr'].sum() if 'revenue_in_inr' in current_week_data.columns else current_week_data['revenue_calc'].sum()
            if prev_revenue > 0:
                kpis['wow_revenue_change'] = ((curr_revenue - prev_revenue) / prev_revenue) * 100
            else:
                kpis['wow_revenue_change'] = 0
        else:
            kpis['wow_revenue_change'] = None
    else:
        kpis['wow_revenue_change'] = None
    
    # Top products by revenue using combined SKU/ASIN identifier
    if 'Sku' in df.columns:
        # Check if ASIN column exists, if not use legacy grouping
        if 'Asin' in df.columns:
            # Group by SKU and ASIN for accurate product identification
            product_revenue = df.groupby(['Sku', 'Asin']).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            
            # Create combined SKU/ASIN display names in format: SKU (ASIN)
            product_revenue['display_name'] = product_revenue.apply(
                lambda row: create_product_identifier(row['Sku'], row['Asin']),
                axis=1
            )
        else:
            # Legacy grouping by SKU only
            product_revenue = df.groupby(['Sku']).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            
            # Create display names using SKU only
            product_revenue['display_name'] = product_revenue['Sku'].apply(
                lambda sku: str(sku) if pd.notna(sku) and str(sku).strip() != '' else "Unknown Product"
            )
        
        # Rename columns to match expected format
        product_revenue = product_revenue.rename(columns={'revenue_calc': 'revenue_in_inr', 'units_sold_calc': 'quantity'})
        
        kpis['top_products_revenue'] = product_revenue.nlargest(10, 'revenue_in_inr')[['display_name', 'revenue_in_inr']]
        kpis['top_products_units'] = product_revenue.nlargest(10, 'quantity')[['display_name', 'quantity']]
    else:
        kpis['top_products_revenue'] = pd.DataFrame(columns=['display_name', 'revenue_in_inr'])
        kpis['top_products_units'] = pd.DataFrame(columns=['display_name', 'quantity'])
    
    # Order status breakdown
    if 'status' in df.columns:
        status_counts = df['status'].value_counts()
        kpis['status_breakdown'] = status_counts
    else:
        kpis['status_breakdown'] = None
    
    # Revenue by region
    if 'region' in df.columns:
        region_revenue = df.groupby('region')['revenue_in_inr'].sum().sort_values(ascending=False).head(10)
        kpis['region_revenue'] = region_revenue
    else:
        kpis['region_revenue'] = None
    
    # Revenue trend
    if 'order_date' in df.columns:
        daily_revenue = df.groupby(df['order_date'].dt.date)['revenue_in_inr'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        kpis['revenue_trend'] = daily_revenue
    else:
        kpis['revenue_trend'] = pd.DataFrame(columns=['date', 'revenue'])
    
    return kpis

def compute_movers_decliners(df: pd.DataFrame, min_units: int = 10) -> Dict:
    """Compute fast movers and decliners"""
    if df is None or df.empty or 'sku' not in df.columns:
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
    
    try:
        # Group by week and SKU/ASIN
        df['week'] = df['order_date'].dt.to_period('W')
        
        # Check if ASIN column exists
        if 'asin' in df.columns:
            weekly_data = df.groupby(['week', 'sku', 'asin']).agg({
                'revenue_in_inr': 'sum',
                'quantity': 'sum'
            }).reset_index()
        else:
            # Legacy grouping by SKU only
            weekly_data = df.groupby(['week', 'sku']).agg({
                'revenue_in_inr': 'sum',
                'quantity': 'sum'
            }).reset_index()
            # Add dummy ASIN column for compatibility
            weekly_data['asin'] = None
        
        # Get current and previous week
        current_week = weekly_data['week'].max()
        previous_week = current_week - 1
        
        current_week_data = weekly_data[weekly_data['week'] == current_week]
        previous_week_data = weekly_data[weekly_data['week'] == previous_week]
        
        if current_week_data.empty or previous_week_data.empty:
            return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
        
        # Merge for comparison
        comparison = current_week_data.merge(
            previous_week_data[['sku', 'revenue_in_inr', 'quantity']], 
            on='sku', 
            suffixes=('_current', '_previous')
        )
        
        # Calculate WoW change
        comparison['wow_change'] = (
            (comparison['revenue_in_inr_current'] - comparison['revenue_in_inr_previous']) / 
            comparison['revenue_in_inr_previous'] * 100
        )
        
        # Create display names using combined SKU/ASIN identifier in format: SKU (ASIN)
        comparison['display_name'] = comparison.apply(
            lambda row: create_product_identifier(row['sku'], row['asin']),
            axis=1
        )
        
        # Decliners: ≥30% drop and ≥N units previous week
        decliners = comparison[
            (comparison['wow_change'] <= -30) & 
            (comparison['quantity_previous'] >= min_units)
        ].sort_values('wow_change')
        
        # Fast movers: ≥30% increase and ≥N units current week
        fast_movers = comparison[
            (comparison['wow_change'] >= 30) & 
            (comparison['quantity_current'] >= min_units)
        ].sort_values('wow_change', ascending=False)
        
        return {
            'decliners': decliners[['display_name', 'revenue_in_inr_current', 'wow_change']],
            'fast_movers': fast_movers[['display_name', 'revenue_in_inr_current', 'wow_change']]
        }
        
    except Exception as e:
        print(f"Error computing movers/decliners: {e}")
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}

def run_sql_query(query: str) -> Tuple[List, List[str]]:
    """Run SQL query and return results"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        return results, columns
    except Exception as e:
        print(f"SQL Error: {e}")
        return [], []

def answer_numeric_question(question: str) -> str:
    """Answer numeric questions with SQL"""
    question_lower = question.lower()
    
    if "total revenue" in question_lower or "revenue" in question_lower:
        # Get transaction-based revenue calculation
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM sales", conn)
        conn.close()
        
        if not df.empty:
            transaction_revenue = calculate_transaction_revenue(df)
            return f"""**Revenue Breakdown:**
• **Net Revenue:** {format_inr(transaction_revenue['net_revenue'])}
• **Gross Revenue:** {format_inr(transaction_revenue['gross_revenue'])}
• **Refunds:** {format_inr(transaction_revenue['refunds'])}
• **Free Replacements:** {format_inr(transaction_revenue['free_replacement_cost'])}"""
    
    elif "orders" in question_lower or "order count" in question_lower:
        results, _ = run_sql_query("SELECT COUNT(DISTINCT order_id) as total_orders FROM sales")
        if results:
            return f"**Total Orders:** {results[0][0]:,}"
    
    elif "units" in question_lower or "quantity" in question_lower:
        results, _ = run_sql_query("SELECT SUM(quantity) as total_units FROM sales")
        if results:
            return f"**Total Units Sold:** {results[0][0]:,}"
    
    elif "top" in question_lower and ("sku" in question_lower or "product" in question_lower or "item" in question_lower or "best" in question_lower):
        limit = 5
        if "top 3" in question_lower: limit = 3
        elif "top 10" in question_lower: limit = 10
        
        # Check if ASIN column exists in database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
        columns = [row[1] for row in cursor.fetchall()]
        conn.close()
        
        if 'asin' in columns:
            results, columns = run_sql_query(f"""
                SELECT sku, asin, SUM(revenue_calc) as revenue 
                FROM sales 
                WHERE sku IS NOT NULL 
                GROUP BY sku, asin
                ORDER BY revenue DESC 
                LIMIT {limit}
            """)
            
            if results:
                response = "**Top Products by Revenue:**\n"
                for i, (sku, asin, revenue) in enumerate(results, 1):
                    display_name = create_product_identifier(sku, asin)
                    response += f"{i}. {display_name}: {format_inr(revenue)}\n"
                return response
        else:
            # Legacy query without ASIN
            results, columns = run_sql_query(f"""
                SELECT sku, SUM(revenue_calc) as revenue 
                FROM sales 
                WHERE sku IS NOT NULL 
                GROUP BY sku
                ORDER BY revenue DESC 
                LIMIT {limit}
            """)
            
            if results:
                response = "**Top Products by Revenue:**\n"
                for i, (sku, revenue) in enumerate(results, 1):
                    display_name = str(sku) if pd.notna(sku) else "Unknown Product"
                    response += f"{i}. {display_name}: {format_inr(revenue)}\n"
                return response
    
    return "I couldn't understand your question. Try asking about revenue, orders, units, or top products."

def answer_descriptive_question(question: str) -> str:
    """Answer descriptive questions with general insights"""
    # Get basic data for descriptive questions
    if not os.path.exists(DB_FILE): return "No data available for analysis."
    
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        
        if df.empty: return "No data available for analysis."
        
        # Basic metrics for descriptive answers
        total_revenue = df['revenue_in_inr'].sum()
        total_orders = df['order_id'].nunique() if 'order_id' in df.columns else 0
        total_units = df['quantity'].sum() if 'quantity' in df.columns else 0
    except:
        return "No data available for analysis."
    
    question_lower = question.lower()
    
    if "trend" in question_lower or "performance" in question_lower:
        return f"Based on the data: Total revenue is {format_inr(total_revenue)}, with {total_orders} orders and {total_units} units sold."
    
    elif "declining" in question_lower or "drop" in question_lower:
        return "To identify declining trends, I'd need time-series data. Currently showing overall totals."
    
    elif "best" in question_lower or "top" in question_lower:
        # Get top product by revenue using combined identifier
        if 'Sku' in df.columns:
            if 'Asin' in df.columns:
                product_revenue = df.groupby(['Sku', 'Asin'])['revenue_calc'].sum().reset_index()
                if not product_revenue.empty:
                    top_product = product_revenue.loc[product_revenue['revenue_calc'].idxmax()]
                    product_name = create_product_identifier(top_product['Sku'], top_product['Asin'])
                    return f"The best performing product is {product_name} with {format_inr(top_product['revenue_calc'])} revenue."
            else:
                # Legacy grouping by SKU only
                product_revenue = df.groupby(['Sku'])['revenue_calc'].sum().reset_index()
                if not product_revenue.empty:
                    top_product = product_revenue.loc[product_revenue['revenue_calc'].idxmax()]
                    product_name = str(top_product['Sku']) if pd.notna(top_product['Sku']) else "Unknown Product"
                    return f"The best performing product is {product_name} with {format_inr(top_product['revenue_calc'])} revenue."
        return "No product data available."
    
    return f"Here's a summary: {total_orders} orders totaling {format_inr(total_revenue)} in revenue."

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="CSV Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'mappings' not in st.session_state:
        st.session_state.mappings = {}
    if 'show_mapping_modal' not in st.session_state:
        st.session_state.show_mapping_modal = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'has_existing_data' not in st.session_state:
        st.session_state.has_existing_data = check_existing_data()
    if 'show_data_management' not in st.session_state:
        st.session_state.show_data_management = False
    if 'show_cleaning_preview' not in st.session_state:
        st.session_state.show_cleaning_preview = False
    if 'cleaning_report' not in st.session_state:
        st.session_state.cleaning_report = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    
    # Header
    st.title("📊 CSV Analytics Dashboard")
    
    # Show database path info
    st.info(f"🗄️ **Database**: {DB_ABSOLUTE_PATH}")
    
    # Show existing data status
    if st.session_state.has_existing_data:
        data_summary = get_total_data_summary()
        st.success(f"✅ **Data Loaded**: {data_summary.get('total_rows', 0):,} rows from {data_summary.get('data_sources', 0)} files")
        if data_summary.get('date_range_start') and data_summary.get('date_range_end'):
            st.info(f"📅 **Date Range**: {data_summary['date_range_start']} to {data_summary['date_range_end']}")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("*Ready to analyze your data! Use the filters below or upload new data.*")
        with col2:
            if st.button("📁 Manage Data", help="View and manage uploaded datasets"):
                st.session_state.show_data_management = not st.session_state.show_data_management
        with col3:
            if st.button("🔄 Refresh Data", help="Reload data from database"):
                st.session_state.has_existing_data = check_existing_data()
                st.rerun()
    else:
        st.markdown("*Upload CSV → Auto-map → View KPIs → Chat about data*")
    
    # Data Management Section
    if st.session_state.show_data_management:
        st.markdown("---")
        st.markdown("### 📁 Data Sources Management")
        
        datasets = get_dataset_registry()
        if datasets:
            st.markdown(f"**Found {len(datasets)} uploaded datasets:**")
            
            for i, dataset in enumerate(datasets):
                with st.expander(f"📄 {dataset['filename']} ({dataset['row_count']:,} rows)", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Uploaded:** {dataset['upload_date']}")
                    with col2:
                        if dataset['date_range_start'] and dataset['date_range_end']:
                            st.markdown(f"**Date Range:** {dataset['date_range_start']} to {dataset['date_range_end']}")
                    with col3:
                        st.markdown(f"**Status:** {dataset['status']}")
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear All Data", type="secondary", help="Remove all data and start fresh"):
                    try:
                        # Clear database tables
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute(f'DELETE FROM {TABLE_NAME}')
                        cursor.execute('DELETE FROM dataset_registry')
                        conn.commit()
                        conn.close()
                        
                        # Delete the database file completely
                        if os.path.exists(DB_FILE):
                            os.remove(DB_FILE)
                        
                        # Reset all session state
                        st.session_state.has_existing_data = False
                        st.session_state.uploaded_df = None
                        st.session_state.mappings = {}
                        st.session_state.show_mapping_modal = False
                        st.session_state.chat_history = []
                        st.session_state.show_data_management = False
                        st.session_state.show_cleaning_preview = False
                        st.session_state.cleaning_report = None
                        st.session_state.cleaned_df = None
                        
                        st.success("✅ All data cleared! Database file deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing data: {e}")
            with col2:
                if st.button("📊 View Summary", help="Show detailed data summary"):
                    summary = get_total_data_summary()
                    st.json(summary)
            with col3:
                if st.button("✅ Close Management", help="Hide data management panel"):
                    st.session_state.show_data_management = False
                    st.rerun()
        else:
            st.info("No datasets found. Upload a CSV file to get started.")
    
    # Sidebar for CSV upload
    with st.sidebar:
        st.header("📁 Upload CSV")
        
        # Show upload mode selection if data exists
        if st.session_state.has_existing_data:
            upload_mode = st.radio(
                "Upload Mode:",
                ["Append to existing data", "Replace all data"],
                help="Append: Add new data to existing dataset\nReplace: Clear all data and start fresh"
            )
        else:
            upload_mode = "Replace all data"
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Check if this is a new file
                is_new = (st.session_state.uploaded_df is None or 
                         list(df.columns) != list(st.session_state.uploaded_df.columns))
                
                if is_new:
                    st.session_state.uploaded_df = df.copy()
                    
                    # Try to load saved mapping
                    saved_mapping = load_mapping(list(df.columns))
                    
                    if saved_mapping:
                        st.session_state.mappings = saved_mapping
                        st.info("✅ Using saved column mapping")
                    else:
                        # Auto-map columns
                        st.session_state.mappings = auto_map_columns(df)
                        
                        # Check for ambiguous fields
                        ambiguous = [field for field, header in st.session_state.mappings.items() 
                                   if header is None and any(normalize_header(h) in 
                                   [normalize_header(s) for s in SYNONYMS.get(field, [])] 
                                   for h in df.columns)]
                        
                        if ambiguous:
                            st.session_state.show_mapping_modal = True
                            st.warning(f"⚠️ {len(ambiguous)} fields need confirmation")
                        else:
                            st.success("✅ All columns mapped automatically!")
                
                # Start data cleaning process
                if st.button("🧹 Clean & Validate Data", type="primary"):
                    if st.session_state.uploaded_df is not None and st.session_state.mappings:
                        # Perform data cleaning
                        cleaned_df, cleaning_report = clean_dataframe_transaction_aware(st.session_state.uploaded_df, st.session_state.mappings)
                        
                        # Store results in session state
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.cleaning_report = cleaning_report
                        st.session_state.show_cleaning_preview = True
                        st.rerun()
                    else:
                        st.error("Please upload a file and complete column mapping first")
                
                # Store data (only if cleaning is complete)
                if st.button("💾 Store Data", type="primary"):
                    if st.session_state.cleaned_df is not None:
                        # Show progress
                        with st.spinner("Storing data..."):
                            mode = "append" if upload_mode == "Append to existing data" else "replace"
                            filename = uploaded_file.name if uploaded_file else "uploaded_file.csv"
                            
                            # Debug info
                            st.info(f"Storing {len(st.session_state.cleaned_df)} rows in {mode} mode...")
                            
                            # Check if derived fields exist
                            has_derived = 'revenue_calc' in st.session_state.cleaned_df.columns
                            if has_derived:
                                st.info(f"✅ Cleaned data has derived fields (revenue_calc, etc.)")
                                # Show summary of derived fields
                                total_revenue_calc = st.session_state.cleaned_df['revenue_calc'].sum()
                                st.write(f"Total revenue_calc in cleaned data: ₹{total_revenue_calc:,.2f}")
                            else:
                                st.warning("⚠️ Cleaned data missing derived fields!")
                            
                            # Save raw and cleaned data
                            try:
                                raw_path, cleaned_path = save_raw_and_cleaned_data(
                                    st.session_state.uploaded_df, 
                                    st.session_state.cleaned_df, 
                                    filename
                                )
                                st.success(f"✅ Saved to: {cleaned_path}")
                            except Exception as e:
                                st.error(f"Error saving files: {e}")
                            
                            # Ingest cleaned data
                            result = robust_ingest_csv(st.session_state.cleaned_df, st.session_state.mappings, filename, mode)
                            st.write(result)
                            
                            if "✅" in result:
                                # Verify data was stored
                                conn = sqlite3.connect(DB_FILE)
                                cursor = conn.cursor()
                                cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
                                row_count = cursor.fetchone()[0]
                                conn.close()
                                
                                st.success(f"✅ Verified: {row_count} rows now in database")
                                
                                # Save mapping
                                save_mapping(list(df.columns), st.session_state.mappings)
                                st.session_state.show_mapping_modal = False
                                st.session_state.show_cleaning_preview = False
                                st.session_state.has_existing_data = check_existing_data()
                                
                                # Show cleaning summary
                                if st.session_state.cleaning_report:
                                    report = st.session_state.cleaning_report
                                    st.success(f"🧹 **Cleaning Summary**: {report['rows_kept']} rows kept, {report['duplicates_found']} duplicates removed, {report['rows_dropped']} rows dropped")
                                
                                st.success("✅ Data stored successfully! You can now view the dashboard.")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Storage failed. Check console output for details.")
                    else:
                        st.warning("Please clean and validate your data first using the 'Clean & Validate Data' button")
                
                # Debug actions
                st.markdown("---")
                st.markdown("### 🔧 Debug Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Show DB Info", help="Show active database path and row counts"):
                        db_info = debug_db_info()
                        st.text_area("Database Information", db_info, height=150)
                
                with col2:
                    if st.button("🔍 Check Duplicates", help="Check for potential duplicates in current file"):
                        if st.session_state.uploaded_df is not None:
                            duplicate_info = debug_check_duplicates(st.session_state.uploaded_df, st.session_state.mappings)
                            st.text_area("Duplicate Check", duplicate_info, height=150)
                        else:
                            st.warning("No file uploaded to check")
            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # Main content area
    if st.session_state.uploaded_df is not None or st.session_state.has_existing_data:
        # Show mapping modal if needed
        if st.session_state.show_mapping_modal:
            st.markdown("---")
            st.markdown("### 🤔 Confirm Column Mapping")
            st.markdown("*Please select the correct column for each field:*")
            
            df = st.session_state.uploaded_df
            ambiguous = [field for field, header in st.session_state.mappings.items() 
                        if header is None and any(normalize_header(h) in 
                        [normalize_header(s) for s in SYNONYMS.get(field, [])] 
                        for h in df.columns)]
            
            for field in ambiguous:
                st.markdown(f"**{field.replace('_', ' ').title()}**")
                
                # Get candidates
                candidates = []
                for header in df.columns:
                    norm = normalize_header(header)
                    for syn in SYNONYMS.get(field, []):
                        if syn in norm or norm in syn:
                            candidates.append(header)
                            break
                
                candidates.append("[Skip this field]")
                
                selected = st.selectbox(
                    f"Map {field} to:",
                    options=candidates,
                    key=f"map_{field}"
                )
                
                if selected != "[Skip this field]":
                    st.session_state.mappings[field] = selected
                else:
                    st.session_state.mappings[field] = None
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✅ Confirm", type="primary"):
                    st.session_state.show_mapping_modal = False
                    st.rerun()
            with col2:
                st.caption("Your choices will be saved for future uploads")
        
        # Show data cleaning preview if needed
        if st.session_state.show_cleaning_preview and st.session_state.cleaning_report:
            st.markdown("---")
            st.markdown("### 🧹 Data Cleaning Preview")
            
            report = st.session_state.cleaning_report
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Rows Read", report['total_rows_read'])
            with col2:
                st.metric("✅ Rows Kept", report['rows_kept'])
            with col3:
                st.metric("🗑️ Rows Dropped", report['rows_dropped'])
            with col4:
                st.metric("🔄 Duplicates Found", report['duplicates_found'])
            
            # Detailed report
            st.markdown("#### 📋 Cleaning Details")
            
            # Columns with missing values
            if report['columns_with_missing']:
                st.markdown("**⚠️ Columns with Missing Values:**")
                for missing_info in report['columns_with_missing']:
                    st.warning(f"• {missing_info}")
            
            # Invalid revenue rows
            if report['invalid_revenue_rows'] > 0:
                st.warning(f"💰 **Invalid Revenue**: {report['invalid_revenue_rows']} rows with negative revenue (fixed to 0)")
            
            # Problematic rows sample
            if report['problematic_rows']:
                st.markdown("**🔍 Sample Problematic Rows:**")
                for i, row_info in enumerate(report['problematic_rows'][:5]):
                    st.text(f"Row {row_info['row_index']}: {', '.join(row_info['issues'])} - {row_info['sample_data']}")
            
            # Cleaning steps
            st.markdown("**🔧 Cleaning Steps Applied:**")
            for step in report['cleaning_steps']:
                st.text(f"• {step}")
            
            # Action buttons
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("✅ Approve & Store Cleaned Data", type="primary", help="Continue with ingestion using cleaned data"):
                    st.session_state.show_cleaning_preview = False
                    st.success("✅ Data cleaning approved! You can now use 'Store Data' to ingest the cleaned data.")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", help="Abort ingestion and return to file upload"):
                    st.session_state.show_cleaning_preview = False
                    st.session_state.cleaned_df = None
                    st.session_state.cleaning_report = None
                    st.info("❌ Data cleaning cancelled. You can modify your data and try again.")
                    st.rerun()
        
        # Main tabs
        tab1, tab2 = st.tabs(["📊 KPIs", "💬 Chat"])
        
        with tab1:
            st.header("📊 Business Dashboard")
            
            # Date and Transaction Type filters
            st.markdown("### 📅 Filters")
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                preset = st.selectbox(
                    "Quick Presets:",
                    ["Custom Range", "Last 7 days", "Last 30 days", "This Month", "Last Month"],
                    key="date_preset"
                )
            
            with col2:
                # Check if transaction_type column exists in database
                if os.path.exists(DB_FILE):
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
                        columns = [row[1] for row in cursor.fetchall()]
                        conn.close()
                        
                        if 'transaction_type' in columns:
                            transaction_type = st.selectbox(
                                "Transaction Type:",
                                ["All", "Shipment", "Cancel", "Refund", "FreeReplacement"],
                                help="Filter by transaction type for analysis"
                            )
                        else:
                            transaction_type = "All"
                            st.info("ℹ️ Transaction type filtering not available for legacy data")
                    except:
                        transaction_type = "All"
                        st.info("ℹ️ Transaction type filtering not available")
                else:
                    transaction_type = "All"
            
            with col3:
                if preset == "Custom Range":
                    # Check if data exists and get its date range
                    if os.path.exists(DB_FILE):
                        try:
                            conn = sqlite3.connect(DB_FILE)
                            cursor = conn.cursor()
                            cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL")
                            db_dates = cursor.fetchone()
                            conn.close()
                            
                            if db_dates[0] and db_dates[1]:
                                default_start = datetime.strptime(db_dates[0], '%Y-%m-%d').date()
                                default_end = datetime.strptime(db_dates[1], '%Y-%m-%d').date()
                                start_date = st.date_input("Start Date", value=default_start)
                                end_date = st.date_input("End Date", value=default_end)
                            else:
                                start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                                end_date = st.date_input("End Date", value=datetime.now().date())
                        except:
                            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                            end_date = st.date_input("End Date", value=datetime.now().date())
                    else:
                        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                        end_date = st.date_input("End Date", value=datetime.now().date())
                else:
                    # For presets, use the actual data range from database
                    if os.path.exists(DB_FILE):
                        try:
                            conn = sqlite3.connect(DB_FILE)
                            cursor = conn.cursor()
                            cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL")
                            db_dates = cursor.fetchone()
                            conn.close()
                            
                            if db_dates[0] and db_dates[1]:
                                # Use the full data range
                                start_date = datetime.strptime(db_dates[0], '%Y-%m-%d').date()
                                end_date = datetime.strptime(db_dates[1], '%Y-%m-%d').date()
                            else:
                                today = datetime.now().date()
                                start_date = today - timedelta(days=30)
                                end_date = today
                        except:
                            today = datetime.now().date()
                            start_date = today - timedelta(days=30)
                            end_date = today
                    else:
                        today = datetime.now().date()
                        if preset == "Last 7 days":
                            start_date = today - timedelta(days=7)
                            end_date = today
                        elif preset == "Last 30 days":
                            start_date = today - timedelta(days=30)
                            end_date = today
                        elif preset == "This Month":
                            start_date = today.replace(day=1)
                            end_date = today
                        elif preset == "Last Month":
                            first_this_month = today.replace(day=1)
                            start_date = (first_this_month - timedelta(days=1)).replace(day=1)
                            end_date = first_this_month - timedelta(days=1)
                        else:
                            start_date = today - timedelta(days=30)
                            end_date = today
                    
                    st.date_input("Start Date", value=start_date, disabled=True)
                    st.date_input("End Date", value=end_date, disabled=True)
            
            with col4:
                st.markdown("**Range:**")
                st.markdown(f"{start_date} to {end_date}")
            
            # Debug section
            with st.expander("🔍 Debug Information", expanded=False):
                if os.path.exists(DB_FILE):
                    try:
                        conn = sqlite3.connect(DB_FILE)
                        cursor = conn.cursor()
                        
                        # Check table existence
                        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
                        table_exists = cursor.fetchone() is not None
                        st.write(f"Table exists: {table_exists}")
                        
                        if table_exists:
                            # Get row count
                            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                            total_rows = cursor.fetchone()[0]
                            st.write(f"Total rows: {total_rows}")
                            
                            # Get date range
                            cursor.execute(f"SELECT MIN(order_date), MAX(order_date) FROM {TABLE_NAME} WHERE order_date IS NOT NULL")
                            date_range = cursor.fetchone()
                            st.write(f"Date range in DB: {date_range[0]} to {date_range[1]}")
                            
                            # Get sample data
                            cursor.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 3")
                            sample_data = cursor.fetchall()
                            st.write("Sample data:")
                            st.write(sample_data)
                        
                        conn.close()
                    except Exception as e:
                        st.write(f"Error: {e}")
                else:
                    st.write("Database file does not exist")
            
            # Get filtered data and compute KPIs
            if st.button("🔄 Refresh Dashboard", type="primary"):
                st.rerun()
            
            # Load and compute KPIs
            df = get_date_filtered_data(str(start_date), str(end_date), transaction_type)
            
            # Debug: Show data loading status
            if df is None:
                st.error("⚠️ No data found in database!")
                st.markdown("""
                ### 🚀 **Quick Start Guide:**
                1. Go to **"📤 Upload CSV"** tab (sidebar)
                2. Upload your CSV file with Transaction, Invoice Amount, etc.
                3. Click **"🧹 Clean & Validate Data"** button
                4. Click **"💾 Store Data"** button
                5. Come back to this dashboard
                
                ### 📊 **Or delete and recreate database:**
                """)
                if st.button("🗑️ Delete Database & Start Fresh"):
                    if os.path.exists(DB_FILE):
                        os.remove(DB_FILE)
                        st.success("✅ Database deleted! Please upload new data.")
                        st.rerun()
                return
            elif df.empty:
                st.warning("⚠️ No data found for the selected date range and filters.")
                st.info(f"💡 Date range: {start_date} to {end_date}, Transaction type: {transaction_type}")
                
                # Show available date range
                try:
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT MIN(order_date), MAX(order_date), COUNT(*) FROM {TABLE_NAME}")
                    db_info = cursor.fetchone()
                    conn.close()
                    
                    st.info(f"""
                    **Data available in database:**
                    - Date range: {db_info[0]} to {db_info[1]}
                    - Total rows: {db_info[2]}
                    
                    **Adjust your date filter to match the available data range.**
                    """)
                except:
                    pass
                return
            else:
                st.success(f"✅ Loaded {len(df)} rows of data")
            
            if df is not None and not df.empty:
                kpis = compute_business_kpis(df)
                movers = compute_movers_decliners(df)
                
                # Main KPI cards
                st.markdown("---")
                st.markdown("### 📈 Key Metrics")
                
                # Primary revenue metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    wow_text = ""
                    if kpis.get('wow_revenue_change') is not None:
                        change = kpis['wow_revenue_change']
                        wow_text = f"WoW: {change:+.1f}%" if change != 0 else "WoW: 0%"
                    
                    st.metric(
                        label="💰 Net Revenue",
                        value=format_inr(kpis['net_revenue']),
                        delta=wow_text if wow_text else None,
                        help="Net Revenue = Gross Revenue - Refunds - Free Replacement Cost"
                    )
                
                with col2:
                    st.metric(
                        label="📈 Gross Revenue",
                        value=format_inr(kpis['gross_revenue']),
                        help="Revenue from Shipment transactions only"
                    )
                
                with col3:
                    st.metric(
                        label="📦 Orders",
                        value=f"{kpis['total_orders']:,}"
                    )
                
                with col4:
                    st.metric(
                        label="💵 AOV",
                        value=format_inr(kpis['aov'])
                    )
                
                # Secondary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="🔄 Refunds",
                        value=format_inr(kpis['refunds']),
                        help="Total refunded amount from Refund transactions"
                    )
                
                with col2:
                    st.metric(
                        label="🆓 Free Replacements",
                        value=format_inr(kpis['free_replacement_cost']),
                        help="Estimated cost of FreeReplacement transactions"
                    )
                
                with col3:
                    st.metric(
                        label="📊 Units Sold",
                        value=f"{kpis['units_sold']:,}",
                        help="Units sold from Shipment transactions only"
                    )
                
                with col4:
                    st.metric(
                        label="🚚 Shipping Loss",
                        value=format_inr(kpis['shipping_cost_loss']),
                        help="Shipping costs from Refunds + 2x shipping for FreeReplacements"
                    )
                
                # Transaction Breakdown
                if kpis.get('transaction_breakdown'):
                    st.markdown("---")
                    st.markdown("### 📊 Transaction Breakdown")
                    
                    breakdown_data = []
                    for txn_type, data in kpis['transaction_breakdown'].items():
                        breakdown_data.append({
                            'Transaction Type': txn_type,
                            'Count': data['count'],
                            'Revenue Impact': format_inr(data['revenue']),
                            'Units': data.get('units', 0),
                            'Orders': data.get('orders', 0),
                            'Description': data['description']
                        })
                    
                    if breakdown_data:
                        breakdown_df = pd.DataFrame(breakdown_data)
                        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                
                # Revenue Trend Chart
                st.markdown("---")
                st.markdown("### 📈 Revenue Trend")
                if not kpis['revenue_trend'].empty:
                    fig = px.line(
                        kpis['revenue_trend'], 
                        x='date', 
                        y='revenue',
                        title="Daily Revenue Trend",
                        labels={'revenue': 'Revenue (₹)', 'date': 'Date'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No revenue trend data available")
                
                # Top Products
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏆 Top 10 Products by Revenue")
                    if not kpis['top_products_revenue'].empty:
                        display_df = kpis['top_products_revenue'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr'].apply(format_inr)
                        display_df = display_df[['display_name', 'Revenue (₹)']]
                        display_df.columns = ['Product', 'Revenue']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "top_products_revenue.csv",
                            "text/csv",
                            key="export_revenue"
                        )
                    else:
                        st.info("No product revenue data available")
                
                with col2:
                    st.markdown("#### 📦 Top 10 Products by Units")
                    if not kpis['top_products_units'].empty:
                        display_df = kpis['top_products_units'].copy()
                        display_df['Units'] = display_df['quantity'].apply(lambda x: f"{x:,}")
                        display_df = display_df[['display_name', 'Units']]
                        display_df.columns = ['Product', 'Units']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "top_products_units.csv",
                            "text/csv",
                            key="export_units"
                        )
                    else:
                        st.info("No product units data available")
                
                # Status and Region breakdowns
                col1, col2 = st.columns(2)
                
                with col1:
                    if kpis['status_breakdown'] is not None:
                        st.markdown("#### 📊 Order Status Breakdown")
                        fig = px.pie(
                            values=kpis['status_breakdown'].values,
                            names=kpis['status_breakdown'].index,
                            title="Order Status Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("#### 📊 Order Status Breakdown")
                        st.info("No status data available")
                
                with col2:
                    if kpis['region_revenue'] is not None:
                        st.markdown("#### 🌍 Revenue by Region")
                        fig = px.bar(
                            x=kpis['region_revenue'].values,
                            y=kpis['region_revenue'].index,
                            orientation='h',
                            title="Revenue by Region (Top 10)",
                            labels={'x': 'Revenue (₹)', 'y': 'Region'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown("#### 🌍 Revenue by Region")
                        st.info("No region data available")
                
                # Movers & Decliners
                st.markdown("---")
                st.markdown("### 📊 Movers & Decliners")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📉 Decliners (≥30% WoW Drop)")
                    if not movers['decliners'].empty:
                        display_df = movers['decliners'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr_current'].apply(format_inr)
                        display_df['WoW Change'] = display_df['wow_change'].apply(lambda x: f"{x:.1f}%")
                        display_df = display_df[['display_name', 'Revenue (₹)', 'WoW Change']]
                        display_df.columns = ['Product', 'Revenue', 'WoW Change']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "decliners.csv",
                            "text/csv",
                            key="export_decliners"
                        )
                    else:
                        st.info("No decliners found")
                
                with col2:
                    st.markdown("#### 📈 Fast Movers (≥30% WoW Growth)")
                    if not movers['fast_movers'].empty:
                        display_df = movers['fast_movers'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_in_inr_current'].apply(format_inr)
                        display_df['WoW Change'] = display_df['wow_change'].apply(lambda x: f"+{x:.1f}%")
                        display_df = display_df[['display_name', 'Revenue (₹)', 'WoW Change']]
                        display_df.columns = ['Product', 'Revenue', 'WoW Change']
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "fast_movers.csv",
                            "text/csv",
                            key="export_movers"
                        )
                    else:
                        st.info("No fast movers found")
                
                # Data summary
                st.markdown("---")
                st.markdown("#### 📋 Data Summary")
                
                # Transaction type breakdown
                if 'transaction_type' in df.columns and not df['transaction_type'].isna().all():
                    txn_breakdown = df['transaction_type'].value_counts()
                    st.markdown("**Transaction Type Breakdown:**")
                    for txn_type, count in txn_breakdown.items():
                        st.text(f"• {txn_type}: {count:,} records")
                else:
                    st.markdown("**Transaction Type Breakdown:**")
                    st.text("• Legacy data (all treated as shipments)")
                
                st.info(f"📊 Showing {len(df):,} rows from {start_date} to {end_date}")
                
            else:
                st.warning("💡 No data found for the selected date range. Please check your date selection or upload data first.")
        
        with tab2:
            st.header("💬 Chat About Your Data")
            st.markdown("*Ask questions about revenue, orders, SKUs, and trends*")
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Chat History")
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.expander(f"Q{i+1}: {question[:50]}{'...' if len(question) > 50 else ''}", expanded=False):
                        st.markdown(f"**Question:** {question}")
                        st.markdown(f"**Answer:** {answer}")
                st.markdown("---")
            
            # Question input
            question = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g., What's the total revenue? Show me top products...",
                key="chat_question"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🚀 Ask", type="primary"):
                    if question.strip():
                        # Determine if numeric or descriptive
                        numeric_keywords = ['total', 'count', 'sum', 'revenue', 'orders', 'units', 'top', 'best']
                        is_numeric = any(keyword in question.lower() for keyword in numeric_keywords)
                        
                        if is_numeric:
                            answer = answer_numeric_question(question)
                        else:
                            answer = answer_descriptive_question(question)
                        
                        # Add to history
                        st.session_state.chat_history.append((question, answer))
                        st.rerun()
                    else:
                        st.warning("Please enter a question.")
            
            with col2:
                st.caption("💡 Try: 'What's the total revenue?', 'Top 5 products', 'How many orders?'")
            
            # Example questions
            st.markdown("---")
            st.markdown("### 💡 Example Questions")
            
            example_cols = st.columns(2)
            with example_cols[0]:
                st.markdown("**Numeric Questions:**")
                st.markdown("- What's the total revenue?")
                st.markdown("- How many orders?")
                st.markdown("- Top 5 products by revenue?")
                st.markdown("- Total units sold?")
            
            with example_cols[1]:
                st.markdown("**Descriptive Questions:**")
                st.markdown("- Show me performance trends")
                st.markdown("- What's our best product?")
                st.markdown("- Any declining trends?")
                st.markdown("- Overall business summary")
    
    else:
        st.info("👆 Upload a CSV file in the sidebar to begin")
        
        # Show sample format
        st.markdown("---")
        st.markdown("### 📝 Expected CSV Format")
        st.markdown("""
        The app automatically recognizes columns like:
        - **Order Date**: Invoice Date, Shipment Date, Order Date, etc.
        - **Order ID**: Order Number, Invoice ID, Order No, etc.
        - **SKU**: SKU, ASIN, Product ID, Item ID, etc.
        - **Product Name**: Item Description, Product Name, Title, etc.
        - **Quantity**: Quantity, Qty, Units, etc.
        - **Revenue**: Invoice Amount, Order Amount, Total Amount, Revenue, etc.
        - **Region**: Region, City, Market, etc.
        - **Status**: Status, Order Status, Fulfillment Status, etc.
        
        Upload any CSV with similar columns and the app will map them automatically!
        """)
        
        st.markdown("---")
        st.markdown("### 🚀 Persistent Data Storage")
        st.markdown("""
        **New Features:**
        - ✅ **Auto-load**: Data persists between sessions
        - ✅ **Append Mode**: Add new monthly data without losing existing data
        - ✅ **Duplicate Detection**: Prevents double-counting of records
        - ✅ **Data Management**: View and manage all uploaded datasets
        - ✅ **Smart Mapping**: Reuses column mappings for similar files
        
        Once you upload your first CSV, the app will remember your data and column mappings!
        """)

if __name__ == "__main__":
    main()