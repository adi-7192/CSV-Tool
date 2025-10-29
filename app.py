"""
CSV Analytics Dashboard - Simple & Clean

Upload CSV → Auto-map columns → View KPIs → Chat about data
"""

import streamlit as st
import pandas as pd
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
from db_manager import store_data, query_data, clear_database, get_row_count, table_exists

# Import AI Assistant
try:
    from ai_assistant import AIAssistant
    AI_ASSISTANT_AVAILABLE = True
except ImportError:
    AI_ASSISTANT_AVAILABLE = False
    print("⚠️ AI Assistant module not available. Chat will use basic functionality.")

# Constants
DB_FILE = 'data.db'
TABLE_NAME = 'sales'
STAGING_TABLE = 'sales_staging'
MAPPING_FILE = 'data/mapping.json'
RAW_DATA_FOLDER = 'data/raw'
CLEANED_DATA_FOLDER = 'data/cleaned'

# Database path logging
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

def cleanup_old_files(folder_path: str, keep_days: int = 7) -> int:
    """Remove timestamped files older than specified days, keeping only the latest version of each base file"""
    if not os.path.exists(folder_path):
        return 0
    
    files_removed = 0
    cutoff_time = datetime.now() - timedelta(days=keep_days)
    
    # Group files by base name to keep only the latest version
    file_groups = {}
    
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
            
        # Extract base name (remove _raw_YYYYMMDD_HHMMSS or _cleaned_YYYYMMDD_HHMMSS)
        base_name = re.sub(r'_(raw|cleaned)_\d{8}_\d{6}\.csv$', '', filename)
        
        if base_name not in file_groups:
            file_groups[base_name] = []
        
        file_groups[base_name].append({
            'filename': filename,
            'path': file_path,
            'mtime': os.path.getmtime(file_path)
        })
    
    # For each group, keep only the latest file and remove others
    for base_name, files in file_groups.items():
        if len(files) <= 1:
            continue  # Only one file, keep it
            
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Keep the newest file, remove the rest
        for file_info in files[1:]:
            try:
                os.remove(file_info['path'])
                files_removed += 1
                print(f"🗑️ Removed old file: {file_info['filename']}")
            except Exception as e:
                print(f"⚠️ Could not remove {file_info['filename']}: {e}")
    
    return files_removed

def save_raw_and_cleaned_data(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, filename: str) -> Tuple[str, str]:
    """Save raw and cleaned data to respective folders with smart file management"""
    base_name = os.path.splitext(filename)[0]
    
    # Use consistent filenames (no timestamps) - files will be overwritten
    raw_filename = f"{base_name}_raw.csv"
    cleaned_filename = f"{base_name}_cleaned.csv"
    
    raw_path = os.path.join(RAW_DATA_FOLDER, raw_filename)
    cleaned_path = os.path.join(CLEANED_DATA_FOLDER, cleaned_filename)
    
    # Save files (overwrite existing files with same name)
    raw_df.to_csv(raw_path, index=False)
    cleaned_df.to_csv(cleaned_path, index=False)
    
    # Clean up old timestamped files in both folders
    raw_removed = cleanup_old_files(RAW_DATA_FOLDER)
    cleaned_removed = cleanup_old_files(CLEANED_DATA_FOLDER)
    
    if raw_removed > 0 or cleaned_removed > 0:
        print(f"🧹 Cleaned up {raw_removed + cleaned_removed} old files")
    
    print(f"💾 Saved files: {raw_filename} and {cleaned_filename}")
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
    """Format amount as INR with Indian number formatting (lakhs and crores)"""
    if amount == 0:
        return "₹0"
    
    # Handle negative amounts
    is_negative = amount < 0
    amount = abs(amount)
    
    # Indian number formatting
    if amount >= 10000000:  # 1 crore = 10 million
        crores = amount / 10000000
        if crores >= 100:
            return f"₹{crores:.1f} Cr" if not is_negative else f"-₹{crores:.1f} Cr"
        else:
            return f"₹{crores:.2f} Cr" if not is_negative else f"-₹{crores:.2f} Cr"
    
    elif amount >= 100000:  # 1 lakh = 100 thousand
        lakhs = amount / 100000
        return f"₹{lakhs:.2f} L" if not is_negative else f"-₹{lakhs:.2f} L"
    
    elif amount >= 1000:  # Thousands
        thousands = amount / 1000
        return f"₹{thousands:.1f}K" if not is_negative else f"-₹{thousands:.1f}K"
    
    else:  # Less than 1000
        return f"₹{amount:.0f}" if not is_negative else f"-₹{amount:.0f}"

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
                cleaned_df.at[idx, 'units_sold_calc'] = int(qty) if pd.notna(qty) and qty > 0 else 1  # Preserve quantity for cost calculation
        
        # Summary after processing
        total_revenue_calc = cleaned_df['revenue_calc'].sum()
        positive_revenue = (cleaned_df['revenue_calc'] > 0).sum()
        negative_revenue = (cleaned_df['revenue_calc'] < 0).sum()
        zero_revenue = (cleaned_df['revenue_calc'] == 0).sum()
        
        # Count FreeReplacement transactions processed
        freereplacement_count = (cleaned_df['transaction_type'] == 'FreeReplacement').sum()
        freereplacement_units = cleaned_df[cleaned_df['transaction_type'] == 'FreeReplacement']['units_sold_calc'].sum()
        
        print(f"\n✅ Derived columns created:")
        print(f"   Total revenue_calc: ₹{total_revenue_calc:,.2f}")
        print(f"   Rows with positive revenue_calc: {positive_revenue}")
        print(f"   Rows with negative revenue_calc: {negative_revenue}")
        print(f"   Rows with zero revenue_calc: {zero_revenue}")
        print(f"   FreeReplacement transactions: {freereplacement_count} (total units: {freereplacement_units})")
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
    
    # Add month tag for easy filtering
    if 'Invoice Date' in cleaned_df.columns and cleaned_df['Invoice Date'].notna().any():
        cleaned_df['month_tag'] = pd.to_datetime(cleaned_df['Invoice Date'], errors='coerce').dt.to_period('M').astype(str)
        print(f"✅ Added month_tag column")
    else:
        cleaned_df['month_tag'] = None
        print(f"⚠️ No valid Invoice Date found, month_tag set to None")
    
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
        orders = df['Invoice Number'].nunique() if 'Invoice Number' in df.columns else 0
        
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
                        'orders': txn_data['Invoice Number'].nunique() if 'Invoice Number' in txn_data.columns else 0,
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
        orders = df['Invoice Number'].nunique() if 'Invoice Number' in df.columns else 0
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
            orders = type_data['Invoice Number'].nunique() if 'Invoice Number' in type_data.columns else 0
            
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

def check_existing_data() -> bool:
    """Check if there's existing data in the database"""
    try:
        # Use DuckDB instead of SQLite
        row_count = get_row_count('sales')
        return row_count > 0
    except:
        return False

def get_dataset_registry() -> List[Dict]:
    """Get list of all uploaded datasets (DuckDB version)"""
    try:
        # Get data from DuckDB instead of SQLite
        if not table_exists('sales'):
            return []
        
        # Get unique months/datasets from DuckDB
        month_query = query_data('SELECT DISTINCT month_tag FROM sales WHERE month_tag IS NOT NULL ORDER BY month_tag DESC')
        
        datasets = []
        for _, row in month_query.iterrows():
            month_tag = row['month_tag']
            # Get basic stats for this month
            stats_query = query_data(f'''
                SELECT 
                    COUNT(*) as row_count,
                    MIN("Invoice Date") as date_range_start,
                    MAX("Invoice Date") as date_range_end
                FROM sales 
                WHERE month_tag = {month_tag}
            ''')
            
            if not stats_query.empty:
                stats = stats_query.iloc[0]
                datasets.append({
                    'filename': f'Month_{month_tag}',
                    'upload_date': '2025-10-26',  # Approximate
                    'row_count': stats['row_count'],
                    'date_range_start': stats['date_range_start'],
                    'date_range_end': stats['date_range_end'],
                    'status': 'active'
                })
        
        return datasets
        
    except Exception as e:
        print(f"Error getting dataset registry: {e}")
        return []

def get_total_data_summary() -> Dict:
    """Get summary of all stored data"""
    try:
        # Use DuckDB instead of SQLite
        # Total rows
        total_rows = get_row_count('sales')
        
        # Date range
        date_query = query_data('SELECT MIN("Invoice Date") as min_date, MAX("Invoice Date") as max_date FROM sales WHERE "Invoice Date" IS NOT NULL')
        date_range = None
        if not date_query.empty:
            min_date = date_query.iloc[0]['min_date']
            max_date = date_query.iloc[0]['max_date']
            date_range = (min_date, max_date)
        
        # Count unique months/datasets from the actual data
        try:
            month_query = query_data('SELECT COUNT(DISTINCT month_tag) as unique_months FROM sales WHERE month_tag IS NOT NULL')
            if not month_query.empty:
                data_sources = month_query.iloc[0]['unique_months']
            else:
                data_sources = 0
        except:
            # Fallback: count distinct invoice dates as proxy for datasets
            try:
                date_count_query = query_data('SELECT COUNT(DISTINCT DATE("Invoice Date")) as unique_dates FROM sales WHERE "Invoice Date" IS NOT NULL')
                if not date_count_query.empty:
                    data_sources = max(1, date_count_query.iloc[0]['unique_dates'] // 30)  # Approximate months
                else:
                    data_sources = 0
            except:
                data_sources = 0
        
        # Get unique records count to check for duplicates
        try:
            unique_query = query_data('SELECT COUNT(DISTINCT "Invoice Number" || \'|\' || "Sku" || \'|\' || "Invoice Date") as unique_count FROM sales')
            unique_records = unique_query.iloc[0]['unique_count'] if not unique_query.empty else total_rows
        except:
            unique_records = total_rows
        
        # Get actual file information from data folders
        file_info = get_uploaded_files_info()
        
        return {
            'total_rows': total_rows,
            'unique_records': unique_records,
            'date_range_start': date_range[0] if date_range and date_range[0] else None,
            'date_range_end': date_range[1] if date_range and date_range[1] else None,
            'data_sources': data_sources,
            'uploaded_files': file_info
        }
    except Exception as e:
        print(f"Error getting data summary: {e}")
        return {}

def get_uploaded_files_info() -> List[Dict]:
    """Get information about uploaded files from data folders"""
    file_info = []
    
    try:
        # Check raw files
        raw_folder = 'data/raw'
        cleaned_folder = 'data/cleaned'
        
        if os.path.exists(raw_folder):
            for filename in os.listdir(raw_folder):
                if filename.endswith('.csv') and not filename.startswith('test_'):
                    raw_path = os.path.join(raw_folder, filename)
                    
                    # Find corresponding cleaned file (handle different naming patterns)
                    cleaned_filename = None
                    cleaned_path = None
                    
                    # Try different patterns for cleaned files
                    patterns = [
                        filename.replace('_raw.csv', '_cleaned.csv'),
                        filename.replace('_raw_', '_cleaned_'),
                        filename.replace('raw', 'cleaned')
                    ]
                    
                    for pattern in patterns:
                        potential_path = os.path.join(cleaned_folder, pattern)
                        if os.path.exists(potential_path):
                            cleaned_filename = pattern
                            cleaned_path = potential_path
                            break
                    
                    # Get file info
                    raw_size = os.path.getsize(raw_path)
                    raw_rows = sum(1 for line in open(raw_path)) - 1  # Subtract header
                    
                    cleaned_size = 0
                    cleaned_rows = 0
                    if cleaned_path and os.path.exists(cleaned_path):
                        cleaned_size = os.path.getsize(cleaned_path)
                        cleaned_rows = sum(1 for line in open(cleaned_path)) - 1  # Subtract header
                    
                    # Extract month from filename
                    base_name = filename.replace('_raw.csv', '').replace('_raw_', '_')
                    month = base_name.replace('Monthly', '').replace('monthly', '')
                    
                    file_info.append({
                        'filename': base_name,
                        'month': month,
                        'raw_rows': raw_rows,
                        'cleaned_rows': cleaned_rows,
                        'raw_size': raw_size,
                        'cleaned_size': cleaned_size,
                        'upload_date': datetime.fromtimestamp(os.path.getmtime(raw_path)).strftime('%Y-%m-%d %H:%M')
                    })
        
        # Sort by upload date (newest first)
        file_info.sort(key=lambda x: x['upload_date'], reverse=True)
        
    except Exception as e:
        print(f"Error getting file info: {e}")
    
    return file_info

def debug_db_info() -> str:
    """Debug function to show active DB path and row counts"""
    try:
        # Use DuckDB instead of SQLite
        if not table_exists('sales'):
            return f"❌ Sales table not found in DuckDB database"
        
        # Get row counts using DuckDB
        total_rows = get_row_count('sales')
        
        # Get unique records count
        unique_query = query_data(f'SELECT COUNT(DISTINCT "Invoice Number" || \'|\' || "Sku" || \'|\' || "Invoice Date") as unique_count FROM sales')
        unique_keys = unique_query.iloc[0]['unique_count'] if not unique_query.empty else 0
        
        # Get data sources count
        sources_query = query_data(f'SELECT COUNT(DISTINCT "Invoice Number") as sources FROM sales')
        data_sources = sources_query.iloc[0]['sources'] if not sources_query.empty else 0
        
        return f"""📊 DuckDB Database Info:
🗄️ Path: data/analytics.duckdb
📈 Total rows: {total_rows:,}
🔑 Unique composite keys: {unique_keys:,}
📁 Data sources: {data_sources}"""
        
    except Exception as e:
        return f"❌ Error checking DuckDB: {e}"

def debug_check_duplicates(df: pd.DataFrame, mappings: Dict[str, Optional[str]]) -> str:
    """Debug function to check potential duplicates in incoming file"""
    try:
        if not table_exists('sales'):
            return "❌ No existing DuckDB database to check against"
        
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
        
        # Check against existing DuckDB database
        duplicates_found = []
        for i, (norm_order_id, norm_sku, norm_order_date) in enumerate(incoming_keys[:5]):  # Check first 5
            duplicate_query = query_data(f'''
                SELECT COUNT(*) as count FROM sales 
                WHERE "Invoice Number" = ? AND "Sku" = ? AND "Invoice Date" = ?
            ''', [norm_order_id, norm_sku, norm_order_date])
            
            if not duplicate_query.empty and duplicate_query.iloc[0]['count'] > 0:
                duplicates_found.append(f"Row {i+1}: {norm_order_id}|{norm_sku}|{norm_order_date}")
        
        if duplicates_found:
            return f"⚠️ Found {len(duplicates_found)} potential duplicates:\n" + "\n".join(duplicates_found)
        else:
            return "✅ No duplicates found in first 5 rows"
            
    except Exception as e:
        return f"❌ Error checking duplicates: {e}"

def get_date_filtered_data(start_date: str, end_date: str, transaction_type: str = None, month_tags: list = None) -> Optional[pd.DataFrame]:
    """Get filtered data for date range and optional transaction type"""
    if not table_exists('sales'): 
        print(f"DuckDB table 'sales' not found")
        return None
    
    try:
        # Check if table has data using DuckDB
        total_rows = get_row_count('sales')
        print(f"✅ Total rows in DuckDB: {total_rows}")
        
        if total_rows == 0:
            print("❌ No data in DuckDB")
            return None
        
        # Check date range in DuckDB
        date_range_query = query_data(f'SELECT MIN("Invoice Date") as min_date, MAX("Invoice Date") as max_date FROM sales WHERE "Invoice Date" IS NOT NULL')
        if not date_range_query.empty:
            db_date_range = (date_range_query.iloc[0]['min_date'], date_range_query.iloc[0]['max_date'])
            print(f"📅 DuckDB date range: {db_date_range[0]} to {db_date_range[1]}")
        print(f"📅 Requested date range: {start_date} to {end_date}")
        
        # Build query with parameters inline (DuckDB doesn't support parameterized queries the same way)
        if transaction_type and transaction_type != "All":
            base_query = f"""
                SELECT * FROM sales 
                WHERE "Invoice Date" >= '{start_date}' AND "Invoice Date" <= '{end_date}' AND "Transaction Type" = '{transaction_type}'
            """
        else:
            base_query = f"""
                SELECT * FROM sales 
                WHERE "Invoice Date" >= '{start_date}' AND "Invoice Date" <= '{end_date}'
            """
        
        # Add month_tag filtering if specified
        if month_tags and len(month_tags) > 0:
            month_list = "', '".join(month_tags)
            base_query += f" AND month_tag IN ('{month_list}')"
        
        base_query += ' ORDER BY "Invoice Date"'
        
        print(f"🔍 Executing DuckDB query: {base_query}")
        
        df = query_data(base_query)
        
        print(f"✅ DuckDB query returned {len(df)} rows")
        
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
        
        # Convert Invoice Date to datetime
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df = df.dropna(subset=['Invoice Date'])
        
        # Ensure numeric columns are properly typed
        if 'Invoice Amount' in df.columns:
            df['Invoice Amount'] = pd.to_numeric(df['Invoice Amount'], errors='coerce').fillna(0.0)
        if 'Quantity' in df.columns:
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
        
        print(f"✅ After date conversion and type casting: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"❌ Error getting filtered data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_months_in_date_range(start_date: str, end_date: str) -> list:
    """Get all months within the specified date range from database"""
    try:
        # Query database for distinct months in the date range using DuckDB-compatible syntax
        # Cast VARCHAR to DATE first, then extract components
        query = f"""
            SELECT DISTINCT 
                EXTRACT(year FROM CAST("Invoice Date" AS DATE)) as year,
                EXTRACT(month FROM CAST("Invoice Date" AS DATE)) as month,
                strftime('%Y-%m', CAST("Invoice Date" AS DATE)) as month_tag,
                strftime('%B %Y', CAST("Invoice Date" AS DATE)) as month_display
            FROM sales 
            WHERE CAST("Invoice Date" AS DATE) >= '{start_date}' 
            AND CAST("Invoice Date" AS DATE) <= '{end_date}'
            AND "Invoice Date" IS NOT NULL
            ORDER BY year, month
        """
        
        months_df = query_data(query)
        
        if months_df.empty:
            return []
        
        # Convert to list of dictionaries with month info
        months = []
        for _, row in months_df.iterrows():
            months.append({
                'year': int(row['year']),
                'month': int(row['month']),
                'month_tag': row['month_tag'],
                'month_display': row['month_display'],
                'month_name': row['month_display'].split()[0]  # Just the month name
            })
        
        return months
    except Exception as e:
        print(f"Error getting months in date range: {e}")
        return []

def get_month_metrics(start_date: str, end_date: str, month_tag: str) -> dict:
    """Get key metrics for a specific month"""
    try:
        query = f"""
            SELECT 
                COUNT(DISTINCT "Invoice Number") as order_count,
                COUNT(DISTINCT "Sku") as sku_count,
                SUM(revenue_calc) as total_revenue,
                SUM(units_sold_calc) as total_units,
                COUNT(*) as total_records
            FROM sales 
            WHERE CAST("Invoice Date" AS DATE) >= '{start_date}' 
            AND CAST("Invoice Date" AS DATE) <= '{end_date}'
            AND strftime('%Y-%m', CAST("Invoice Date" AS DATE)) = '{month_tag}'
        """
        
        result = query_data(query)
        
        if result.empty:
            return {
                'order_count': 0,
                'sku_count': 0,
                'total_revenue': 0.0,
                'total_units': 0,
                'total_records': 0
            }
        
        row = result.iloc[0]
        return {
            'order_count': int(row['order_count']) if pd.notna(row['order_count']) else 0,
            'sku_count': int(row['sku_count']) if pd.notna(row['sku_count']) else 0,
            'total_revenue': float(row['total_revenue']) if pd.notna(row['total_revenue']) else 0.0,
            'total_units': int(row['total_units']) if pd.notna(row['total_units']) else 0,
            'total_records': int(row['total_records']) if pd.notna(row['total_records']) else 0
        }
    except Exception as e:
        print(f"Error getting month metrics for {month_tag}: {e}")
        return {
            'order_count': 0,
            'sku_count': 0,
            'total_revenue': 0.0,
            'total_units': 0,
            'total_records': 0
        }


def debug_transaction_types():
    """Debug function to check what transaction types exist in the database"""
    try:
        result = query_data('SELECT DISTINCT transaction_type FROM sales ORDER BY transaction_type')
        print("Transaction types in database:")
        for row in result.itertuples():
            print(f"  - '{row.transaction_type}'")
        
        # Check counts for each type
        counts = query_data('SELECT transaction_type, COUNT(*) as count FROM sales GROUP BY transaction_type ORDER BY count DESC')
        print("\nTransaction type counts:")
        for row in counts.itertuples():
            print(f"  {row.transaction_type}: {row.count} records")
            
    except Exception as e:
        print(f"Error checking transaction types: {e}")


def calculate_asin_average_prices():
    """
    Calculate average prices for each ASIN from Shipment transactions.
    This provides the baseline pricing for free replacement cost calculations.
    
    Returns:
        dict: Dictionary mapping ASIN to average price
    """
    try:
        query = """
            SELECT 
                "Asin",
                COUNT(*) as transaction_count,
                AVG(ABS("Invoice Amount")) as avg_price,
                MIN(ABS("Invoice Amount")) as min_price,
                MAX(ABS("Invoice Amount")) as max_price,
                STDDEV(ABS("Invoice Amount")) as price_stddev
            FROM sales 
            WHERE transaction_type = 'Shipment' 
            AND "Asin" IS NOT NULL 
            AND "Asin" != ''
            AND ABS("Invoice Amount") > 0
            GROUP BY "Asin"
            HAVING COUNT(*) >= 3
            ORDER BY transaction_count DESC
        """
        
        result = query_data(query)
        
        if result.empty:
            print("⚠️ No ASIN pricing data found")
            return {}
        
        asin_prices = {}
        for _, row in result.iterrows():
            asin = row['Asin']
            avg_price = float(row['avg_price'])
            transaction_count = int(row['transaction_count'])
            price_stddev = float(row['price_stddev']) if pd.notna(row['price_stddev']) else 0
            
            # Use median if standard deviation is too high (outliers present)
            if price_stddev > avg_price * 0.5:  # High variance indicates outliers
                median_query = f"""
                    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS("Invoice Amount")) as median_price
                    FROM sales 
                    WHERE transaction_type = 'Shipment' 
                    AND "Asin" = '{asin}'
                    AND ABS("Invoice Amount") > 0
                """
                median_result = query_data(median_query)
                if not median_result.empty:
                    avg_price = float(median_result.iloc[0]['median_price'])
            
            asin_prices[asin] = {
                'avg_price': avg_price,
                'transaction_count': transaction_count,
                'min_price': float(row['min_price']),
                'max_price': float(row['max_price']),
                'price_stddev': price_stddev
            }
        
        print(f"✅ Calculated average prices for {len(asin_prices)} ASINs")
        return asin_prices
        
    except Exception as e:
        print(f"❌ Error calculating ASIN average prices: {e}")
        return {}


def calculate_free_replacement_cost(start_date: str, end_date: str, month_tag: str = None):
    """
    Calculate the true cost impact of free replacements using ASIN-based average pricing.
    
    Args:
        start_date (str): Start date for filtering
        end_date (str): End date for filtering  
        month_tag (str): Optional month tag for specific month filtering
        
    Returns:
        dict: Detailed breakdown of free replacement costs
    """
    try:
        # Get ASIN average prices
        asin_prices = calculate_asin_average_prices()
        
        if not asin_prices:
            print("⚠️ No ASIN pricing data available for free replacement calculation")
            return {
                'total_cost': 0.0,
                'transaction_count': 0,
                'asin_breakdown': {},
                'fallback_used': True,
                'overall_avg_price': 0.0
            }
        
        # Build date filter condition
        date_filter = f"""
            CAST("Invoice Date" AS DATE) >= '{start_date}' 
            AND CAST("Invoice Date" AS DATE) <= '{end_date}'
        """
        
        if month_tag:
            date_filter += f" AND strftime('%Y-%m', CAST(\"Invoice Date\" AS DATE)) = '{month_tag}'"
        
        # Query free replacement transactions
        query = f"""
            SELECT 
                "Asin",
                COUNT(*) as replacement_count,
                SUM(ABS(units_sold_calc)) as total_units
            FROM sales 
            WHERE transaction_type IN ('FreeReplacement', 'Free Replacement', 'Free_Replacement')
            AND {date_filter}
            AND "Asin" IS NOT NULL 
            AND "Asin" != ''
            GROUP BY "Asin"
            ORDER BY replacement_count DESC
        """
        
        result = query_data(query)
        
        if result.empty:
            print(f"ℹ️ No free replacement transactions found for the specified period")
            return {
                'total_cost': 0.0,
                'transaction_count': 0,
                'asin_breakdown': {},
                'fallback_used': False,
                'overall_avg_price': 0.0
            }
        
        # Calculate overall average price as fallback
        overall_avg_query = f"""
            SELECT AVG(ABS("Invoice Amount")) as overall_avg
            FROM sales 
            WHERE transaction_type = 'Shipment'
            AND {date_filter}
            AND ABS("Invoice Amount") > 0
        """
        overall_avg_result = query_data(overall_avg_query)
        overall_avg_price = float(overall_avg_result.iloc[0]['overall_avg']) if not overall_avg_result.empty else 0.0
        
        # Calculate costs for each ASIN
        total_cost = 0.0
        total_transactions = 0
        asin_breakdown = {}
        fallback_used = False
        
        for _, row in result.iterrows():
            asin = row['Asin']
            replacement_count = int(row['replacement_count'])
            total_units = int(row['total_units'])
            
            # Get average price for this ASIN
            if asin in asin_prices:
                avg_price = asin_prices[asin]['avg_price']
                price_source = 'asin_specific'
            else:
                avg_price = overall_avg_price
                price_source = 'overall_average'
                fallback_used = True
                print(f"⚠️ Using overall average price for ASIN {asin} (no specific pricing data)")
            
            # Calculate cost: 2x average price per unit (original + replacement)
            asin_cost = total_units * avg_price * 2
            total_cost += asin_cost
            total_transactions += replacement_count
            
            asin_breakdown[asin] = {
                'replacement_count': replacement_count,
                'total_units': total_units,
                'avg_price': avg_price,
                'cost_per_unit': avg_price * 2,
                'total_cost': asin_cost,
                'price_source': price_source
            }
        
        print(f"✅ Free replacement cost calculation complete:")
        print(f"   Total transactions: {total_transactions}")
        print(f"   Total units: {sum(b['total_units'] for b in asin_breakdown.values())}")
        print(f"   Total cost: ₹{total_cost:,.2f}")
        print(f"   Fallback pricing used: {fallback_used}")
        
        return {
            'total_cost': total_cost,
            'transaction_count': total_transactions,
            'asin_breakdown': asin_breakdown,
            'fallback_used': fallback_used,
            'overall_avg_price': overall_avg_price
        }
        
    except Exception as e:
        print(f"❌ Error calculating free replacement cost: {e}")
        return {
            'total_cost': 0.0,
            'transaction_count': 0,
            'asin_breakdown': {},
            'fallback_used': True,
            'overall_avg_price': 0.0
        }


def display_free_replacement_breakdown(metrics: dict):
    """
    Display detailed breakdown of free replacement costs in the UI.
    
    Args:
        metrics (dict): Metrics dictionary containing free_replacement_breakdown
    """
    if 'free_replacement_breakdown' not in metrics:
        return
    
    breakdown = metrics['free_replacement_breakdown']
    
    if breakdown['transaction_count'] == 0:
        st.info("ℹ️ No free replacement transactions found for this period.")
        return
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="🎁 Free Replacements",
            value=f"{breakdown['transaction_count']:,}",
            help="Number of free replacement transactions"
        )
    with col2:
        st.metric(
            label="💰 Total Cost Impact",
            value=format_inr(breakdown['total_cost']),
            help="True business cost (2x average ASIN price)"
        )
    with col3:
        avg_cost_per_unit = breakdown['total_cost'] / sum(b['total_units'] for b in breakdown['asin_breakdown'].values()) if sum(b['total_units'] for b in breakdown['asin_breakdown'].values()) > 0 else 0
        st.metric(
            label="📊 Avg Cost per Unit",
            value=format_inr(avg_cost_per_unit),
            help="Average cost per replacement unit"
        )
    
    # Display detailed breakdown
    if breakdown['asin_breakdown']:
        st.markdown("#### 🔍 **ASIN Breakdown**")
        
        # Create a DataFrame for better display
        breakdown_data = []
        for asin, data in breakdown['asin_breakdown'].items():
            breakdown_data.append({
                'ASIN': asin,
                'Replacements': data['replacement_count'],
                'Units': data['total_units'],
                'Avg Price': format_inr(data['avg_price']),
                'Cost per Unit': format_inr(data['cost_per_unit']),
                'Total Cost': format_inr(data['total_cost']),
                'Price Source': 'ASIN-specific' if data['price_source'] == 'asin_specific' else 'Overall average'
            })
        
        df_breakdown = pd.DataFrame(breakdown_data)
        st.dataframe(df_breakdown, use_container_width=True)
        
        # Show calculation methodology
        with st.expander("ℹ️ **Calculation Methodology**"):
            st.markdown("""
            **Free Replacement Cost Calculation:**
            
            1. **ASIN Price Lookup**: For each free replacement ASIN, we calculate the average selling price from regular Shipment transactions
            2. **Cost Multiplier**: Each free replacement costs 2x the average price (original product + replacement product)
            3. **Fallback Pricing**: If no ASIN-specific pricing exists, we use the overall average product price
            4. **Outlier Handling**: For ASINs with high price variance, we use median instead of average
            
            **Business Impact:**
            - Original product cost (lost revenue)
            - Replacement product cost (additional inventory)
            - Shipping charges for both shipments
            - Customer service overhead
            
            This gives you the true financial impact of free replacements on your business.
            """)
        
        if breakdown['fallback_used']:
            st.warning(f"⚠️ **Fallback Pricing Used**: Some ASINs used overall average price (₹{format_inr(breakdown['overall_avg_price'])}) due to insufficient pricing data.")


def get_month_metrics_by_transaction_type(start_date: str, end_date: str, month_tag: str, transaction_type: str) -> dict:
    """Get key metrics for a specific month filtered by transaction type"""
    try:
        # Define transaction type filters with common variations
        transaction_filters = {
            'Revenue (Shipments)': "transaction_type = 'Shipment'",
            'Refunds': "transaction_type = 'Refund'",
            'Free Replacements': "transaction_type IN ('FreeReplacement', 'Free Replacement', 'Free_Replacement')",
            'All Transactions': "1=1"  # No filter
        }
        
        filter_condition = transaction_filters.get(transaction_type, "1=1")
        
        # Debug: Print the filter condition
        print(f"Debug: Filtering {transaction_type} with condition: {filter_condition}")
        
        # Special handling for Free Replacements - use ASIN-based cost calculation
        if transaction_type == 'Free Replacements':
            free_replacement_data = calculate_free_replacement_cost(start_date, end_date, month_tag)
            
            # Get basic transaction counts
            count_query = f"""
                SELECT 
                    COUNT(DISTINCT "Invoice Number") as order_count,
                    COUNT(DISTINCT "Sku") as sku_count,
                    COUNT(*) as total_records,
                    SUM(ABS(units_sold_calc)) as total_units
                FROM sales 
                WHERE CAST("Invoice Date" AS DATE) >= '{start_date}' 
                AND CAST("Invoice Date" AS DATE) <= '{end_date}'
                AND strftime('%Y-%m', CAST("Invoice Date" AS DATE)) = '{month_tag}'
                AND {filter_condition}
            """
            
            count_result = query_data(count_query)
            
            if count_result.empty:
                return {
                    'order_count': 0,
                    'sku_count': 0,
                    'total_amount': 0.0,
                    'total_units': 0,
                    'total_records': 0,
                    'transaction_type': transaction_type,
                    'free_replacement_breakdown': free_replacement_data
                }
            
            row = count_result.iloc[0]
            return {
                'order_count': int(row['order_count']) if pd.notna(row['order_count']) else 0,
                'sku_count': int(row['sku_count']) if pd.notna(row['sku_count']) else 0,
                'total_amount': free_replacement_data['total_cost'],  # Use calculated cost
                'total_units': int(row['total_units']) if pd.notna(row['total_units']) else 0,
                'total_records': int(row['total_records']) if pd.notna(row['total_records']) else 0,
                'transaction_type': transaction_type,
                'free_replacement_breakdown': free_replacement_data
            }
        
        # Standard calculation for other transaction types
        query = f"""
            SELECT 
                COUNT(DISTINCT "Invoice Number") as order_count,
                COUNT(DISTINCT "Sku") as sku_count,
                SUM(CASE 
                    WHEN transaction_type = 'Shipment' THEN ABS("Invoice Amount")
                    WHEN transaction_type = 'Refund' THEN ABS("Invoice Amount")
                    WHEN transaction_type = 'FreeReplacement' THEN ABS("Invoice Amount") * 2
                    ELSE 0
                END) as total_amount,
                SUM(CASE 
                    WHEN transaction_type = 'Shipment' THEN ABS(units_sold_calc)
                    WHEN transaction_type = 'Refund' THEN ABS(units_sold_calc)
                    WHEN transaction_type = 'FreeReplacement' THEN ABS(units_sold_calc)
                    ELSE 0
                END) as total_units,
                COUNT(*) as total_records
            FROM sales 
            WHERE CAST("Invoice Date" AS DATE) >= '{start_date}' 
            AND CAST("Invoice Date" AS DATE) <= '{end_date}'
            AND strftime('%Y-%m', CAST("Invoice Date" AS DATE)) = '{month_tag}'
            AND {filter_condition}
            AND CASE 
                WHEN transaction_type = 'Shipment' THEN ABS("Invoice Amount")
                WHEN transaction_type = 'Refund' THEN ABS("Invoice Amount")
                WHEN transaction_type = 'FreeReplacement' THEN ABS("Invoice Amount") * 2
                ELSE 0
            END > 0
        """
        
        result = query_data(query)
        
        if result.empty:
            return {
                'order_count': 0,
                'sku_count': 0,
                'total_amount': 0.0,
                'total_units': 0,
                'total_records': 0,
                'transaction_type': transaction_type
            }
        
        row = result.iloc[0]
        return {
            'order_count': int(row['order_count']) if pd.notna(row['order_count']) else 0,
            'sku_count': int(row['sku_count']) if pd.notna(row['sku_count']) else 0,
            'total_amount': float(row['total_amount']) if pd.notna(row['total_amount']) else 0.0,
            'total_units': int(row['total_units']) if pd.notna(row['total_units']) else 0,
            'total_records': int(row['total_records']) if pd.notna(row['total_records']) else 0,
            'transaction_type': transaction_type
        }
    except Exception as e:
        print(f"Error getting month metrics for {month_tag} ({transaction_type}): {e}")
        return {
            'order_count': 0,
            'sku_count': 0,
            'total_amount': 0.0,
            'total_units': 0,
            'total_records': 0,
            'transaction_type': transaction_type
        }

def calculate_mom_change(current_metrics: dict, previous_metrics: dict) -> dict:
    """Calculate month-over-month percentage changes"""
    changes = {}
    
    # Use 'total_amount' for transaction-specific metrics, fallback to 'total_revenue' for backward compatibility
    amount_key = 'total_amount' if 'total_amount' in current_metrics else 'total_revenue'
    
    for metric in [amount_key, 'order_count', 'sku_count', 'total_units']:
        current = current_metrics.get(metric, 0)
        previous = previous_metrics.get(metric, 0)
        
        if previous > 0:
            change_pct = ((current - previous) / previous) * 100
            changes[metric] = {
                'value': change_pct,
                'direction': 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'flat',
                'formatted': f"{change_pct:+.1f}%"
            }
        else:
            changes[metric] = {
                'value': 0,
                'direction': 'flat',
                'formatted': "N/A"
            }
    
    return changes

def get_available_months() -> list:
    """Get list of available months from DuckDB database"""
    if not table_exists('sales'):
        print("❌ Sales table does not exist")
        return []
    
    try:
        # First check if month_tag column exists
        table_info = query_data("DESCRIBE sales")
        columns = table_info['column_name'].tolist() if not table_info.empty else []
        
        if 'month_tag' not in columns:
            print("❌ month_tag column not found in sales table")
            # Try to extract months from Invoice Date instead
            months_query = query_data("""
                SELECT DISTINCT 
                    EXTRACT(YEAR FROM "Invoice Date") as year,
                    EXTRACT(MONTH FROM "Invoice Date") as month,
                    CONCAT(EXTRACT(YEAR FROM "Invoice Date"), '-', LPAD(EXTRACT(MONTH FROM "Invoice Date")::VARCHAR, 2, '0')) as month_tag
                FROM sales 
                WHERE "Invoice Date" IS NOT NULL 
                ORDER BY year, month
            """)
            if not months_query.empty:
                months = months_query['month_tag'].tolist()
                print(f"✅ Extracted {len(months)} months from Invoice Date: {months}")
                return months
            else:
                print("❌ No valid Invoice Date found for month extraction")
                return []
        
        # Use month_tag column
        months_query = query_data("SELECT DISTINCT month_tag FROM sales WHERE month_tag IS NOT NULL ORDER BY month_tag")
        months = months_query['month_tag'].tolist() if not months_query.empty else []
        
        if months:
            print(f"✅ Found {len(months)} months in month_tag column: {months}")
        else:
            print("❌ No months found in month_tag column")
            # Check if there are any rows at all
            total_rows = get_row_count('sales')
            print(f"📊 Total rows in sales table: {total_rows}")
            
            # Check sample month_tag values
            sample_query = query_data("SELECT month_tag FROM sales LIMIT 5")
            if not sample_query.empty:
                sample_values = sample_query['month_tag'].tolist()
                print(f"📊 Sample month_tag values: {sample_values}")
        
        return months
    except Exception as e:
        print(f"❌ Error getting available months from DuckDB: {e}")
        import traceback
        traceback.print_exc()
        return []

def calculate_mom_comparison(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> Dict:
    """Calculate Month-over-Month comparison metrics"""
    if current_df.empty or previous_df.empty:
        return {}
    
    # Calculate current month metrics
    current_revenue = current_df['revenue_calc'].sum() if 'revenue_calc' in current_df.columns else 0
    current_units = current_df['units_sold_calc'].sum() if 'units_sold_calc' in current_df.columns else 0
    current_orders = current_df['Invoice Number'].nunique() if 'Invoice Number' in current_df.columns else 0
    
    # Calculate previous month metrics
    prev_revenue = previous_df['revenue_calc'].sum() if 'revenue_calc' in previous_df.columns else 0
    prev_units = previous_df['units_sold_calc'].sum() if 'units_sold_calc' in previous_df.columns else 0
    prev_orders = previous_df['Invoice Number'].nunique() if 'Invoice Number' in previous_df.columns else 0
    
    # Calculate growth rates
    revenue_growth = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    units_growth = ((current_units - prev_units) / prev_units * 100) if prev_units > 0 else 0
    orders_growth = ((current_orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0
    
    return {
        'current_revenue': current_revenue,
        'previous_revenue': prev_revenue,
        'revenue_growth': revenue_growth,
        'current_units': current_units,
        'previous_units': prev_units,
        'units_growth': units_growth,
        'current_orders': current_orders,
        'previous_orders': prev_orders,
        'orders_growth': orders_growth
    }

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
        # Convert order_date to datetime if it's a string
        if df['order_date'].dtype == 'object':
            df['order_date'] = pd.to_datetime(df['order_date'])
        
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
    # Handle both uppercase and lowercase column names for database compatibility
    sku_col = 'Sku' if 'Sku' in df.columns else 'sku' if 'sku' in df.columns else None
    asin_col = 'Asin' if 'Asin' in df.columns else 'asin' if 'asin' in df.columns else None
    
    if sku_col:
        # Check if ASIN column exists, if not use legacy grouping
        if asin_col:
            # Group by SKU and ASIN for accurate product identification
            product_revenue = df.groupby([sku_col, asin_col]).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            
            # Create combined SKU/ASIN display names in format: SKU (ASIN)
            product_revenue['display_name'] = product_revenue.apply(
                lambda row: create_product_identifier(row[sku_col], row[asin_col]),
                axis=1
            )
        else:
            # Legacy grouping by SKU only
            product_revenue = df.groupby([sku_col]).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            
            # Create display names using SKU only
            product_revenue['display_name'] = product_revenue[sku_col].apply(
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
    
    # Revenue by region (with case normalization)
    if 'Ship To City' in df.columns:
        # Normalize region names to title case for consistent grouping
        df_normalized = df.copy()
        df_normalized['region_normalized'] = df_normalized['Ship To City'].str.strip().str.title()
        
        # Use revenue_calc if revenue_in_inr is not available
        revenue_col = 'revenue_in_inr' if 'revenue_in_inr' in df.columns else 'revenue_calc'
        
        region_revenue = df_normalized.groupby('region_normalized')[revenue_col].sum().sort_values(ascending=False).head(10)
        kpis['region_revenue'] = region_revenue
    else:
        kpis['region_revenue'] = None
    
    # Revenue trend
    if 'Invoice Date' in df.columns:
        # Ensure Invoice Date is datetime before using .dt accessor
        if df['Invoice Date'].dtype == 'object':
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        
        daily_revenue = df.groupby(df['Invoice Date'].dt.date)['revenue_calc'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        kpis['revenue_trend'] = daily_revenue
    else:
        kpis['revenue_trend'] = pd.DataFrame(columns=['date', 'revenue'])
    
    # Month-based breakdowns
    if 'month_tag' in df.columns:
        # Monthly revenue breakdown
        monthly_revenue = df.groupby('month_tag')['revenue_calc'].sum().sort_index()
        kpis['monthly_revenue'] = monthly_revenue
        
        # Monthly units breakdown
        monthly_units = df.groupby('month_tag')['units_sold_calc'].sum().sort_index()
        kpis['monthly_units'] = monthly_units
        
        # Monthly orders breakdown
        monthly_orders = df.groupby('month_tag')['Invoice Number'].nunique().sort_index()
        kpis['monthly_orders'] = monthly_orders
        
        # Top products by month
        if sku_col and asin_col:
            monthly_products = df.groupby(['month_tag', sku_col, asin_col]).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            
            # Create display names
            monthly_products['display_name'] = monthly_products.apply(
                lambda row: create_product_identifier(row[sku_col], row[asin_col]),
                axis=1
            )
            
            kpis['monthly_products'] = monthly_products
        else:
            kpis['monthly_products'] = pd.DataFrame()
    else:
        kpis['monthly_revenue'] = pd.Series()
        kpis['monthly_units'] = pd.Series()
        kpis['monthly_orders'] = pd.Series()
        kpis['monthly_products'] = pd.DataFrame()
    
    return kpis

def compute_movers_decliners(df: pd.DataFrame, min_units: int = 10) -> Dict:
    """Compute fast movers and decliners"""
    if df is None or df.empty or 'Sku' not in df.columns:
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
    
    try:
        # Ensure Invoice Date is datetime
        if 'Invoice Date' in df.columns:
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
            df = df.dropna(subset=['Invoice Date'])
        
        if df.empty:
            return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
        
        # Group by week and SKU/ASIN
        # Ensure Invoice Date is datetime before using .dt accessor
        if df['Invoice Date'].dtype == 'object':
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        
        df['week'] = df['Invoice Date'].dt.to_period('W')
        
        # Check if ASIN column exists
        if 'Asin' in df.columns:
            weekly_data = df.groupby(['week', 'Sku', 'Asin']).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
        else:
            # Legacy grouping by SKU only
            weekly_data = df.groupby(['week', 'Sku']).agg({
                'revenue_calc': 'sum',
                'units_sold_calc': 'sum'
            }).reset_index()
            # Add dummy ASIN column for compatibility
            weekly_data['Asin'] = None
        
        # Get current and previous week
        current_week = weekly_data['week'].max()
        previous_week = current_week - 1
        
        current_week_data = weekly_data[weekly_data['week'] == current_week]
        previous_week_data = weekly_data[weekly_data['week'] == previous_week]
        
        if current_week_data.empty or previous_week_data.empty:
            return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}
        
        # Merge for comparison
        comparison = current_week_data.merge(
            previous_week_data[['Sku', 'revenue_calc', 'units_sold_calc']], 
            on='Sku', 
            suffixes=('_current', '_previous')
        )
        
        # Calculate WoW change
        comparison['wow_change'] = (
            (comparison['revenue_calc_current'] - comparison['revenue_calc_previous']) / 
            comparison['revenue_calc_previous'] * 100
        )
        
        # Create display names using combined SKU/ASIN identifier in format: SKU (ASIN)
        comparison['display_name'] = comparison.apply(
            lambda row: create_product_identifier(row['Sku'], row['Asin']),
            axis=1
        )
        
        # Decliners: ≥30% drop and ≥N units previous week
        decliners = comparison[
            (comparison['wow_change'] <= -30) & 
            (comparison['units_sold_calc_previous'] >= min_units)
        ].sort_values('wow_change')
        
        # Fast movers: ≥30% increase and ≥N units current week
        fast_movers = comparison[
            (comparison['wow_change'] >= 30) & 
            (comparison['units_sold_calc_current'] >= min_units)
        ].sort_values('wow_change', ascending=False)
        
        return {
            'decliners': decliners[['display_name', 'revenue_calc_current', 'wow_change']],
            'fast_movers': fast_movers[['display_name', 'revenue_calc_current', 'wow_change']]
        }
        
    except Exception as e:
        print(f"Error computing movers/decliners: {e}")
        return {'decliners': pd.DataFrame(), 'fast_movers': pd.DataFrame()}


def answer_numeric_question(question: str) -> str:
    """Answer numeric questions with DuckDB SQL"""
    question_lower = question.lower()
    
    if "total revenue" in question_lower or "revenue" in question_lower:
        # Get transaction-based revenue calculation using DuckDB
        df = query_data("SELECT * FROM sales")
        
        if not df.empty:
            transaction_revenue = calculate_transaction_revenue(df)
            return f"""**Revenue Breakdown:**
• **Net Revenue:** {format_inr(transaction_revenue['net_revenue'])}
• **Gross Revenue:** {format_inr(transaction_revenue['gross_revenue'])}
• **Refunds:** {format_inr(transaction_revenue['refunds'])}
• **Free Replacements:** {format_inr(transaction_revenue['free_replacement_cost'])}"""
    
    elif "orders" in question_lower or "order count" in question_lower:
        results = query_data("SELECT COUNT(DISTINCT \"Invoice Number\") as total_orders FROM sales")
        if not results.empty:
            return f"**Total Orders:** {results.iloc[0]['total_orders']:,}"
    
    elif "units" in question_lower or "quantity" in question_lower:
        results = query_data("SELECT SUM(\"Quantity\") as total_units FROM sales")
        if not results.empty:
            return f"**Total Units Sold:** {results.iloc[0]['total_units']:,}"
    
    elif "top" in question_lower and ("sku" in question_lower or "product" in question_lower or "item" in question_lower or "best" in question_lower):
        limit = 5
        if "top 3" in question_lower: limit = 3
        elif "top 10" in question_lower: limit = 10
        
        # Check if ASIN column exists in DuckDB database
        table_info = query_data("DESCRIBE sales")
        columns = table_info['column_name'].tolist() if not table_info.empty else []
        
        if 'Asin' in columns:
            results = query_data(f"""
                SELECT "Sku", "Asin", SUM(revenue_calc) as revenue 
                FROM sales 
                WHERE "Sku" IS NOT NULL 
                GROUP BY "Sku", "Asin"
                ORDER BY revenue DESC 
                LIMIT {limit}
            """)
            
            if not results.empty:
                response = "**Top Products by Revenue:**\n"
                for i, row in results.iterrows():
                    sku = row['Sku']
                    asin = row['Asin']
                    revenue = row['revenue']
                    display_name = create_product_identifier(sku, asin)
                    response += f"{i+1}. {display_name}: {format_inr(revenue)}\n"
                return response
        else:
            # Legacy query without ASIN
            results = query_data(f"""
                SELECT "Sku", SUM(revenue_calc) as revenue 
                FROM sales 
                WHERE "Sku" IS NOT NULL 
                GROUP BY "Sku"
                ORDER BY revenue DESC 
                LIMIT {limit}
            """)
            
            if not results.empty:
                response = "**Top Products by Revenue:**\n"
                for i, (sku, revenue) in enumerate(results, 1):
                    display_name = str(sku) if pd.notna(sku) else "Unknown Product"
                    response += f"{i}. {display_name}: {format_inr(revenue)}\n"
                return response
    
    return "I couldn't understand your question. Try asking about revenue, orders, units, or top products."

def answer_descriptive_question(question: str) -> str:
    """Answer descriptive questions with general insights"""
    # Get basic data for descriptive questions using DuckDB
    if not table_exists('sales'): return "No data available for analysis."
    
    try:
        df = query_data(f"SELECT * FROM sales")
        
        if df.empty: return "No data available for analysis."
        
        # Basic metrics for descriptive answers
        total_revenue = df['revenue_in_inr'].sum()
        total_orders = df['Invoice Number'].nunique() if 'Invoice Number' in df.columns else 0
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

def initialize_ai_assistant():
    """Initialize AI Assistant if available and data exists"""
    if not AI_ASSISTANT_AVAILABLE:
        return False
    
    if st.session_state.ai_assistant is None and st.session_state.has_existing_data:
        try:
            with st.spinner("🤖 Initializing AI Assistant..."):
                st.session_state.ai_assistant = AIAssistant()
            
            # Check if initialization was successful
            if st.session_state.ai_assistant.is_ollama_available:
                st.success("✅ AI Assistant initialized successfully!")
                return True
            else:
                st.warning("⚠️ AI Assistant initialized with limited functionality")
                st.info("🔧 Ollama is not available. Some AI features will be disabled.")
                return True  # Still return True as it's initialized, just limited
                
        except ConnectionError as e:
            st.error("❌ **AI Assistant Connection Failed**")
            st.info("🔧 Cannot connect to Ollama service. Please ensure Ollama is running: `ollama serve`")
            st.session_state.ai_assistant = None
            return False
        except TimeoutError as e:
            st.error("❌ **AI Assistant Initialization Timeout**")
            st.info("⏱️ Ollama is taking too long to respond. Please check if it's running properly.")
            st.session_state.ai_assistant = None
            return False
        except Exception as e:
            # Log the error for debugging
            import logging
            logging.error(f"Unexpected error initializing AI Assistant: {e}")
            
            st.error("❌ **AI Assistant Initialization Failed**")
            st.info("🔧 I encountered an unexpected error. Please try again or check if Ollama is running.")
            
            # Show technical details in expander for debugging
            with st.expander("🔍 Technical Details (for debugging)", expanded=False):
                st.code(f"Error: {str(e)}", language='text')
            
            st.session_state.ai_assistant = None
            return False
    return st.session_state.ai_assistant is not None

def get_suggested_questions():
    """Get suggested questions for the AI chat"""
    return [
        "What is the total revenue?",
        "How many orders do we have?",
        "Show me top 5 products by revenue",
        "Which city has the highest revenue?",
        "What was the revenue last month?",
        "How many unique SKUs do we have?",
        "What is the average order value?",
        "Show me revenue by city",
        "What are the top 3 cities by sales?",
        "Compare September vs August performance"
    ]

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
    if 'uploaded_file_ids' not in st.session_state:
        st.session_state.uploaded_file_ids = set()  # Track processed file IDs
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []  # List of processed file metadata
    if 'total_rows_loaded' not in st.session_state:
        st.session_state.total_rows_loaded = 0
    if 'last_upload_time' not in st.session_state:
        st.session_state.last_upload_time = None
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
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = None
    if 'ai_chat_history' not in st.session_state:
        st.session_state.ai_chat_history = []
    
    # Header
    st.title("📊 CSV Analytics Dashboard")
    
    # Show existing data status
    if st.session_state.has_existing_data:
        data_summary = get_total_data_summary()
        
        # Beautiful, user-friendly data status
        if data_summary.get('total_rows', 0) > 0:
            file_count = len(data_summary.get('uploaded_files', []))
            
            # Calculate unique records for clean display
            unique_records = data_summary.get('unique_records', data_summary.get('total_rows', 0))
            
            # Show clean, positive message
            st.success(f"🎉 **Welcome back!** Your dashboard is ready with {unique_records:,} records from {file_count} datasets")
            
            # Show uploaded files in a clean way
            if data_summary.get('uploaded_files'):
                st.markdown("**📊 Your Data:**")
                for file_info in data_summary['uploaded_files'][:3]:  # Show first 3 files
                    # Clean up month display
                    month_display = file_info['month']
                    
                    # Extract clean month name
                    if 'July' in month_display:
                        month_display = 'July'
                    elif 'Sept' in month_display:
                        month_display = 'September'
                    elif 'Aug' in month_display:
                        month_display = 'August'
                    elif 'MTR' in month_display:
                        month_display = 'August (MTR)'
                    else:
                        # Fallback: clean up and capitalize
                        month_display = month_display.replace('_20251026_205319.csv', '').replace('_20251026_205329.csv', '').replace('_20251026_205351.csv', '').replace('_20251024_153740.csv', '')
                        month_display = month_display.replace('July_', 'July').replace('Sept_', 'September').replace('Aug_', 'August')
                        month_display = month_display.title()
                    
                    st.markdown(f"• **{month_display}** - {file_info['cleaned_rows']:,} records")
                if len(data_summary['uploaded_files']) > 3:
                    st.markdown(f"• **+{len(data_summary['uploaded_files']) - 3} more datasets**")
        else:
            st.warning("⚠️ **No data found** in database")
        
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
        # Beautiful welcome message for new users
        st.markdown("---")
        st.markdown("### 🚀 Welcome to Your Analytics Dashboard!")
        st.markdown("""
        **Get started in 3 simple steps:**
        1. 📁 **Upload your CSV files** using the sidebar
        2. 🔄 **Auto-mapping** will handle column detection
        3. 📊 **Explore insights** with interactive charts and KPIs
        
        *Your data will be automatically processed and ready for analysis!*
        """)
        st.markdown("---")
    
    # Data Management Section
    if st.session_state.show_data_management:
        st.markdown("---")
        st.markdown("### 📁 Data Sources Management")
        
        # Show processed files summary
        if st.session_state.processed_files:
            st.markdown("#### 📊 Recent Uploads")
            st.markdown(f"**Files Loaded**: {len(st.session_state.processed_files)}")
            st.markdown(f"**Total Rows**: {st.session_state.total_rows_loaded:,}")
            if st.session_state.last_upload_time:
                st.markdown(f"**Last Upload**: {st.session_state.last_upload_time}")
            
            # Show recent files
            recent_files = st.session_state.processed_files[-5:]  # Last 5 files
            for file_data in recent_files:
                status_icon = "✅" if file_data['status'] == 'success' else "❌"
                st.markdown(f"{status_icon} **{file_data['filename']}** - {file_data['rows']:,} rows ({file_data['month']})")
            
            st.markdown("---")
        
        # Get actual data summary from database
        data_summary = get_total_data_summary()
        
        if data_summary.get('total_rows', 0) > 0:
            file_count = len(data_summary.get('uploaded_files', []))
            st.markdown(f"**📊 Data Sources ({file_count} files):**")
            st.markdown(f"**Total Rows in Database**: {data_summary.get('total_rows', 0):,}")
            
            # Show duplicate analysis
            if data_summary.get('unique_records', 0) < data_summary.get('total_rows', 0):
                duplicate_count = data_summary.get('total_rows', 0) - data_summary.get('unique_records', 0)
                st.warning(f"⚠️ **Duplicates**: {duplicate_count:,} duplicate records found")
                st.markdown(f"**Unique Records**: {data_summary.get('unique_records', 0):,}")
            
            if data_summary.get('date_range_start') and data_summary.get('date_range_end'):
                st.markdown(f"**Date Range**: {data_summary['date_range_start']} to {data_summary['date_range_end']}")
            
            # Show detailed file information
            if data_summary.get('uploaded_files'):
                st.markdown("#### 📁 Uploaded Files")
                
                # Calculate totals
                total_file_rows = sum(f['cleaned_rows'] for f in data_summary['uploaded_files'])
                st.markdown(f"**Total Rows in Files**: {total_file_rows:,}")
                
                # Show each file
                for file_info in data_summary['uploaded_files']:
                    with st.expander(f"📄 {file_info['filename']} ({file_info['cleaned_rows']:,} rows)", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Month**: {file_info['month']}")
                            st.markdown(f"**Uploaded**: {file_info['upload_date']}")
                        with col2:
                            st.markdown(f"**Raw Rows**: {file_info['raw_rows']:,}")
                            st.markdown(f"**Cleaned Rows**: {file_info['cleaned_rows']:,}")
                        with col3:
                            st.markdown(f"**Raw Size**: {file_info['raw_size']:,} bytes")
                            st.markdown(f"**Cleaned Size**: {file_info['cleaned_size']:,} bytes")
            else:
                st.info("📊 No file information available. Data may have been loaded from previous sessions.")
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear All Data", type="secondary", help="Remove all data and start fresh"):
                    try:
                        # Clear database tables using DuckDB
                        clear_result = clear_database('sales')
                        if clear_result.get('success', False):
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
                            st.session_state.processed_files = []
                            st.session_state.uploaded_file_ids = set()
                            st.session_state.total_rows_loaded = 0
                            
                            st.success("✅ All data cleared! DuckDB database cleared.")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to clear data: {clear_result.get('error', 'Unknown error')}")
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
        
        uploaded_files = st.file_uploader(
            "📁 Upload Monthly CSV Files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files to analyze",
            key="csv_uploader"
        )
        
        # Debug buttons (only show if needed)
        if st.checkbox("🔧 Show Debug Options", help="Show debugging tools for troubleshooting"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Force Re-upload (Clear Session State)", help="Clear session state to allow re-uploading files"):
                    st.session_state.uploaded_file_ids = set()
                    st.session_state.processed_files = []
                    st.success("Session state cleared! You can now re-upload files.")
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Database", help="Clear the DuckDB database and start fresh"):
                    result = clear_database('sales')
                    if result.get('success', False):
                        st.session_state.uploaded_file_ids = set()
                        st.session_state.processed_files = []
                        st.success("Database cleared! You can now upload files.")
                        st.rerun()
                    else:
                        st.error(f"Failed to clear database: {result.get('error', 'Unknown error')}")
        
        if uploaded_files:
            # Get IDs of newly uploaded files
            current_file_ids = {file.file_id for file in uploaded_files}
            new_files = [f for f in uploaded_files if f.file_id not in st.session_state.uploaded_file_ids]
            
            if new_files:
                print(f"📁 NEW FILES DETECTED: {len(new_files)} new file(s) to process")
                st.info(f"Processing {len(new_files)} new file(s)...")
                
                successful_files = 0
                failed_files = 0
                total_rows_processed = 0
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(new_files):
                    try:
                        filename = file.name
                        print(f"📄 Processing new file {i+1}/{len(new_files)}: {filename}")
                        
                        # Update progress
                        progress = (i + 1) / len(new_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {filename}...")
                        
                        # Read CSV
                        df_raw = pd.read_csv(file)
                        print(f"📊 Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns from {filename}")
                        
                        # Auto-map columns (use first file as template or saved mapping)
                        if len(st.session_state.processed_files) == 0:  # First file determines mapping
                            saved_mapping = load_mapping(list(df_raw.columns))
                            if saved_mapping:
                                mappings = saved_mapping
                                st.session_state.mappings = mappings
                                print(f"✅ Using saved column mapping for {filename}")
                            else:
                                mappings = auto_map_columns(df_raw)
                                st.session_state.mappings = mappings
                                print(f"✅ Auto-mapped columns for {filename}")
                        else:
                            # Use same mapping for subsequent files
                            mappings = st.session_state.mappings
                            print(f"✅ Using existing mapping for {filename}")
                        
                        # Auto-clean data
                        print(f"🧹 Auto-cleaning {filename}...")
                        df_cleaned, cleaning_report = clean_dataframe_transaction_aware(df_raw, mappings)
                        print(f"✅ Cleaned {filename}: {len(df_cleaned)} rows")
                        
                        # Ensure DataFrame has consistent columns for database storage
                        # This prevents column mismatch errors when appending
                        expected_columns = [
                            'Invoice Date', 'Invoice Number', 'Sku', 'Asin', 'Item Description',
                            'Quantity', 'Invoice Amount', 'Transaction Type', 'Ship To City',
                            'transaction_type', 'revenue_calc', 'shipping_loss_calc', 'units_sold_calc',
                            'needs_estimation', 'month_tag'
                        ]
                        
                        # Add missing columns with default values
                        for col in expected_columns:
                            if col not in df_cleaned.columns:
                                if col in ['revenue_calc', 'shipping_loss_calc']:
                                    df_cleaned[col] = 0.0
                                elif col in ['units_sold_calc']:
                                    df_cleaned[col] = 0
                                elif col in ['needs_estimation']:
                                    df_cleaned[col] = False
                            else:
                                    df_cleaned[col] = None
                        
                        # Remove extra columns that aren't in expected schema
                        df_cleaned = df_cleaned[expected_columns]
                        
                        print(f"✅ Aligned columns for {filename}: {len(df_cleaned.columns)} columns")
                        
                        # Store to database (APPEND mode for subsequent files)
                        mode = "append" if upload_mode == "Append to existing data" or len(st.session_state.processed_files) > 0 else "replace"
                        print(f"💾 Storing {filename} in {mode} mode...")
                        
                        # Save raw and cleaned data
                        try:
                            raw_path, cleaned_path = save_raw_and_cleaned_data(df_raw, df_cleaned, filename)
                            print(f"✅ Saved {filename} to: {cleaned_path}")
                        except Exception as e:
                            print(f"⚠️ Warning saving {filename}: {e}")
                        use_append = (mode == "append")
                        
                        # Check if table exists and get its schema
                        if table_exists('sales'):
                            table_info = query_data("DESCRIBE sales")
                            existing_columns = table_info['column_name'].tolist() if not table_info.empty else []
                            
                            # If columns don't match, use replace mode instead of append
                            if len(df_cleaned.columns) != len(existing_columns):
                                print(f"⚠️ Column mismatch detected! Using REPLACE mode instead of APPEND")
                                use_append = False
                                mode = "replace"
                        
                        result = store_data(df_cleaned, 'sales', append=use_append)
                        print(f"📊 DuckDB storage result for {filename}: {result}")
                        
                        if result.get('success', False):
                            successful_files += 1
                            rows_stored = result.get('rows_stored', len(df_cleaned))
                            total_rows_processed += rows_stored
                            
                            # Track processed file
                            st.session_state.uploaded_file_ids.add(file.file_id)
                            file_metadata = {
                                'filename': filename,
                                'rows': rows_stored,
                                'month': df_cleaned['month_tag'].iloc[0] if 'month_tag' in df_cleaned.columns and len(df_cleaned) > 0 else 'Unknown',
                                'status': 'success',
                                'timestamp': current_time
                            }
                            st.session_state.processed_files.append(file_metadata)
                            
                            st.success(f"✅ {filename}: {len(df_cleaned)} cleaned → {rows_stored} stored")
                            print(f"✅ Successfully processed {filename}: {rows_stored} rows stored")
                        else:
                            failed_files += 1
                            error_msg = result.get('error', 'Unknown DuckDB error')
                            print(f"❌ Failed to process {filename}: {error_msg}")
                            
                            # Track failed file
                            st.session_state.uploaded_file_ids.add(file.file_id)
                            file_metadata = {
                                'filename': filename,
                                'rows': 0,
                                'month': 'Failed',
                                'status': 'failed',
                                'timestamp': current_time,
                                'error': error_msg
                            }
                            st.session_state.processed_files.append(file_metadata)
                        
                    except Exception as e:
                        failed_files += 1
                        error_msg = f"Error processing {file.name}: {e}"
                        st.error(error_msg)
                        print(f"❌ {error_msg}")
                        
                        # Track failed file
                        st.session_state.uploaded_file_ids.add(file.file_id)
                        file_metadata = {
                            'filename': file.name,
                            'rows': 0,
                            'month': 'Failed',
                            'status': 'failed',
                            'timestamp': current_time,
                            'error': str(e)
                        }
                        st.session_state.processed_files.append(file_metadata)
                
                # Clear progress bar
                progress_bar.empty()
                status_text.empty()
                
                # Update session state
                st.session_state.total_rows_loaded += total_rows_processed
                st.session_state.last_upload_time = current_time
                st.session_state.has_existing_data = True
                
                # Clear AI Assistant caches when new data is uploaded
                if st.session_state.ai_assistant:
                    st.session_state.ai_assistant.clear_all_caches()
                    st.info("🧹 **Caches cleared** - AI Assistant will learn from new data")
                
                # Show results
                if successful_files > 0:
                    st.success(f"✅ **Processed {successful_files}/{len(new_files)} files successfully!**")
                    st.success(f"📊 **Total rows loaded**: {total_rows_processed:,}")
                    
                    # Verify total data in database using DuckDB
                    try:
                        total_db_rows = get_row_count('sales')
                        st.info(f"🗄️ **Total rows in database**: {total_db_rows:,}")
                        print(f"✅ PROCESSING COMPLETE: {successful_files}/{len(new_files)} files, {total_rows_processed} rows")
                    except Exception as e:
                        st.warning(f"⚠️ Could not verify database count: {e}")
                        print(f"⚠️ Database verification failed: {e}")
                
                if failed_files > 0:
                    st.warning(f"⚠️ **{failed_files} files failed to process**")
                
                # Only rerun if we successfully processed at least one file
                if successful_files > 0:
                    st.rerun()
                else:
                    st.error("❌ No files were successfully processed. Please check the error messages above.")
            else:
                st.info("All uploaded files have already been processed.")
        
        # Display processed files summary
        if st.session_state.processed_files:
            st.markdown("### 📊 Loaded Files")
            for pf in st.session_state.processed_files:
                status_icon = "✅" if pf['status'] == 'success' else "❌"
                st.text(f"{status_icon} {pf['filename']} ({pf['rows']} rows)")
    
    # Query History & Performance Tracking
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Query History & Performance")
    
    try:
        # Import the logging functions
        from ai_assistant import get_query_history, calculate_performance_stats
        
        # Get performance statistics
        perf_stats = calculate_performance_stats()
        
        # Display performance metrics
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.sidebar.metric("Total Queries", perf_stats['total_queries'])
            st.sidebar.metric("Success Rate", f"{perf_stats['success_rate']}%")
        with col2:
            st.sidebar.metric("Avg Response Time", f"{perf_stats['avg_response_time']}s")
            st.sidebar.metric("Fastest Query", f"{perf_stats['fastest_query_time']}s")
        
        # Show recent query history
        query_history = get_query_history(limit=10)
        if not query_history.empty:
            st.sidebar.markdown("**Recent Queries:**")
            # Display only user-friendly columns
            display_cols = ['timestamp', 'question', 'total_time_seconds', 'query_success', 'rows_returned']
            if all(col in query_history.columns for col in display_cols):
                # Format the display
                display_df = query_history[display_cols].copy()
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M')
                display_df['question'] = display_df['question'].str[:30] + '...'
                display_df['total_time_seconds'] = display_df['total_time_seconds'].round(2)
                display_df['query_success'] = display_df['query_success'].map({True: '✅', False: '❌'})
                display_df['rows_returned'] = display_df['rows_returned'].astype(str)
                
                # Rename columns for display
                display_df.columns = ['Time', 'Question', 'Time(s)', 'Success', 'Rows']
                
                st.sidebar.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export button
        if st.sidebar.button("📥 Download Complete Log", help="Download full query log as CSV"):
            try:
                full_history = get_query_history(limit=1000)  # Get all history
                if not full_history.empty:
                    csv_data = full_history.to_csv(index=False)
                    st.sidebar.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"ai_query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.sidebar.warning("No query history available")
            except Exception as e:
                st.sidebar.error(f"Error downloading log: {e}")
                
    except Exception as e:
        st.sidebar.error(f"Error loading query history: {e}")

    # Main content area
    if st.session_state.uploaded_df is not None or st.session_state.has_existing_data or st.session_state.processed_files:
        # Show mapping modal if needed
        if st.session_state.show_mapping_modal:
            st.markdown("---")
            st.markdown("### 🤔 Confirm Column Mapping")
            st.markdown("*Please select the correct column for each field:*")
            
            # Get the first uploaded file for mapping
            if st.session_state.uploaded_files:
                first_file = list(st.session_state.uploaded_files.keys())[0]
                df = st.session_state.uploaded_files[first_file]['dataframe']
            else:
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
            
            # Date, Month, and Transaction Type filters
            st.markdown("### 📅 Filters")
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                preset = st.selectbox(
                    "Quick Presets:",
                    ["Custom Range", "Last 7 days", "Last 30 days", "This Month", "Last Month"],
                    key="date_preset"
                )
            
            with col2:
                # Check if transaction_type column exists in database using DuckDB
                if table_exists('sales'):
                    try:
                        # Get table schema from DuckDB
                        table_info = query_data("DESCRIBE sales")
                        columns = table_info['column_name'].tolist() if not table_info.empty else []
                        
                        if 'transaction_type' in columns or 'Transaction Type' in columns:
                            transaction_type = st.selectbox(
                                "Transaction Type:",
                                ["All", "Shipment", "Cancel", "Refund", "FreeReplacement"],
                                help="Filter by transaction type for analysis"
                            )
                        else:
                            transaction_type = "All"
                            st.info("ℹ️ Transaction type filtering not available for legacy data")
                    except Exception as e:
                        print(f"Error checking transaction type column: {e}")
                        transaction_type = "All"
                        st.info("ℹ️ Transaction type filtering not available")
                else:
                    transaction_type = "All"
            
            with col3:
                if preset == "Custom Range":
                    # Check if data exists and get its date range using DuckDB
                    if table_exists('sales'):
                        try:
                            date_range_query = query_data('SELECT MIN("Invoice Date") as min_date, MAX("Invoice Date") as max_date FROM sales WHERE "Invoice Date" IS NOT NULL')
                            
                            if not date_range_query.empty:
                                min_date = date_range_query.iloc[0]['min_date']
                                max_date = date_range_query.iloc[0]['max_date']
                                
                                if min_date and max_date:
                                    default_start = datetime.strptime(str(min_date), '%Y-%m-%d').date()
                                    default_end = datetime.strptime(str(max_date), '%Y-%m-%d').date()
                                start_date = st.date_input("Start Date", value=default_start)
                                end_date = st.date_input("End Date", value=default_end)
                            else:
                                start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                                end_date = st.date_input("End Date", value=datetime.now().date())
                        except Exception as e:
                            print(f"Error getting date range: {e}")
                            start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                            end_date = st.date_input("End Date", value=datetime.now().date())
                    else:
                        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                        end_date = st.date_input("End Date", value=datetime.now().date())
                else:
                    # For presets, use the actual data range from database using DuckDB
                    if table_exists('sales'):
                        try:
                            date_range_query = query_data('SELECT MIN("Invoice Date") as min_date, MAX("Invoice Date") as max_date FROM sales WHERE "Invoice Date" IS NOT NULL')
                            
                            if not date_range_query.empty:
                                min_date = date_range_query.iloc[0]['min_date']
                                max_date = date_range_query.iloc[0]['max_date']
                            
                            if min_date and max_date:
                                # Use the full data range
                                start_date = datetime.strptime(str(min_date), '%Y-%m-%d').date()
                                end_date = datetime.strptime(str(max_date), '%Y-%m-%d').date()
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
                pass  # Range display removed - redundant with Start/End Date fields
            
            # Get filtered data for dashboard (no month filtering needed - handled automatically above)
            df = get_date_filtered_data(str(start_date), str(end_date), transaction_type)
            
            # Month Analysis - Original Layout with Accurate Data
            st.markdown("### 📊 Month Analysis")
            
            # Transaction Type Filter (Simple)
            transaction_type_options = [
                "Revenue (Shipments)",
                "Refunds", 
                "Free Replacements",
                "All Transactions"
            ]
            selected_transaction_type = st.selectbox(
                "Select Transaction Type:",
                options=transaction_type_options,
                index=0,  # Default to Revenue (Shipments)
                help="Choose what type of transactions to analyze"
            )
            
            # Get months automatically from the selected date range
            months_in_range = get_months_in_date_range(str(start_date), str(end_date))
            
            if not months_in_range:
                st.info("📅 **No data found** in the selected date range. Try adjusting your date filters above.")
            elif len(months_in_range) == 1:
                # Single month - show detailed analysis
                month = months_in_range[0]
                metrics = get_month_metrics_by_transaction_type(
                    str(start_date), str(end_date), month['month_tag'], selected_transaction_type
                )
                
                st.markdown(f"#### 📈 **{month['month_display']}** - Detailed Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    amount_key = 'total_amount' if 'total_amount' in metrics else 'total_revenue'
                    st.metric(
                        label=f"💰 {selected_transaction_type}",
                        value=format_inr(metrics[amount_key]),
                        help=f"Total {selected_transaction_type.lower()} for this month"
                    )
                with col2:
                    st.metric(
                        label="📦 Orders",
                        value=f"{metrics['order_count']:,}",
                        help="Total number of orders"
                    )
                with col3:
                    st.metric(
                        label="🏷️ SKUs",
                        value=f"{metrics['sku_count']:,}",
                        help="Number of unique products"
                    )
                with col4:
                    st.metric(
                        label="📊 Units",
                        value=f"{metrics['total_units']:,}",
                        help="Total units"
                    )
                
                # Show additional insights
                st.markdown("#### 📋 **Month Summary**")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📅 **Period:** {month['month_display']}")
                    st.info(f"📊 **Records:** {metrics['total_records']:,} transactions")
                with col2:
                    avg_order_value = metrics[amount_key] / metrics['order_count'] if metrics['order_count'] > 0 else 0
                    st.info(f"💵 **Avg Order Value:** {format_inr(avg_order_value)}")
                    st.info(f"📈 **Amount per SKU:** {format_inr(metrics[amount_key] / metrics['sku_count']) if metrics['sku_count'] > 0 else 'N/A'}")
                
                # Show free replacement breakdown if this is the selected transaction type
                if selected_transaction_type == 'Free Replacements':
                    st.markdown("---")
                    display_free_replacement_breakdown(metrics)
                    
            else:
                # Multiple months - show card-based comparison (ORIGINAL LAYOUT)
                st.markdown(f"#### 📊 **Multi-Month Analysis** ({len(months_in_range)} months)")
                
                # Calculate metrics for all months with selected transaction type
                month_data = []
                for month in months_in_range:
                    metrics = get_month_metrics_by_transaction_type(
                        str(start_date), str(end_date), month['month_tag'], selected_transaction_type
                    )
                    # Only include months that have data for the selected transaction type
                    if metrics['total_records'] > 0:
                        month_data.append({
                            'month': month,
                            'metrics': metrics
                        })
                
                if not month_data:
                    st.warning(f"⚠️ **No {selected_transaction_type.lower()} data found** in the selected date range.")
                    st.info("💡 Try selecting a different transaction type or adjusting your date range.")
                else:
                    # Calculate MoM changes
                    for i in range(len(month_data)):
                        if i > 0:
                            changes = calculate_mom_change(
                                month_data[i]['metrics'], 
                                month_data[i-1]['metrics']
                            )
                            month_data[i]['changes'] = changes
                        else:
                            month_data[i]['changes'] = None
                    
                    # Create responsive grid layout (ORIGINAL)
                    if len(month_data) <= 3:
                        cols = st.columns(len(month_data))
                    elif len(month_data) <= 6:
                        cols = st.columns(3)
                    else:
                        cols = st.columns(4)
                    
                    # Display month cards (ORIGINAL DESIGN WITH SMALLER CARDS)
                    for i, data in enumerate(month_data):
                        month = data['month']
                        metrics = data['metrics']
                        changes = data['changes']
                        
                        col_idx = i % len(cols)
                        
                        with cols[col_idx]:
                            # Create smaller modern card with shadow effect
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                padding: 15px;
                                border-radius: 12px;
                                box-shadow: 0 6px 20px rgba(0,0,0,0.1);
                                margin-bottom: 15px;
                                color: white;
                            ">
                                <h3 style="margin: 0 0 10px 0; font-size: 1.1em;">{month['month_display']}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Amount metric with trend (UPDATED WITH ACCURATE DATA)
                            amount_key = 'total_amount' if 'total_amount' in metrics else 'total_revenue'
                            amount_col, trend_col = st.columns([3, 1])
                            with amount_col:
                                st.metric(
                                    label=f"💰 {selected_transaction_type}",
                                    value=format_inr(metrics[amount_key]),
                                    delta=changes[amount_key]['formatted'] if changes and amount_key in changes else None,
                                    help=f"Total {selected_transaction_type.lower()} for this month"
                                )
                            with trend_col:
                                if changes and amount_key in changes and changes[amount_key]['direction'] == 'up':
                                    st.markdown("📈")
                                elif changes and amount_key in changes and changes[amount_key]['direction'] == 'down':
                                    st.markdown("📉")
                                else:
                                    st.markdown("➡️")
                            
                            # Other metrics (smaller)
                            st.metric(
                                label="📦 Orders",
                                value=f"{metrics['order_count']:,}",
                                delta=changes['order_count']['formatted'] if changes else None
                            )
                            
                            st.metric(
                                label="🏷️ SKUs",
                                value=f"{metrics['sku_count']:,}",
                                delta=changes['sku_count']['formatted'] if changes else None
                            )
                            
                            st.metric(
                                label="📊 Units",
                                value=f"{metrics['total_units']:,}",
                                delta=changes['total_units']['formatted'] if changes else None
                            )
                    
                    # Overall period summary
                    st.markdown("---")
                    st.markdown("#### 📈 **Period Overview**")
                    
                    total_amount = sum(data['metrics'].get('total_amount', data['metrics'].get('total_revenue', 0)) for data in month_data)
                    total_orders = sum(data['metrics']['order_count'] for data in month_data)
                    total_skus = sum(data['metrics']['sku_count'] for data in month_data)
                    total_units = sum(data['metrics']['total_units'] for data in month_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(f"💰 Total {selected_transaction_type}", format_inr(total_amount))
                    with col2:
                        st.metric("📦 Total Orders", f"{total_orders:,}")
                    with col3:
                        st.metric("🏷️ Total SKUs", f"{total_skus:,}")
                    with col4:
                        st.metric("📊 Total Units", f"{total_units:,}")
                    
                    # Best and worst performing months
                    if len(month_data) > 1:
                        best_month = max(month_data, key=lambda x: x['metrics'].get('total_amount', x['metrics'].get('total_revenue', 0)))
                        worst_month = min(month_data, key=lambda x: x['metrics'].get('total_amount', x['metrics'].get('total_revenue', 0)))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            best_amount = best_month['metrics'].get('total_amount', best_month['metrics'].get('total_revenue', 0))
                            st.success(f"🏆 **Best Month:** {best_month['month']['month_display']} ({format_inr(best_amount)})")
                        with col2:
                            worst_amount = worst_month['metrics'].get('total_amount', worst_month['metrics'].get('total_revenue', 0))
                            st.warning(f"📉 **Needs Attention:** {worst_month['month']['month_display']} ({format_inr(worst_amount)})")
                    
                    # Show free replacement breakdown if this is the selected transaction type
                    if selected_transaction_type == 'Free Replacements':
                        st.markdown("---")
                        # Calculate overall free replacement data for the period
                        overall_free_replacement_data = calculate_free_replacement_cost(str(start_date), str(end_date))
                        if overall_free_replacement_data['transaction_count'] > 0:
                            st.markdown("#### 🎁 **Period Free Replacement Analysis**")
                            display_free_replacement_breakdown({'free_replacement_breakdown': overall_free_replacement_data})
            
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
                    # Use DuckDB clear_database function
                    result = clear_database('sales')
                    if result.get('success', False):
                        st.success("✅ Database cleared! Please upload new data.")
                        # Reset session state
                        st.session_state.processed_files = []
                        st.session_state.uploaded_file_ids = set()
                        st.session_state.has_existing_data = False
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to clear database: {result.get('error', 'Unknown error')}")
                return
            elif df.empty:
                st.warning("⚠️ No data found for the selected date range and filters.")
                st.info(f"💡 Date range: {start_date} to {end_date}, Transaction type: {transaction_type}")
                
                # Show available date range using DuckDB
                try:
                    date_range_query = query_data('SELECT MIN("Invoice Date") as min_date, MAX("Invoice Date") as max_date, COUNT(*) as total_rows FROM sales')
                    if not date_range_query.empty:
                        min_date = date_range_query.iloc[0]['min_date']
                        max_date = date_range_query.iloc[0]['max_date']
                        total_rows = date_range_query.iloc[0]['total_rows']
                    
                    st.info(f"""
                    **Data available in database:**
                        - Date range: {min_date} to {max_date}
                        - Total rows: {total_rows}
                    
                    **Adjust your date filter to match the available data range.**
                    """)
                except:
                    pass
                return
            else:
                pass  # Data loaded message already shown above
            
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
                        labels={'revenue': 'Revenue (₹)', 'date': 'Date'},
                        markers=True,  # Add data point markers
                        line_shape='spline'  # Smooth line
                    )
                    fig.update_layout(
                        height=400,
                        hovermode='x unified',
                        xaxis_title="Date",
                        yaxis_title="Revenue (₹)",
                        showlegend=False
                    )
                    # Add hover template for better data display
                    fig.update_traces(
                        hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trend summary
                    if len(kpis['revenue_trend']) > 1:
                        total_revenue = kpis['revenue_trend']['revenue'].sum()
                        avg_daily = kpis['revenue_trend']['revenue'].mean()
                        max_daily = kpis['revenue_trend']['revenue'].max()
                        min_daily = kpis['revenue_trend']['revenue'].min()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Revenue", format_inr(total_revenue))
                        with col2:
                            st.metric("Avg Daily", format_inr(avg_daily))
                        with col3:
                            st.metric("Peak Day", format_inr(max_daily))
                        with col4:
                            st.metric("Lowest Day", format_inr(min_daily))
                else:
                    st.info("No revenue trend data available - try adjusting your date range")
                
                # Revenue by Region and Top Products (side by side)
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 🌍 Revenue by Region")
                    if kpis['region_revenue'] is not None and not kpis['region_revenue'].empty:
                        # Create a more detailed table view for better display
                        region_df = pd.DataFrame({
                            'Region': kpis['region_revenue'].index,
                            'Revenue': kpis['region_revenue'].values
                        })
                        region_df['Revenue (₹)'] = region_df['Revenue'].apply(format_inr)
                        region_df = region_df[['Region', 'Revenue (₹)']]
                        
                        # Display as table for better readability
                        st.dataframe(region_df, use_container_width=True, hide_index=True)
                        
                        # Export button
                        csv = region_df.to_csv(index=False)
                        st.download_button(
                            "📥 Export CSV",
                            csv,
                            "revenue_by_region.csv",
                            "text/csv",
                            key="export_region"
                        )
                        
                        # Also show the chart below the table
                        # Create properly sorted data for chart (highest revenue at top)
                        chart_data = pd.DataFrame({
                            'Region': kpis['region_revenue'].index,
                            'Revenue': kpis['region_revenue'].values
                        }).sort_values('Revenue', ascending=True)  # Sort ascending for horizontal bars (highest at top)
                        
                        # Create Indian currency formatted labels
                        def format_chart_amount(amount):
                            """Format amount for chart display in Indian currency"""
                            if amount >= 10000000:  # 1 crore = 10 million
                                crores = amount / 10000000
                                if crores >= 100:
                                    return f"₹{crores:.1f} Cr"
                                else:
                                    return f"₹{crores:.2f} Cr"
                            elif amount >= 100000:  # 1 lakh = 100 thousand
                                lakhs = amount / 100000
                                return f"₹{lakhs:.2f} L"
                            elif amount >= 1000:  # Thousands
                                thousands = amount / 1000
                                return f"₹{thousands:.1f}K"
                            else:
                                return f"₹{amount:.0f}"
                        
                        # Create the chart with proper Y-axis labels
                        fig = px.bar(
                            x=chart_data['Revenue'],
                            y=chart_data['Region'],
                            orientation='h',
                            title="Revenue by Region Chart",
                            labels={'x': 'Revenue (₹)', 'y': 'Region'},
                            text=[format_chart_amount(rev) for rev in chart_data['Revenue']]
                        )
                        
                        # Fix chart styling to ensure Y-axis labels are visible and complete full width usage
                        fig.update_layout(
                            height=max(400, len(chart_data) * 35),  # Dynamic height based on number of regions
                            margin=dict(l=120, r=0, t=50, b=50),  # Zero right margin for complete width usage
                            xaxis=dict(
                                title="Revenue (₹)",
                                showgrid=False,  # Remove vertical grid lines
                                tickformat='₹.2s',
                                automargin=True  # Auto-adjust margins
                            ),
                            yaxis=dict(
                                title="Region",
                                showgrid=False,
                                tickfont=dict(size=12, color='white'),
                                automargin=True  # Auto-adjust margins
                            ),
                            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                            font=dict(size=12, color='white'),
                            showlegend=False,
                            width=None,  # Use full container width
                            autosize=True,  # Enable auto-sizing
                            bargap=0.1  # Reduce gap between bars for better space usage
                        )
                        
                        # Update bar colors and text positioning
                        fig.update_traces(
                            marker_color='#1f77b4',  # Consistent blue color
                            textposition='outside',
                            textfont=dict(size=10, color='white'),
                            hovertemplate='<b>%{y}</b><br>Revenue: %{text}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("No region data available")
                
                with col2:
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
                
                # Top Products by Units (full width below)
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
                
                # Movers & Decliners
                st.markdown("---")
                st.markdown("### 📊 Movers & Decliners")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📉 Decliners (≥30% WoW Drop)")
                    if not movers['decliners'].empty:
                        display_df = movers['decliners'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_calc_current'].apply(format_inr)
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
                        st.info("No significant decliners found this period")
                
                with col2:
                    st.markdown("#### 📈 Fast Movers (≥30% WoW Growth)")
                    if not movers['fast_movers'].empty:
                        display_df = movers['fast_movers'].copy()
                        display_df['Revenue (₹)'] = display_df['revenue_calc_current'].apply(format_inr)
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
                        st.info("No significant fast movers found this period")
                
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
            st.header("🤖 AI Chat Assistant")
            st.markdown("*Ask questions about your business data in natural language*")
            
            # Initialize AI Assistant
            ai_available = initialize_ai_assistant()
            
            if not ai_available:
                st.warning("⚠️ **AI Assistant not available**")
                st.info("""
                **To enable AI chat:**
                1. Make sure Ollama is running: `ollama serve`
                2. Ensure llama3.1:8b model is installed: `ollama pull llama3.1:8b`
                3. Refresh this page
                
                **Fallback:** Using basic chat functionality below.
                """)
                
                # Fallback to basic chat
                st.markdown("---")
                st.markdown("### 💬 Basic Chat (Fallback)")
                
                # Chat history
                if st.session_state.chat_history:
                    st.markdown("#### 💬 Chat History")
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
                # AI-powered chat interface
                st.success("✅ **AI Assistant Ready!** Ask any question about your data.")
                
                # Chat history with improved, centered layout
                if st.session_state.ai_chat_history:
                    st.markdown("### 💬 Chat History")
                    
                    # Show chat history in reverse order (newest first)
                    chat_items = st.session_state.ai_chat_history[-5:]  # Show last 5
                    chat_items.reverse()  # Show newest first
                    
                    for i, chat_item in enumerate(chat_items):
                        # Make the first (most recent) item expanded by default
                        is_expanded = (i == 0)
                        
                        with st.expander(f"Q{len(st.session_state.ai_chat_history) - i}: {chat_item['question'][:50]}{'...' if len(chat_item['question']) > 50 else ''}", expanded=is_expanded):
                            st.markdown(f"**Question:** {chat_item['question']}")
                            
                            # Display answer with simple formatting
                            st.markdown("**Answer:**")
                            st.markdown(chat_item['answer'])
                            
                            if chat_item.get('sql'):
                                with st.expander("🔍 View SQL Query", expanded=False):
                                    st.code(chat_item['sql'], language='sql')
                    
                    st.markdown("---")
                
                # Question input with better width
                st.markdown("---")
                st.markdown("### 💬 Ask AI Assistant")
                
                # Use wider columns for question input
                col1, col2, col3 = st.columns([1, 4, 1])
                with col2:
                    question = st.text_area(
                        "Ask a question about your data:",
                        placeholder="e.g., What's the total revenue? Show me top products by city...",
                        key="ai_chat_question",
                        height=100,
                        help="Ask any question about your business data in natural language"
                    )
                
                # Center the buttons
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1:
                    if st.button("🤖 Ask AI", type="primary", use_container_width=True):
                        if question.strip():
                            # Create progress indicators for different stages
                            progress_container = st.container()
                            
                            with progress_container:
                                st.info("🤖 **AI Assistant Status**")
                                status_col1, status_col2, status_col3 = st.columns(3)
                                
                                with status_col1:
                                    st.markdown("🔍 **Analyzing Question**")
                                with status_col2:
                                    st.markdown("📊 **Generating SQL**")
                                with status_col3:
                                    st.markdown("🧠 **Creating Analysis**")
                            
                            with st.spinner("🤖 AI is thinking..."):
                                try:
                                    # Call AI assistant (timeout handling is now built into the AI assistant)
                                    result = st.session_state.ai_assistant.ask_question(question)
                                    
                                    if result['success']:
                                        # Add to AI chat history
                                        st.session_state.ai_chat_history.append({
                                            'question': question,
                                            'answer': result['response'],
                                            'sql': result.get('sql', ''),
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'offline_mode': result.get('offline_mode', False)
                                        })
                                        
                                        # Display the response in full width with simple formatting
                                        st.markdown("---")
                                        st.markdown("### 🤖 AI Response")
                                        
                                        # Use simple, readable formatting that matches the page design
                                        st.markdown(result['response'])
                                        
                                        # Add navigation buttons
                                        col1, col2, col3 = st.columns([1, 1, 1])
                                        with col1:
                                            if st.button("⬆️ Back to Top", help="Scroll to the top of the chat"):
                                                st.markdown("---")
                                                st.markdown("### 💬 Chat History")
                                        
                                        with col2:
                                            if st.button("💬 Ask Another Question", type="secondary", help="Ask a follow-up question"):
                                                st.markdown("---")
                                                st.markdown("### 💬 Ask AI Assistant")
                                        
                                        # Question input will be cleared by user manually
                                        
                                        # Show SQL query in expander
                                        if result.get('sql'):
                                            with st.expander("🔍 View Generated SQL Query", expanded=False):
                                                st.code(result['sql'], language='sql')
                                        
                                        # Show offline mode warning if applicable
                                        if result.get('offline_mode'):
                                            st.warning("⚠️ AI features are limited. Ollama is not available.")
                                        
                                        # Show cache indicator if applicable
                                        if result.get('cached'):
                                            if result.get('similarity'):
                                                st.info(f"📋 **Similar Cached Result** - Similarity: {result['similarity']:.1%}")
                                            else:
                                                st.info("📋 **Cached Result** - This question was answered before")
                                        
                                        # Show response time
                                        if result.get('response_time'):
                                            st.caption(f"⏱️ Response time: {result['response_time']:.2f} seconds")
                                        
                                        # Don't rerun - let the chat interface update naturally
                                    else:
                                        # Handle different error types with specific messages
                                        error_type = result.get('error_type', 'unknown')
                                        
                                        if error_type == 'connection_error':
                                            st.error("❌ **AI Assistant Unavailable**")
                                            st.info("🔧 Please ensure Ollama is running: `ollama serve`")
                                        elif error_type == 'timeout_error':
                                            st.error("❌ **Query Timeout**")
                                            st.info("⏱️ This query is taking too long. Try a simpler question.")
                                        elif error_type == 'sql_error' or error_type == 'sql_generation_failed':
                                            st.error("❌ **Question Not Understood**")
                                            st.info("💡 I couldn't understand that question. Could you rephrase it?")
                                        elif error_type == 'database_query_failed':
                                            st.error("❌ **Data Retrieval Failed**")
                                            st.info("📊 Unable to retrieve data. Please try again.")
                                        elif error_type == 'offline_mode':
                                            st.error("❌ **AI Assistant Offline**")
                                            st.info("🔧 AI assistant is currently unavailable. Please ensure Ollama is running.")
                                        else:
                                            st.error(f"❌ **Error:** {result['error']}")
                                            st.info("💡 Try rephrasing your question or check if Ollama is running.")
                                
                                except Exception as e:
                                    # Log the error for debugging
                                    import logging
                                    logging.error(f"Unexpected error in chat: {e}")
                                    
                                    st.error("❌ **Unexpected Error**")
                                    st.info("🔧 I encountered an unexpected error. Please try again or check if Ollama is running.")
                                    
                                    # Show technical details in expander for debugging
                                    with st.expander("🔍 Technical Details (for debugging)", expanded=False):
                                        st.code(f"Error: {str(e)}", language='text')
                        else:
                            st.warning("Please enter a question.")
                
                with col2:
                    if st.button("🗑️ Clear", help="Clear the question input", use_container_width=True):
                        st.rerun()
                
                with col3:
                    if st.button("💬 New Chat", help="Start a new conversation", use_container_width=True):
                        st.session_state.ai_chat_history = []
                        st.rerun()
                
                with col4:
                    st.caption("💡 **Tip:** Ask specific questions like 'What's my total revenue?' or 'Show me top 5 products'")
                
                # Show AI status
                st.markdown("---")
                st.markdown("### 🤖 AI Assistant Status")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success("✅ **Connected** to Ollama")
                with col2:
                    st.info(f"🤖 **Model:** llama3.1:8b")
                with col3:
                    st.info(f"💬 **Chat History:** {len(st.session_state.ai_chat_history)} questions")
                
                # Show cache statistics
                if st.session_state.ai_assistant:
                    cache_stats = st.session_state.ai_assistant.get_cache_stats()
                    st.markdown("### 📊 Performance Stats")
                    st.info(f"🚀 **Cache Hit Rate:** {cache_stats['cache_hit_rate']}")
                    st.info(f"💾 **Question Cache:** {cache_stats['question_cache_size']} items")
                    st.info(f"🔍 **SQL Cache:** {cache_stats['sql_cache_size']} items")
                    st.info(f"🧠 **Analysis Cache:** {cache_stats['analysis_cache_size']} items")
                    
                    # Cache management buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Clear SQL Cache", help="Clear cached SQL queries (useful when rules change)"):
                            st.session_state.ai_assistant.clear_sql_cache()
                            st.success("SQL cache cleared!")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Clear All Caches", help="Clear all caches"):
                            st.session_state.ai_assistant.clear_all_caches()
                            st.success("All caches cleared!")
                            st.rerun()
                    
                    # Show performance stats
                    perf_stats = cache_stats['performance_stats']
                    if perf_stats['total_questions'] > 0:
                        avg_time = perf_stats.get('avg_response_time', 0)
                        st.info(f"⏱️ **Avg Response Time:** {avg_time:.2f}s")
                        st.info(f"📈 **Total Questions:** {perf_stats['total_questions']}")
                        
                        # Show slow queries if any
                        if perf_stats['slow_queries']:
                            with st.expander("🐌 Slow Queries", expanded=False):
                                for query in perf_stats['slow_queries'][-5:]:  # Show last 5
                                    st.text(f"{query['time']:.2f}s - {query['question'][:50]}...")
    
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