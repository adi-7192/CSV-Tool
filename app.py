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
    "sku": ["sku", "asin", "product id", "item id"],
    "product_name": ["product name", "item description", "title", "product"],
    "quantity": ["quantity", "qty", "units"],
    "revenue_amount": ["invoice amount", "order amount", "total amount", "amount", "revenue"],
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
    
    # Normalize order_date: convert to date-only (drop time)
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
        report['cleaning_steps'].append("💰 Cleaned currency fields (removed ₹, $, commas)")
    
    # Step 6: Fill missing quantities with 0
    if mappings.get('quantity'):
        quantity_col = mappings['quantity']
        cleaned_df[quantity_col] = pd.to_numeric(cleaned_df[quantity_col], errors='coerce').fillna(0).astype(int)
        report['cleaning_steps'].append("📦 Filled missing quantities with 0")
    
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
    
    # Step 10: Detect and handle duplicates
    if len(key_columns) >= 3:  # Need at least order_id, sku, order_date
        # Create composite key for duplicate detection
        cleaned_df['composite_key'] = (
            cleaned_df[mappings['order_id']].astype(str) + "|" +
            cleaned_df[mappings['sku']].astype(str) + "|" +
            cleaned_df[mappings['order_date']].astype(str)
        )
        
        # Find duplicates
        duplicate_mask = cleaned_df.duplicated(subset=['composite_key'], keep='first')
        duplicates_count = duplicate_mask.sum()
        report['duplicates_found'] = duplicates_count
        
        if duplicates_count > 0:
            # Keep first occurrence, drop duplicates
            cleaned_df = cleaned_df[~duplicate_mask]
            report['cleaning_steps'].append(f"🔄 Removed {duplicates_count} duplicate rows (kept first occurrence)")
    
    # Step 11: Check for invalid revenue (negative values)
    if mappings.get('revenue_amount'):
        revenue_col = mappings['revenue_amount']
        invalid_revenue = (cleaned_df[revenue_col] < 0).sum()
        report['invalid_revenue_rows'] = invalid_revenue
        if invalid_revenue > 0:
            # Set negative revenue to 0
            cleaned_df.loc[cleaned_df[revenue_col] < 0, revenue_col] = 0
            report['cleaning_steps'].append(f"💰 Fixed {invalid_revenue} rows with negative revenue (set to 0)")
    
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
    
    print(f"🧹 Data cleaning completed: {report['rows_kept']} rows kept, {report['rows_dropped']} dropped")
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

def parse_revenue(series: pd.Series) -> pd.Series:
    """Parse revenue column, stripping currency symbols"""
    cleaned = series.astype(str).str.replace(r'[₹$,\s]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def create_sales_table(conn: sqlite3.Connection) -> None:
    """Create sales table with standard schema (idempotent)"""
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT,
            order_id TEXT,
            sku TEXT,
            product_name TEXT,
            quantity INTEGER,
            revenue_in_inr REAL,
            region TEXT,
            status TEXT,
            data_source TEXT,
            upload_date TEXT,
            month_tag TEXT,
            normalized_order_id TEXT,
            normalized_sku TEXT,
            normalized_order_date TEXT,
            UNIQUE(normalized_order_id, normalized_sku, normalized_order_date)
        )
    ''')
    conn.commit()

def create_staging_table(conn: sqlite3.Connection) -> None:
    """Create staging table for CSV ingestion"""
    cursor = conn.cursor()
    cursor.execute(f'DROP TABLE IF EXISTS {STAGING_TABLE}')
    cursor.execute(f'''
        CREATE TABLE {STAGING_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT,
            order_id TEXT,
            sku TEXT,
            product_name TEXT,
            quantity INTEGER,
            revenue_in_inr REAL,
            region TEXT,
            status TEXT,
            data_source TEXT,
            upload_date TEXT,
            month_tag TEXT,
            normalized_order_id TEXT,
            normalized_sku TEXT,
            normalized_order_date TEXT
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
        clean_df["product_name"] = df[mappings["product_name"]].astype(str) if mappings.get("product_name") else None
        
        if mappings.get("quantity"):
            clean_df["quantity"] = pd.to_numeric(df[mappings["quantity"]], errors='coerce').fillna(0).astype(int)
        else: 
            clean_df["quantity"] = 0
        
        if mappings.get("revenue_amount"):
            clean_df["revenue_in_inr"] = parse_revenue(df[mappings["revenue_amount"]])
        else: 
            clean_df["revenue_in_inr"] = 0.0
        
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
                    (order_date, order_id, sku, product_name, quantity, revenue_in_inr, 
                     region, status, data_source, upload_date, month_tag,
                     normalized_order_id, normalized_sku, normalized_order_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['order_date'], row['order_id'], row['sku'], row['product_name'], 
                      row['quantity'], row['revenue_in_inr'], row['region'], row['status'],
                      row['data_source'], row['upload_date'], row['month_tag'],
                      row['normalized_order_id'], row['normalized_sku'], row['normalized_order_date']))
            
            # Upsert from staging to main table
            cursor.execute(f'''
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (order_date, order_id, sku, product_name, quantity, revenue_in_inr, 
                 region, status, data_source, upload_date, month_tag,
                 normalized_order_id, normalized_sku, normalized_order_date)
                SELECT order_date, order_id, sku, product_name, quantity, revenue_in_inr, 
                       region, status, data_source, upload_date, month_tag,
                       normalized_order_id, normalized_sku, normalized_order_date
                FROM {STAGING_TABLE}
            ''')
            
            # Get statistics
            cursor.execute(f'SELECT COUNT(*) FROM {STAGING_TABLE}')
            total_processed = cursor.fetchone()[0]
            
            cursor.execute(f'SELECT COUNT(*) FROM {TABLE_NAME}')
            total_in_main = cursor.fetchone()[0]
            
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
        clean_df["product_name"] = df[mappings["product_name"]].astype(str) if mappings.get("product_name") else None
        
        if mappings.get("quantity"):
            clean_df["quantity"] = pd.to_numeric(df[mappings["quantity"]], errors='coerce').fillna(0).astype(int)
        else: clean_df["quantity"] = 0
        
        if mappings.get("revenue_amount"):
            clean_df["revenue_in_inr"] = parse_revenue(df[mappings["revenue_amount"]])
        else: clean_df["revenue_in_inr"] = 0.0
        
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
                    (order_date, order_id, sku, product_name, quantity, revenue_in_inr, 
                     region, status, data_source, upload_date, month_tag)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (row['order_date'], row['order_id'], row['sku'], row['product_name'], 
                      row['quantity'], row['revenue_in_inr'], row['region'], row['status'],
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
                        (order_date, order_id, sku, product_name, quantity, revenue_in_inr, 
                         region, status, data_source, upload_date, month_tag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['order_date'], row['order_id'], row['sku'], row['product_name'], 
                          row['quantity'], row['revenue_in_inr'], row['region'], row['status'],
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

def get_date_filtered_data(start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Get filtered data for date range"""
    if not os.path.exists(DB_FILE): return None
    
    try:
        conn = sqlite3.connect(DB_FILE)
        query = f"""
            SELECT * FROM {TABLE_NAME} 
            WHERE order_date >= ? AND order_date <= ?
            ORDER BY order_date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        if df.empty: return None
        
        # Convert order_date to datetime
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df = df.dropna(subset=['order_date'])
        
        return df
        
    except Exception as e:
        print(f"Error getting filtered data: {e}")
        return None

def compute_business_kpis(df: pd.DataFrame) -> Dict:
    """Compute business-grade KPIs from filtered data"""
    if df is None or df.empty:
        return {}
    
    print(f"Computing KPIs for {len(df)} rows")
    
    kpis = {}
    
    # Basic metrics
    kpis['total_revenue'] = df['revenue_in_inr'].sum()
    kpis['total_orders'] = df['order_id'].nunique() if 'order_id' in df.columns else 0
    kpis['units_sold'] = df['quantity'].sum() if 'quantity' in df.columns else 0
    kpis['aov'] = kpis['total_revenue'] / kpis['total_orders'] if kpis['total_orders'] > 0 else 0
    
    # WoW comparison (if we have enough data)
    if len(df) > 0:
        current_week_start = df['order_date'].max() - timedelta(days=7)
        previous_week_start = current_week_start - timedelta(days=7)
        
        current_week_data = df[df['order_date'] >= current_week_start]
        previous_week_data = df[(df['order_date'] >= previous_week_start) & (df['order_date'] < current_week_start)]
        
        if len(previous_week_data) > 0:
            prev_revenue = previous_week_data['revenue_in_inr'].sum()
            curr_revenue = current_week_data['revenue_in_inr'].sum()
            if prev_revenue > 0:
                kpis['wow_revenue_change'] = ((curr_revenue - prev_revenue) / prev_revenue) * 100
            else:
                kpis['wow_revenue_change'] = 0
        else:
            kpis['wow_revenue_change'] = None
    else:
        kpis['wow_revenue_change'] = None
    
    # Top products by revenue
    if 'sku' in df.columns and 'product_name' in df.columns:
        product_revenue = df.groupby(['sku', 'product_name']).agg({
            'revenue_in_inr': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
        # Create display names
        product_revenue['display_name'] = product_revenue.apply(
            lambda row: row['product_name'] if pd.notna(row['product_name']) and str(row['product_name']).strip() != '' else f"Unknown (SKU: {row['sku']})",
            axis=1
        )
        
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
        # Group by week and SKU
        df['week'] = df['order_date'].dt.to_period('W')
        weekly_data = df.groupby(['week', 'sku', 'product_name']).agg({
            'revenue_in_inr': 'sum',
            'quantity': 'sum'
        }).reset_index()
        
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
        
        # Create display names
        comparison['display_name'] = comparison.apply(
            lambda row: row['product_name'] if pd.notna(row['product_name']) and str(row['product_name']).strip() != '' else f"Unknown (SKU: {row['sku']})",
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
        results, _ = run_sql_query("SELECT SUM(revenue_in_inr) as total_revenue FROM sales")
        if results:
            return f"**Total Revenue:** {format_inr(results[0][0])}"
    
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
        
        results, columns = run_sql_query(f"""
            SELECT sku, product_name, SUM(revenue_in_inr) as revenue 
            FROM sales 
            WHERE sku IS NOT NULL 
            GROUP BY sku, product_name
            ORDER BY revenue DESC 
            LIMIT {limit}
        """)
        
        if results:
            response = "**Top Products by Revenue:**\n"
            for i, (sku, product_name, revenue) in enumerate(results, 1):
                if product_name and product_name.strip():
                    display_name = f"{product_name} (SKU: {sku})"
                else:
                    display_name = f"SKU: {sku}"
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
        # Get top product by revenue
        if 'sku' in df.columns and 'product_name' in df.columns:
            product_revenue = df.groupby(['sku', 'product_name'])['revenue_in_inr'].sum().reset_index()
            if not product_revenue.empty:
                top_product = product_revenue.loc[product_revenue['revenue_in_inr'].idxmax()]
                product_name = top_product['product_name'] if pd.notna(top_product['product_name']) else f"Unknown (SKU: {top_product['sku']})"
                return f"The best performing product is {product_name} with {format_inr(top_product['revenue_in_inr'])} revenue."
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
                    if st.button("⚠️ Confirm Clear", type="secondary"):
                        try:
                            conn = sqlite3.connect(DB_FILE)
                            cursor = conn.cursor()
                            cursor.execute(f'DELETE FROM {TABLE_NAME}')
                            cursor.execute('DELETE FROM dataset_registry')
                            conn.commit()
                            conn.close()
                            st.session_state.has_existing_data = False
                            st.success("✅ All data cleared!")
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
                        cleaned_df, cleaning_report = clean_dataframe(st.session_state.uploaded_df, st.session_state.mappings)
                        
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
                        mode = "append" if upload_mode == "Append to existing data" else "replace"
                        filename = uploaded_file.name if uploaded_file else "uploaded_file.csv"
                        
                        # Save raw and cleaned data
                        raw_path, cleaned_path = save_raw_and_cleaned_data(
                            st.session_state.uploaded_df, 
                            st.session_state.cleaned_df, 
                            filename
                        )
                        
                        # Ingest cleaned data
                        result = robust_ingest_csv(st.session_state.cleaned_df, st.session_state.mappings, filename, mode)
                        st.write(result)
                        
                        if "✅" in result:
                            # Save mapping
                            save_mapping(list(df.columns), st.session_state.mappings)
                            st.session_state.show_mapping_modal = False
                            st.session_state.show_cleaning_preview = False
                            st.session_state.has_existing_data = check_existing_data()
                            
                            # Show cleaning summary
                            if st.session_state.cleaning_report:
                                report = st.session_state.cleaning_report
                                st.success(f"🧹 **Cleaning Summary**: {report['rows_kept']} rows kept, {report['duplicates_found']} duplicates removed, {report['rows_dropped']} rows dropped")
                            
                            st.rerun()
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
            
            # Date filter
            st.markdown("### 📅 Date Range Filter")
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                preset = st.selectbox(
                    "Quick Presets:",
                    ["Custom Range", "Last 7 days", "Last 30 days", "This Month", "Last Month"],
                    key="date_preset"
                )
            
            with col2:
                if preset == "Custom Range":
                    start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
                    end_date = st.date_input("End Date", value=datetime.now().date())
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
            
            with col3:
                st.markdown("**Range:**")
                st.markdown(f"{start_date} to {end_date}")
            
            # Get filtered data and compute KPIs
            if st.button("🔄 Refresh Dashboard", type="primary"):
                st.rerun()
            
            # Load and compute KPIs
            df = get_date_filtered_data(str(start_date), str(end_date))
            if df is not None and not df.empty:
                kpis = compute_business_kpis(df)
                movers = compute_movers_decliners(df)
                
                # Main KPI cards
                st.markdown("---")
                st.markdown("### 📈 Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    wow_text = ""
                    if kpis.get('wow_revenue_change') is not None:
                        change = kpis['wow_revenue_change']
                        wow_text = f"WoW: {change:+.1f}%" if change != 0 else "WoW: 0%"
                    
                    st.metric(
                        label="💰 Total Revenue",
                        value=format_inr(kpis['total_revenue']),
                        delta=wow_text if wow_text else None
                    )
                
                with col2:
                    st.metric(
                        label="📦 Orders",
                        value=f"{kpis['total_orders']:,}"
                    )
                
                with col3:
                    st.metric(
                        label="📊 Units Sold",
                        value=f"{kpis['units_sold']:,}"
                    )
                
                with col4:
                    st.metric(
                        label="💵 AOV",
                        value=format_inr(kpis['aov'])
                    )
                
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