"""
Upload Service - Production-ready CSV processing with schema standardization
Handles any CSV format, transforms to standardized schema, adds data lineage
"""
import pandas as pd
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from core.config import settings
from core.database import get_connection, table_exists

logger = logging.getLogger(__name__)

# =============================================================================
# STANDARDIZED SCHEMA DEFINITION
# =============================================================================

STANDARD_SCHEMA = {
    'order_id': 'VARCHAR',          # Unique order/invoice identifier
    'order_date': 'DATE',           # Order placement date
    'revenue_amount': 'DOUBLE',     # Transaction amount (revenue or refund)
    'transaction_type': 'VARCHAR',  # Shipment, Refund, Cancel, FreeReplacement
    'sku': 'VARCHAR',               # Product SKU/ASIN
    'quantity': 'INTEGER',          # Order quantity
    'region': 'VARCHAR',            # Geographic region/city
    'shipping_amount': 'DOUBLE',    # Shipping cost
    
    # Data lineage columns (always added)
    'source_file': 'VARCHAR',       # Original CSV filename
    'ingestion_id': 'VARCHAR',      # Unique upload session ID
    'loaded_at': 'TIMESTAMP',       # When uploaded
    'updated_at': 'TIMESTAMP',      # When last modified
}

# Business key for deduplication (order_id + transaction_type uniquely identifies a record)
BUSINESS_KEY = ['order_id', 'transaction_type']

# =============================================================================
# HELPERS
# =============================================================================

def generate_synthetic_order_id(transaction_type: Optional[str], index: Optional[int] = None) -> str:
    """
    Generate a synthetic order_id for rows missing an identifier.

    Format: "UNKNOWN_{transaction_type}_{index}_{timestamp}"
    - transaction_type: preserved as given (defaults to 'Unknown')
    - index: per-batch monotonic seed; if not provided, derived from uuid
    - timestamp: YYYYMMDDHHMMSS
    """
    txn = (transaction_type or 'Unknown')
    ts = datetime.now().strftime('%Y%m%d%H%M%S')
    # Derive a numeric seed from UUID if index not provided
    if index is None:
        index_val = int(uuid.uuid4().int % 1_000_000)
    else:
        index_val = int(index)
    return f"UNKNOWN_{txn}_{index_val}_{ts}"

# =============================================================================
# COLUMN MAPPING & DETECTION
# =============================================================================

def detect_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect column mapping from CSV to standardized schema
    Case-insensitive, handles common variations
    
    Returns: {'order_id': 'Invoice Number', 'order_date': 'Invoice Date', ...}
    """
    
    # Create lowercase version for matching
    columns_lower = {col: col.lower().strip() for col in df.columns}
    mapping = {}
    
    # Define synonyms for each standard column
    synonyms = {
        'order_id': [
            'invoice number', 'invoice_number', 'invoice no', 'invoice#',
            'order id', 'order_id', 'order number', 'order_number',
            'transaction id', 'transaction_id'
        ],
        'order_date': [
            'invoice date', 'invoice_date',
            'order date', 'order_date',
            'date', 'transaction date', 'created date', 'timestamp'
        ],
        'revenue_amount': [
            'invoice amount', 'invoice_amount',
            'order amount', 'order_amount',
            'amount', 'total', 'revenue', 'price', 'value',
            'grand total', 'net amount'
        ],
        'transaction_type': [
            'transaction type', 'transaction_type',
            'type', 'status', 'order type', 'invoice type'
        ],
        'sku': [
            'sku',  # Exact match has highest priority - will be matched first
            'asin', 
            'product code', 
            'item code',  # "item code" is OK, but not "Shipment Item Id"
            'product code (sku)',  # Common variation
            # Note: "item" is excluded to avoid matching "Shipment Item Id"
        ],
        'quantity': [
            'quantity', 'qty', 'amount', 'units', 'count'
        ],
        'region': [
            # PRIORITY: "Ship To City" is the correct column (customer delivery location)
            'ship to city',  # Highest priority - customer location
            'ship_to_city',  # Alternative format
            # Then other region synonyms
            'region', 'city', 'state', 'location', 'area', 'place',
            # EXCLUDE: "Bill From City" - this is seller location, NOT customer location
            # We explicitly don't include it here to avoid mapping it
        ],
        'shipping_amount': [
            'shipping', 'shipping amount', 'shipping_amount',
            'freight', 'delivery charge'
        ],
    }
    
    # Match each standard column
    for standard_col, synonym_list in synonyms.items():
        # Special handling for SKU - prioritize exact match and avoid "item" in compound names
        if standard_col == 'sku':
            sku_found = False
            
            # First pass: Look for exact "sku" match (highest priority)
            for original_col, col_lower in columns_lower.items():
                if col_lower == 'sku' and original_col not in mapping.values():
                    mapping[standard_col] = original_col
                    logger.info(f"Mapped '{standard_col}' → '{original_col}' (exact match)")
                    sku_found = True
                    break
            
            # Second pass: Try other synonyms, but AVOID matching "item" from compound names
            if not sku_found:
                for original_col, col_lower in columns_lower.items():
                    # Skip if already matched
                    if original_col in mapping.values():
                        continue
                    
                    # Check each synonym
                    for synonym in synonym_list:
                        # Skip "item" synonym to avoid matching "Shipment Item Id"
                        if synonym == 'item':
                            continue
                        
                        # Check if synonym matches (as substring)
                        if synonym in col_lower:
                            # Additional check: avoid matching "item" anywhere in the column name
                            # (even if it's part of another synonym match)
                            if 'item' in col_lower:
                                # Only proceed if it's clearly an SKU-related column
                                # Allow: "item code", "product item", "item_code"
                                # Reject: "shipment item id", "item description", "item name"
                                if any(bad_pattern in col_lower for bad_pattern in [
                                    'shipment item', 'item id', 'item description', 
                                    'item name', 'item detail', 'item number'
                                ]):
                                    continue
                            
                            mapping[standard_col] = original_col
                            logger.info(f"Mapped '{standard_col}' → '{original_col}' (via '{synonym}')")
                            sku_found = True
                            break
                    
                    if sku_found:
                        break
        else:
            # Normal matching for other columns
            # Special handling for region - prioritize "Ship To City" and exclude "Bill From City"
            if standard_col == 'region':
                region_found = False
                
                # FIRST PRIORITY: Look for "Ship To City" (exact match)
                for original_col, col_lower in columns_lower.items():
                    if col_lower == 'ship to city' and original_col not in mapping.values():
                        mapping[standard_col] = original_col
                        logger.info(f"Mapped '{standard_col}' → '{original_col}' (Ship To City - customer location)")
                        region_found = True
                        break
                
                # SECOND PRIORITY: Look for "ship_to_city" (alternative format)
                if not region_found:
                    for original_col, col_lower in columns_lower.items():
                        if col_lower == 'ship_to_city' and original_col not in mapping.values():
                            mapping[standard_col] = original_col
                            logger.info(f"Mapped '{standard_col}' → '{original_col}' (ship_to_city - customer location)")
                            region_found = True
                            break
                
                # THIRD PRIORITY: Other synonyms, but EXCLUDE "Bill From City"
                if not region_found:
                    for original_col, col_lower in columns_lower.items():
                        # Skip if already matched
                        if original_col in mapping.values():
                            continue
                        
                        # EXCLUDE "Bill From City" - this is seller location, NOT customer location
                        if col_lower == 'bill from city' or col_lower == 'bill_from_city':
                            logger.info(f"Skipping '{original_col}' - this is Bill From City (seller location, not customer)")
                            continue
                        
                        # Check if any synonym matches
                        for synonym in synonym_list:
                            if synonym in col_lower:
                                mapping[standard_col] = original_col
                                logger.info(f"Mapped '{standard_col}' → '{original_col}' (via '{synonym}')")
                                region_found = True
                                break
                        
                        if region_found:
                            break
            else:
                # Normal matching for other columns
                for original_col, col_lower in columns_lower.items():
                    # Skip if already matched
                    if original_col in mapping.values():
                        continue
                    
                    # Check if any synonym matches
                    for synonym in synonym_list:
                        if synonym in col_lower:
                            # Special handling for revenue_amount - exclude tax columns
                            if standard_col == 'revenue_amount':
                                if any(tax_word in col_lower for tax_word in ['tax', 'gst', 'cgst', 'sgst', 'igst']):
                                    continue
                            
                            mapping[standard_col] = original_col
                            logger.info(f"Mapped '{standard_col}' → '{original_col}'")
                            break
                    
                    if standard_col in mapping:
                        break
    
    logger.info(f"Column mapping complete: {len(mapping)} columns mapped")
    return mapping

# =============================================================================
# DATA TRANSFORMATION & CLEANING
# =============================================================================

def transform_to_standard_schema(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    filename: str,
    ingestion_id: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Transform CSV data to standardized schema with full data quality reporting
    
    Returns: (transformed_df, validation_report)
    """
    
    validation_report = {
        'rows_raw': len(df),
        'rows_cleaned': 0,
        'columns_original': len(df.columns),
        'columns_mapped': len(column_mapping),
        'issues': [],
        'warnings': [],
        'transformations': [],
    }
    
    # Start with empty DataFrame with standard schema
    df_standard = pd.DataFrame()
    
    # 1. Map and rename columns
    for standard_col, original_col in column_mapping.items():
        if original_col in df.columns:
            df_standard[standard_col] = df[original_col].copy()
            validation_report['transformations'].append(
                f"Renamed '{original_col}' → '{standard_col}'"
            )
    
    # 2. Convert data types with validation
    
    # 2a. Order Date → DATE
    if 'order_date' in df_standard.columns:
        df_standard['order_date'] = pd.to_datetime(df_standard['order_date'], errors='coerce')
        invalid_dates = df_standard['order_date'].isna().sum()
        if invalid_dates > 0:
            validation_report['warnings'].append(f"{invalid_dates} invalid dates found and excluded")
            df_standard = df_standard[df_standard['order_date'].notna()]
    else:
        validation_report['issues'].append("order_date column not found - this is critical!")
    
    # 2b. Revenue Amount → NUMERIC
    if 'revenue_amount' in df_standard.columns:
        # Clean currency symbols and commas
        df_standard['revenue_amount'] = df_standard['revenue_amount'].astype(str).str.replace(r'[₹$,\s]', '', regex=True)
        df_standard['revenue_amount'] = pd.to_numeric(df_standard['revenue_amount'], errors='coerce')
        invalid_amounts = df_standard['revenue_amount'].isna().sum()
        if invalid_amounts > 0:
            validation_report['warnings'].append(f"{invalid_amounts} invalid amounts converted to 0")
            df_standard['revenue_amount'].fillna(0, inplace=True)
    else:
        validation_report['issues'].append("revenue_amount column not found - this is critical!")
    
    # 2c. Quantity → INTEGER
    if 'quantity' in df_standard.columns:
        df_standard['quantity'] = pd.to_numeric(df_standard['quantity'], errors='coerce').fillna(1).astype(int)
    else:
        df_standard['quantity'] = 1  # Default to 1 if not present
    
    # 2d. Shipping Amount → NUMERIC
    if 'shipping_amount' in df_standard.columns:
        df_standard['shipping_amount'] = pd.to_numeric(df_standard['shipping_amount'], errors='coerce').fillna(0)
    else:
        df_standard['shipping_amount'] = 0.0  # Default to 0 if not present
    
    # 2e. String columns → clean whitespace
    for col in ['order_id', 'transaction_type', 'sku', 'region']:
        if col in df_standard.columns:
            df_standard[col] = df_standard[col].astype(str).str.strip()
            # Replace 'nan' string with None
            df_standard[col] = df_standard[col].replace('nan', None)
    
    # 3. Normalize transaction types
    if 'transaction_type' in df_standard.columns:
        def normalize_txn_type(val):
            if pd.isna(val) or val is None:
                return None
            val_str = str(val).strip().lower()
            if val_str in ['shipment', 'ship']:
                return 'Shipment'
            elif val_str in ['refund', 'refunds']:
                return 'Refund'
            elif val_str in ['cancel', 'cancelled', 'cancellation']:
                return 'Cancel'
            elif val_str in ['freereplacement', 'free replacement', 'free_replacement']:
                return 'FreeReplacement'
            else:
                return val_str.title()  # Capitalize first letter
        
        df_standard['transaction_type'] = df_standard['transaction_type'].apply(normalize_txn_type)
        validation_report['transformations'].append("Normalized transaction_type values")
    
    # 4.1 Ensure missing order_id values are synthesized for reliable deduplication
    if 'order_id' in df_standard.columns:
        def _is_missing_order_id(val) -> bool:
            if val is None or pd.isna(val):
                return True
            s = str(val).strip()
            return s == '' or s.lower() in {'none', 'null'}

        missing_mask = df_standard['order_id'].apply(_is_missing_order_id)
        if missing_mask.any():
            base_index = int(uuid.uuid4().int % 1_000_000)
            count_generated = 0
            for pos, row_idx in enumerate(df_standard.index[missing_mask]):
                txn_val = df_standard.at[row_idx, 'transaction_type'] if 'transaction_type' in df_standard.columns else 'Unknown'
                synthetic_id = generate_synthetic_order_id(txn_val, base_index + pos)
                df_standard.at[row_idx, 'order_id'] = synthetic_id
                count_generated += 1
            logger.info(f"🔧 Synthesized {count_generated} order_id values for missing IDs")
            validation_report['warnings'].append(
                f"{count_generated} rows had missing order_id; synthetic IDs generated to enable deduplication"
            )

    # 4. Add data lineage columns
    df_standard['source_file'] = filename
    df_standard['ingestion_id'] = ingestion_id
    df_standard['loaded_at'] = datetime.now()
    df_standard['updated_at'] = datetime.now()
    
    validation_report['transformations'].append("Added data lineage: source_file, ingestion_id, loaded_at, updated_at")
    
    # 5. Remove duplicates based on business key
    if all(col in df_standard.columns for col in BUSINESS_KEY):
        before_dedup = len(df_standard)
        df_standard = df_standard.drop_duplicates(subset=BUSINESS_KEY, keep='last')
        duplicates_removed = before_dedup - len(df_standard)
        if duplicates_removed > 0:
            validation_report['warnings'].append(
                f"{duplicates_removed} duplicate records removed (business key: {', '.join(BUSINESS_KEY)})"
            )
    
    validation_report['rows_cleaned'] = len(df_standard)
    
    # 6. Validate critical columns exist
    required_cols = ['order_id', 'order_date', 'revenue_amount', 'transaction_type']
    missing_cols = [col for col in required_cols if col not in df_standard.columns]
    if missing_cols:
        validation_report['issues'].append(f"Critical columns missing: {', '.join(missing_cols)}")
    
    logger.info(f"✅ Transformation complete: {len(df_standard)} rows, {len(df_standard.columns)} columns")
    logger.info(f"📊 Standardized columns: {list(df_standard.columns)}")
    
    return df_standard, validation_report

# =============================================================================
# DATABASE STORAGE WITH UPSERT
# =============================================================================

def store_to_database(
    df_standard: pd.DataFrame,
    table_name: str = 'sales',
) -> Dict[str, Any]:
    """
    Store standardized data to database with UPSERT logic
    Prevents duplicates by deleting matching business keys before insert
    
    Returns: Storage metadata
    """
    
    conn = get_connection()
    
    # Register DataFrame as temporary table
    conn.register('temp_upload', df_standard)
    
    rows_inserted = len(df_standard)
    rows_deleted = 0
    
    # UPSERT logic: Delete existing records with matching business keys, then insert all
    if table_exists(table_name):
        # Check which business key columns exist in both table and DataFrame
        table_info = conn.execute(f"DESCRIBE {table_name}").df()
        table_columns = table_info['column_name'].tolist()
        
        # Build WHERE clause for business key matching
        business_key_conditions = []
        for col in BUSINESS_KEY:
            if col in df_standard.columns and col in table_columns:
                business_key_conditions.append(f'"{col}"')
        
        if len(business_key_conditions) >= 2:
            # Build EXISTS clause for matching business keys
            where_parts = []
            for col in business_key_conditions:
                col_name = col.strip('"')
                where_parts.append(f'{table_name}."{col_name}" = temp_upload."{col_name}"')
            
            where_clause = ' AND '.join(where_parts)
            
            delete_sql = f"""
            DELETE FROM {table_name}
            WHERE EXISTS (
                SELECT 1 FROM temp_upload
                WHERE {where_clause}
            )
            """
            
            try:
                result = conn.execute(delete_sql)
                # Try to get rowcount
                if hasattr(result, 'rowcount'):
                    rows_deleted = result.rowcount
                else:
                    # Fallback: count affected rows
                    check_sql = f"""
                    SELECT COUNT(*) as count FROM {table_name}
                    WHERE EXISTS (
                        SELECT 1 FROM temp_upload
                        WHERE {where_clause}
                    )
                    """
                    count_result = conn.execute(check_sql).fetchone()
                    rows_deleted = count_result[0] if count_result else 0
                
                logger.info(f"🗑️  Upsert: Deleted {rows_deleted} existing records with matching business keys")
            except Exception as e:
                logger.warning(f"Could not delete existing records: {e}")
        
        # Insert all records from temp table
        # First, ensure column order matches
        table_info = conn.execute(f"DESCRIBE {table_name}").df()
        table_columns = table_info['column_name'].tolist()
        
        # Get columns that exist in both
        df_columns = df_standard.columns.tolist()
        common_columns = [col for col in table_columns if col in df_columns]
        new_columns = [col for col in df_columns if col not in table_columns]
        
        # Add new columns to table if any
        if new_columns:
            logger.info(f"➕ Adding {len(new_columns)} new columns to table: {new_columns[:5]}...")
            for new_col in new_columns:
                try:
                    # Infer column type
                    col_type = "VARCHAR"
                    if df_standard[new_col].dtype in ['int64', 'Int64']:
                        col_type = "BIGINT"
                    elif df_standard[new_col].dtype in ['float64', 'Float64']:
                        col_type = "DOUBLE"
                    elif pd.api.types.is_datetime64_any_dtype(df_standard[new_col]):
                        col_type = "TIMESTAMP"
                    
                    conn.execute(f'ALTER TABLE {table_name} ADD COLUMN "{new_col}" {col_type}')
                    common_columns.append(new_col)
                    logger.info(f"✅ Added column {new_col} ({col_type})")
                except Exception as e:
                    logger.warning(f"Could not add column {new_col}: {e}")
        
        # Add missing columns to DataFrame (fill with NULL)
        missing_in_df = [col for col in table_columns if col not in df_standard.columns]
        if missing_in_df:
            logger.info(f"📝 Adding {len(missing_in_df)} missing columns to DataFrame (filled with NULL)")
            for missing_col in missing_in_df:
                df_standard[missing_col] = None
        
        # Re-register with aligned columns
        df_aligned = df_standard[table_columns].copy()
        conn.register('temp_upload', df_aligned)
        
        # Insert all records
        col_list = ', '.join([f'"{col}"' for col in common_columns])
        insert_sql = f"INSERT INTO {table_name} ({col_list}) SELECT {col_list} FROM temp_upload"
        conn.execute(insert_sql)
        logger.info(f"✅ Upsert: Inserted {rows_inserted} records")
    
    else:
        # First upload - create table with standard schema
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_upload")
        logger.info(f"📊 Created new table '{table_name}' with {rows_inserted} records")
    
    # Commit transaction
    conn.commit()
    
    return {
        'table_name': table_name,
        'rows_inserted': rows_inserted,
        'rows_deleted': rows_deleted,
        'upsert_performed': rows_deleted > 0,
    }

# =============================================================================
# MAIN UPLOAD PIPELINE
# =============================================================================

def process_csv_upload(
    file_content: bytes,
    filename: str,
) -> Dict[str, Any]:
    """
    Complete CSV upload pipeline with full error handling
    
    Pipeline:
    1. Read CSV
    2. Detect column mapping
    3. Transform to standard schema
    4. Validate data quality
    5. Store to database with upsert
    6. Log to ingestion_log (optional)
    
    Returns: Complete upload result with validation report
    """
    
    import time
    start_time = time.time()
    
    ingestion_id = str(uuid.uuid4())
    
    try:
        # Step 1: Read CSV
        df_raw = pd.read_csv(pd.io.common.BytesIO(file_content))
        logger.info(f"📄 CSV loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
        logger.info(f"📋 CSV columns: {list(df_raw.columns)[:10]}...")
        
        # Step 2: Detect column mapping
        column_mapping = detect_column_mapping(df_raw)
        
        if not column_mapping:
            return {
                'success': False,
                'error': 'Could not detect any required columns in CSV',
                'csv_columns': list(df_raw.columns),
            }
        
        # Verify critical columns are mapped
        required_cols = ['order_id', 'revenue_amount']
        missing_required = [col for col in required_cols if col not in column_mapping]
        
        if missing_required:
            return {
                'success': False,
                'error': f'Could not detect critical columns: {", ".join(missing_required)}',
                'detected_mapping': column_mapping,
                'csv_columns': list(df_raw.columns),
            }
        
        # Step 3: Transform to standard schema
        df_standard, validation_report = transform_to_standard_schema(
            df_raw, column_mapping, filename, ingestion_id
        )
        
        if len(df_standard) == 0:
            return {
                'success': False,
                'error': 'No valid rows after transformation',
                'validation_report': validation_report,
            }
        
        # Prepare user-friendly validation warnings
        validation_warnings: list[str] = []
        for msg in validation_report.get('warnings', []):
            # Normalize phrasing to be user-friendly
            if 'invalid dates' in msg:
                # e.g., "12 invalid dates found and excluded"
                num = ''.join([c for c in msg if c.isdigit()])
                if num:
                    validation_warnings.append(f"Excluded {num} rows with invalid dates")
                else:
                    validation_warnings.append("Excluded rows with invalid dates")
            elif 'duplicate' in msg:
                # e.g., "5 duplicate records removed"
                num = ''.join([c for c in msg if c.isdigit()])
                if num:
                    validation_warnings.append(f"Found and merged {num} duplicate transactions")
                else:
                    validation_warnings.append("Found and merged duplicate transactions")
            elif 'missing order_id' in msg or 'synthetic IDs' in msg:
                # Already framed well from transform step
                validation_warnings.append(msg.replace('order_id', 'order numbers'))
            else:
                validation_warnings.append(msg)

        # Step 4: Store to database
        storage_result = store_to_database(df_standard)
        
        # Step 5: Calculate processing time
        processing_time = time.time() - start_time
        
        # Step 6: Return complete result
        logger.info(f"✅ Upload complete in {processing_time:.2f}s: {len(df_standard)} rows uploaded")
        
        return {
            'success': True,
            'ingestion_id': ingestion_id,
            'filename': filename,
            'rows_raw': len(df_raw),
            'rows_uploaded': len(df_standard),
            'rows_deleted': storage_result['rows_deleted'],
            'column_mapping': column_mapping,
            'validation_report': validation_report,
            'storage_result': storage_result,
            'processing_time_seconds': round(processing_time, 2),
            'schema_standardized': True,
            'data_lineage_added': True,
            # New concise fields for UI consumption
            'rows_processed': len(df_raw),
            'rows_inserted': storage_result.get('rows_inserted', len(df_standard)),
            'validation_warnings': validation_warnings,
        }
    
    except Exception as e:
        logger.error(f"❌ Upload processing failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'error': str(e),
            'ingestion_id': ingestion_id,
            'filename': filename,
        }
