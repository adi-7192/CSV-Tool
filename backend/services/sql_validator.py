"""
SQL Validator Service

Validates SQL queries against the actual database schema before execution.
Prevents errors from wrong column names, invalid tables, etc.
"""

import re
import logging
from typing import Tuple, List, Set, Optional

logger = logging.getLogger(__name__)


def get_actual_columns() -> Set[str]:
    """Get set of actual column names from database"""
    from core.database import get_connection
    
    try:
        conn = get_connection()
        tables = conn.execute("SHOW TABLES").fetchdf()
        if 'sales' not in tables['name'].values:
            return set()
        
        columns_df = conn.execute("DESCRIBE sales").fetchdf()
        return set(columns_df['column_name'].tolist())
    except Exception as e:
        logger.error(f"Error getting columns: {e}")
        return set()


def get_date_column() -> str:
    """Get the actual date column name from database"""
    columns = get_actual_columns()
    
    # Priority order for date column
    for preferred in ['order_date', 'Invoice Date', 'invoice_date', 'Order Date']:
        if preferred in columns:
            return preferred
    
    # Fallback: any column with 'date' in name
    date_cols = [c for c in columns if 'date' in c.lower()]
    return date_cols[0] if date_cols else 'order_date'


def get_revenue_column() -> str:
    """Get the actual revenue column name from database"""
    columns = get_actual_columns()
    
    for preferred in ['revenue_amount', 'Invoice Amount', 'invoice_amount', 'revenue']:
        if preferred in columns:
            return preferred
    
    rev_cols = [c for c in columns if 'revenue' in c.lower() or 'amount' in c.lower()]
    return rev_cols[0] if rev_cols else 'revenue_amount'


def get_transaction_type_column() -> str:
    """Get the actual transaction type column name from database"""
    columns = get_actual_columns()
    
    for preferred in ['transaction_type', 'Transaction Type']:
        if preferred in columns:
            return preferred
    
    txn_cols = [c for c in columns if 'transaction' in c.lower() or 'type' in c.lower()]
    return txn_cols[0] if txn_cols else 'transaction_type'


def extract_quoted_columns(sql: str) -> List[str]:
    """Extract column names from double-quoted identifiers in SQL"""
    pattern = r'"([^"]+)"'
    return re.findall(pattern, sql)


def extract_unquoted_columns(sql: str) -> List[str]:
    """
    Extract potential column references that aren't quoted.
    This is a heuristic and may not be 100% accurate.
    """
    # Common SQL patterns that might contain column names
    # After SELECT, WHERE, GROUP BY, ORDER BY, etc.
    potential_cols = []
    
    # Look for patterns like column = 'value' or column >= 'value'
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|>=|<=|>|<|!=|LIKE|ILIKE|IN|IS)\s'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    potential_cols.extend(matches)
    
    # Look for patterns after SELECT (before FROM)
    select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        # Extract identifiers
        col_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        cols = re.findall(col_pattern, select_clause)
        # Filter out SQL keywords
        keywords = {'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 
                   'AS', 'AND', 'OR', 'NOT', 'NULL', 'DISTINCT', 'ABS', 'CAST', 'DATE', 'NULLIF'}
        potential_cols.extend([c for c in cols if c.upper() not in keywords])
    
    return list(set(potential_cols))


def validate_sql_columns(sql: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate that SQL uses correct column names
    
    Returns:
        (is_valid, error_message, corrected_sql)
    """
    actual_columns = get_actual_columns()
    if not actual_columns:
        return True, "", sql  # Can't validate, let it through
    
    # Column mappings for common mistakes
    column_corrections = {
        'Invoice Date': get_date_column(),
        'invoice_date': get_date_column(),
        'Invoice Amount': get_revenue_column(),
        'invoice_amount': get_revenue_column(),
        'Transaction Type': get_transaction_type_column(),
    }
    
    # Check quoted columns
    quoted_cols = extract_quoted_columns(sql)
    corrected_sql = sql
    errors = []
    
    for col in quoted_cols:
        if col not in actual_columns:
            # Check if we have a correction
            if col in column_corrections:
                correct_col = column_corrections[col]
                if correct_col in actual_columns:
                    corrected_sql = corrected_sql.replace(f'"{col}"', f'"{correct_col}"')
                    logger.info(f"Auto-corrected column '{col}' to '{correct_col}'")
                else:
                    errors.append(f"Column '{col}' not found (suggested: {correct_col} also not found)")
            else:
                # Try to find a similar column
                similar = find_similar_column(col, actual_columns)
                if similar:
                    corrected_sql = corrected_sql.replace(f'"{col}"', f'"{similar}"')
                    logger.info(f"Auto-corrected column '{col}' to '{similar}'")
                else:
                    errors.append(f"Column '{col}' not found. Available: {', '.join(sorted(actual_columns)[:10])}")
    
    if errors:
        return False, "; ".join(errors), corrected_sql
    
    return True, "", corrected_sql


def find_similar_column(target: str, available: Set[str]) -> Optional[str]:
    """Find a similar column name using fuzzy matching"""
    target_lower = target.lower().replace(' ', '_').replace('-', '_')
    
    # Exact match (case-insensitive)
    for col in available:
        if col.lower() == target_lower:
            return col
    
    # Partial match
    for col in available:
        col_normalized = col.lower().replace(' ', '_').replace('-', '_')
        if target_lower in col_normalized or col_normalized in target_lower:
            return col
    
    # Word overlap
    target_words = set(target_lower.split('_'))
    best_match = None
    best_score = 0
    
    for col in available:
        col_words = set(col.lower().replace(' ', '_').replace('-', '_').split('_'))
        overlap = len(target_words & col_words)
        if overlap > best_score:
            best_score = overlap
            best_match = col
    
    if best_score > 0:
        return best_match
    
    return None


def auto_correct_sql(sql: str) -> str:
    """
    Automatically correct common column name issues in SQL
    
    Returns corrected SQL
    """
    actual_columns = get_actual_columns()
    if not actual_columns:
        return sql
    
    date_col = get_date_column()
    rev_col = get_revenue_column()
    txn_col = get_transaction_type_column()
    
    corrected = sql
    
    # Fix common column name issues
    corrections = [
        (r'"Invoice Date"', f'"{date_col}"'),
        (r'"Invoice Amount"', f'"{rev_col}"'),
        (r'"Transaction Type"', f'"{txn_col}"'),
        (r'"invoice_date"', f'"{date_col}"'),
        (r'"invoice_amount"', f'"{rev_col}"'),
    ]
    
    for pattern, replacement in corrections:
        if pattern != replacement:  # Only replace if different
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    return corrected

