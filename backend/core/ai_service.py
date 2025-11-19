"""
AI Service - LLM integration for natural language queries

Supports multiple LLM providers:
- OpenAI (GPT-4 Turbo) - if user has API key
- Anthropic (Claude 3 Sonnet) - if user has API key
- Ollama (local) - fallback

Extracted from legacy/ai_assistant.py
"""
import requests
import json
import logging
import re
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from calendar import monthrange
from core.config import settings
from core.database import execute_query, get_connection
from utils.external_llm import generate_sql_with_external_llm, calculate_ollama_confidence
from utils.encryption import APIKeyEncryption

logger = logging.getLogger(__name__)


def normalize_header(header: str) -> str:
    """Normalize header for matching (lowercase, strip, collapse spaces)"""
    return re.sub(r'\s+', ' ', str(header).strip().lower())


def check_ollama_connection() -> bool:
    """
    Check if Ollama server is accessible
    
    Returns:
        True if connected, False otherwise
    """
    try:
        response = requests.get(
            f"{settings.OLLAMA_URL}/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Ollama connection failed: {str(e)}")
        return False


def get_database_schema() -> Dict[str, Any]:
    """
    Get database schema information for LLM context
    
    Returns detailed schema with column names, types, and business meaning
    """
    try:
        conn = get_connection()
        
        # Check if sales table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        if 'sales' not in tables['name'].values:
            return {
                'table': 'sales',
                'columns': [],
                'columns_with_descriptions': [],
                'important_notes': ['Table does not exist yet']
            }
        
        # Get column information with types
        columns_df = conn.execute("DESCRIBE sales").fetchdf()
        columns = columns_df['column_name'].tolist()
        column_types = dict(zip(columns_df['column_name'], columns_df['column_type']))
        
        # Get sample data to understand actual column names
        sample_df = execute_query("SELECT * FROM sales LIMIT 3")
        
        # Map actual columns to business meaning
        # The table may have original CSV column names or standardized names
        column_descriptions = {}
        
        # Check for common revenue column names
        revenue_cols = [c for c in columns if any(word in normalize_header(c) for word in ['revenue_calc', 'invoice amount', 'order amount', 'revenue_amount'])]
        if revenue_cols:
            primary_rev_col = revenue_cols[0]
            column_descriptions[primary_rev_col] = 'Revenue amount in Indian Rupees (₹). Already in INR - DO NOT multiply or convert. For shipments use this directly, for refunds take absolute value.'
        
        # Check for transaction type column
        txn_cols = [c for c in columns if 'transaction_type' in normalize_header(c) or 'transaction type' in normalize_header(c)]
        if txn_cols:
            primary_txn_col = txn_cols[0]
            column_descriptions[primary_txn_col] = 'Transaction type: "Shipment" (revenue), "Refund" (return), "Cancel" (cancellation), "FreeReplacement"'
        
        # Check for date columns and detect their actual type
        date_cols = [c for c in columns if 'date' in normalize_header(c)]
        if date_cols:
            primary_date_col = date_cols[0]
            date_col_type = column_types.get(primary_date_col, '').upper()
            # Check if it's VARCHAR/TEXT - needs casting
            if 'VARCHAR' in date_col_type or 'TEXT' in date_col_type or 'CHAR' in date_col_type:
                column_descriptions[primary_date_col] = f'Order/invoice date (VARCHAR/TEXT - MUST CAST to DATE: CAST("{primary_date_col}" AS DATE)). For month extraction use DATE_PART(\'month\', CAST("{primary_date_col}" AS DATE)) or EXTRACT(MONTH FROM CAST("{primary_date_col}" AS DATE))'
            else:
                column_descriptions[primary_date_col] = f'Order/invoice date ({date_col_type}). Use for filtering by date ranges.'
        
        # Check for order ID columns
        id_cols = [c for c in columns if any(word in normalize_header(c) for word in ['invoice number', 'order id', 'invoice_number', 'order_id'])]
        if id_cols:
            primary_id_col = id_cols[0]
            column_descriptions[primary_id_col] = 'Unique order/invoice identifier (TEXT)'
        
        # Check for SKU columns
        sku_cols = [c for c in columns if 'sku' in normalize_header(c)]
        if sku_cols:
            primary_sku_col = sku_cols[0]
            column_descriptions[primary_sku_col] = 'Product SKU identifier (TEXT)'
        
        # Build column list with descriptions and types
        columns_with_desc = []
        for col in columns[:20]:  # Limit to first 20 columns to avoid overwhelming prompt
            col_type = column_types.get(col, '')
            desc = column_descriptions.get(col, '')
            if desc:
                columns_with_desc.append(f'"{col}" ({col_type}) - {desc}')
            else:
                columns_with_desc.append(f'"{col}" ({col_type})')
        
        # Important notes for SQL generation
        important_notes = [
            'Revenue is ALREADY in Indian Rupees (₹). DO NOT multiply by any conversion factor.',
            'For revenue queries, filter by transaction_type = \'Shipment\' (use actual column name from schema)',
            'For refunds, use transaction_type = \'Refund\' and take ABS() of revenue amount',
            'CRITICAL: NET REVENUE calculation must include ALL components:',
            '  When user asks about "net revenue", "net earnings", "profit", "after costs", or "after deductions":',
            '  Formula: net_revenue = gross_revenue - refund_amount - cancellation_amount - free_replacement_amount',
            '  gross_revenue = SUM(revenue_amount) WHERE transaction_type = \'Shipment\'',
            '  refund_amount = ABS(SUM(revenue_amount)) WHERE transaction_type = \'Refund\'',
            '  cancellation_amount = ABS(SUM(revenue_amount)) WHERE transaction_type = \'Cancel\'',
            '  free_replacement_amount = ABS(SUM(revenue_amount)) WHERE transaction_type = \'FreeReplacement\'',
            '  Example SQL for net revenue:',
            '    SELECT SUM(CASE WHEN transaction_type = \'Shipment\' THEN revenue_amount ELSE 0 END) as gross_revenue,',
            '           ABS(SUM(CASE WHEN transaction_type = \'Refund\' THEN revenue_amount ELSE 0 END)) as refund_amount,',
            '           ABS(SUM(CASE WHEN transaction_type = \'Cancel\' THEN revenue_amount ELSE 0 END)) as cancel_amount,',
            '           ABS(SUM(CASE WHEN transaction_type = \'FreeReplacement\' THEN revenue_amount ELSE 0 END)) as free_repl_amount,',
            '           (SUM(CASE WHEN transaction_type = \'Shipment\' THEN revenue_amount ELSE 0 END)',
            '            - ABS(SUM(CASE WHEN transaction_type = \'Refund\' THEN revenue_amount ELSE 0 END))',
            '            - ABS(SUM(CASE WHEN transaction_type = \'Cancel\' THEN revenue_amount ELSE 0 END))',
            '            - ABS(SUM(CASE WHEN transaction_type = \'FreeReplacement\' THEN revenue_amount ELSE 0 END))) as net_revenue',
            '    FROM sales WHERE [date filters]',
            '  DO NOT just calculate refunds - net revenue requires ALL 4 components!',
            'Use actual column names from the schema above - they may have spaces (use double quotes)',
            'DO NOT reference columns that do not exist in the schema (e.g., needs_estimation)',
            'Use standard DuckDB SQL syntax (DATE_TRUNC, CURRENT_DATE, INTERVAL)',
            'CRITICAL: Date columns may be VARCHAR/TEXT - ALWAYS cast to DATE before using date functions',
            'For month extraction: Use DATE_PART(\'month\', CAST("Invoice Date" AS DATE)) or EXTRACT(MONTH FROM CAST("Invoice Date" AS DATE))',
            'For date filtering: Use CAST("Invoice Date" AS DATE) >= \'2025-07-01\'',
            'NEVER use EXTRACT or DATE_PART on VARCHAR columns without CAST - always CAST("Column" AS DATE) first'
        ]
        
        # Add specific column warnings based on what exists
        if revenue_cols:
            important_notes.insert(0, f'Use column "{revenue_cols[0]}" for revenue (already in INR, no conversion needed)')
        if txn_cols:
            important_notes.insert(1, f'Use column "{txn_cols[0]}" for transaction type filtering')
        
        return {
            'table': 'sales',
            'columns': columns,
            'columns_with_descriptions': columns_with_desc,
            'important_notes': important_notes,
            'sample_row_count': len(sample_df) if not sample_df.empty else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting database schema: {str(e)}")
        # Return fallback schema
        return {
            'table': 'sales',
            'columns': ['Invoice Number', 'Invoice Date', 'Invoice Amount', 'Transaction Type', 'Sku'],
            'columns_with_descriptions': [
                '"Invoice Number" (VARCHAR) - Order identifier',
                '"Invoice Date" (DATE) - Order date',
                '"Invoice Amount" (DOUBLE) - Revenue in INR (no conversion needed)',
                '"Transaction Type" (VARCHAR) - Shipment/Refund/Cancel',
                '"Sku" (VARCHAR) - Product SKU'
            ],
            'important_notes': [
                'Invoice Amount is already in INR - do not convert',
                'Filter by "Transaction Type" = \'Shipment\' for revenue',
                'Use double quotes around column names with spaces'
            ]
        }




async def generate_sql_from_question(
    question: str,
    schema: Optional[Dict[str, Any]] = None,
    source_file: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, float, float]:
    """
    Generate SQL query from natural language question
    
    Tries external LLMs first (OpenAI, Anthropic), then falls back to Ollama.
    
    Args:
        question: Natural language question
        schema: Optional database schema information (if None, will fetch)
        source_file: Optional source file filter to prevent multi-source data mixing
        user_id: Optional user ID to check for API keys
    
    Returns:
        Tuple of (sql_query, error_message, provider, response_time, confidence)
        - sql_query: Generated SQL or None
        - error_message: Error message or None
        - provider: 'openai', 'anthropic', 'ollama', or 'none'
        - response_time: Time taken in seconds
        - confidence: Confidence score (0.0-1.0)
    """
    # Get schema if not provided
    if schema is None:
        schema = get_database_schema()
    
    # Check if multiple source files exist (warn about potential data mixing)
    if source_file:
        # Add note about source file filtering
        schema_notes = schema.get('important_notes', [])
        if 'source_file' in str(schema.get('columns', [])):
            schema_notes.insert(0, f'IMPORTANT: Filter by source_file = \'{source_file}\' to avoid data from multiple CSV files')
    
    # Build detailed prompt with schema context
    columns_desc = '\n'.join([f'  {col}' for col in schema.get('columns_with_descriptions', [])])
    important_notes = '\n'.join([f'  - {note}' for note in schema.get('important_notes', [])])
    
    # Detect net revenue keywords
    net_revenue_keywords = ['net revenue', 'net earnings', 'profit', 'after costs', 'after deductions', 
                           'net profit', 'after refunds', 'after cancels', 'after cancellations']
    is_net_revenue_query = any(keyword in question.lower() for keyword in net_revenue_keywords)
    
    # Build comprehensive prompt
    net_revenue_instruction = ""
    if is_net_revenue_query:
        net_revenue_instruction = """
CRITICAL: User is asking about NET REVENUE. You MUST include ALL components:
- gross_revenue (Shipments)
- refund_amount (Refunds - absolute value)
- cancel_amount (Cancels - absolute value)
- free_repl_amount (FreeReplacement - absolute value)
- net_revenue = gross_revenue - refund_amount - cancel_amount - free_repl_amount

DO NOT generate SQL that only calculates refunds or only calculates revenue minus refunds.
The complete formula requires ALL 4 transaction types!

Example SQL structure:
SELECT 
  SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END) as gross_revenue,
  ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END)) as refund_amount,
  ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END)) as cancel_amount,
  ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END)) as free_repl_amount,
  (SUM(CASE WHEN transaction_type = 'Shipment' THEN revenue_amount ELSE 0 END)
   - ABS(SUM(CASE WHEN transaction_type = 'Refund' THEN revenue_amount ELSE 0 END))
   - ABS(SUM(CASE WHEN transaction_type = 'Cancel' THEN revenue_amount ELSE 0 END))
   - ABS(SUM(CASE WHEN transaction_type = 'FreeReplacement' THEN revenue_amount ELSE 0 END))
  ) as net_revenue
FROM sales
WHERE [date filters]

"""
    
    prompt = f"""You are a SQL expert. Generate a DuckDB SQL query to answer this business question.

DATABASE SCHEMA:
Table: {schema.get('table', 'sales')}
Columns:
{columns_desc if columns_desc else '  (No columns available)'}

CRITICAL RULES - READ CAREFULLY:
{important_notes}
{net_revenue_instruction}
QUESTION: {question}

REQUIREMENTS:
- Return ONLY the SQL query, no explanations, no markdown, no code blocks
- Use DuckDB SQL syntax (not MySQL, PostgreSQL, or SQL Server)
- Use double quotes for column names that contain spaces: "Invoice Date" not Invoice Date
- Use single quotes for string literals: 'Shipment' not "Shipment"
- For revenue: SUM the revenue column WHERE transaction_type = 'Shipment'
- Revenue is already in INR - DO NOT multiply by any number
- DO NOT use columns that are not in the schema above
- Add LIMIT 100 if query returns many rows
- CRITICAL DATE HANDLING:
  * Date columns are VARCHAR/TEXT - MUST cast to DATE before using date functions
  * For month extraction: EXTRACT(MONTH FROM CAST("Invoice Date" AS DATE)) or DATE_PART('month', CAST("Invoice Date" AS DATE))
  * For date filtering: WHERE CAST("Invoice Date" AS DATE) >= '2025-07-01'
  * NEVER use EXTRACT or DATE_PART directly on VARCHAR columns - always CAST first
  * CRITICAL: When user asks "in July" or "in month X", ALWAYS filter by YEAR AND MONTH using date range
  * Use: WHERE CAST("Invoice Date" AS DATE) >= '2025-07-01' AND CAST("Invoice Date" AS DATE) <= '2025-07-31'
  * NEVER use only EXTRACT(MONTH) = 7 without year filter - this includes ALL years and causes incorrect results
  * If year not specified, use current year (2025) as default

SQL QUERY (only the query, nothing else):"""
    
    # Try external LLMs first if user_id is provided
    if user_id:
        try:
            sql, error, provider, response_time, confidence = await generate_sql_with_external_llm(
                prompt, user_id
            )
            
            if sql and not error:
                # Post-process external LLM SQL (same as Ollama)
                sql = post_process_sql(sql, question, schema, source_file)
                if sql:
                    # Check confidence threshold for external LLMs
                    external_llm_threshold = 0.85
                    if confidence >= external_llm_threshold:
                        logger.info(f"✅ External LLM ({provider}) SQL generation successful (response_time: {response_time:.2f}s, confidence: {confidence:.2f})")
                        return sql, None, provider, response_time, confidence
                    else:
                        logger.warning(f"External LLM ({provider}) confidence {confidence:.2f} below threshold {external_llm_threshold}, falling back to Ollama")
                else:
                    logger.warning(f"External LLM SQL post-processing failed, falling back to Ollama")
            else:
                # Log the specific error from external LLM
                if error:
                    logger.warning(f"External LLM failed: {error}, falling back to Ollama")
                else:
                    logger.warning(f"External LLM returned no SQL, falling back to Ollama")
        except Exception as e:
            logger.error(f"External LLM call exception: {str(e)}, falling back to Ollama")
    
    # Fallback to Ollama
    logger.info(f"Ollama query started: {question}")
    sql, error, provider, response_time, confidence = generate_sql_with_ollama(prompt, question, schema, source_file, is_net_revenue_query)
    
    # Check confidence threshold for Ollama
    if sql and not error:
        ollama_threshold = 0.75
        if confidence >= ollama_threshold:
            logger.info(f"Ollama confidence {confidence:.2f} meets threshold {ollama_threshold}, returning result")
            return sql, None, provider, response_time, confidence
        else:
            logger.warning(f"Ollama confidence {confidence:.2f} below threshold {ollama_threshold}, but returning result anyway (Ollama naturally has lower confidence)")
            # Still return the result since Ollama naturally has lower confidence but answers can be useful
            return sql, None, provider, response_time, confidence
    
    # Return error case
    return sql, error, provider, response_time, confidence


def post_process_sql(
    sql: str,
    question: str,
    schema: Dict[str, Any],
    source_file: Optional[str] = None
) -> Optional[str]:
    """
    Post-process SQL query (fixes common mistakes)
    
    This is shared between Ollama and external LLMs.
    """
    if not sql:
        return None
    
    original_sql = sql
    
    # Basic SQL safety check
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT', 'CREATE', 'REPLACE']
    sql_upper = sql.upper().strip()
    
    for keyword in dangerous_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            logger.warning(f"SQL contains dangerous keyword: {keyword}")
            return None
    
    if not sql_upper.startswith('SELECT'):
        logger.warning("SQL does not start with SELECT")
        return None
    
    # Remove currency conversion multipliers
    sql = re.sub(r'\*\s*\d+\.?\d*\s*(?:--.*)?$', '', sql, flags=re.MULTILINE)
    sql = sql.strip()
    
    # Fix date function calls
    extract_pattern = r'EXTRACT\s*\(\s*(MONTH|YEAR|DAY|QUARTER|WEEK)\s+FROM\s+"([^"]+)"\s*\)'
    def fix_extract(match):
        extract_type = match.group(1)
        col_name = match.group(2)
        match_str = match.group(0)
        if f'CAST("{col_name}"' not in match_str:
            return f'EXTRACT({extract_type} FROM CAST("{col_name}" AS DATE))'
        return match.group(0)
    
    if re.search(extract_pattern, sql, re.IGNORECASE):
        sql = re.sub(extract_pattern, fix_extract, sql, flags=re.IGNORECASE)
    
    datepart_pattern = r'DATE_PART\s*\(\s*([^,]+?)\s*,\s*"([^"]+)"\s*\)'
    def fix_datepart(match):
        part_type = match.group(1).strip()
        col_name = match.group(2)
        match_str = match.group(0)
        if f'CAST("{col_name}"' not in match_str:
            return f'DATE_PART({part_type}, CAST("{col_name}" AS DATE))'
        return match.group(0)
    
    if re.search(datepart_pattern, sql, re.IGNORECASE):
        sql = re.sub(datepart_pattern, fix_datepart, sql, flags=re.IGNORECASE)
    
    # Fix month-only queries
    month_only_pattern = r"EXTRACT\s*\(\s*MONTH\s+FROM\s+CAST\([^)]+AS\s+DATE\)\s*\)\s*=\s*(\d+)"
    if re.search(month_only_pattern, sql, re.IGNORECASE):
        if 'EXTRACT(YEAR' not in sql.upper():
            match = re.search(month_only_pattern, sql, re.IGNORECASE)
            if match:
                month_num = int(match.group(1))
                date_col_match = re.search(r'EXTRACT\s*\(\s*MONTH\s+FROM\s+CAST\("([^"]+)"\s+AS\s+DATE\)', sql, re.IGNORECASE)
                if date_col_match:
                    date_col = date_col_match.group(1)
                    current_year = datetime.now().year
                    last_day = monthrange(current_year, month_num)[1]
                    date_range_filter = f'CAST("{date_col}" AS DATE) >= \'{current_year}-{month_num:02d}-01\' AND CAST("{date_col}" AS DATE) <= \'{current_year}-{month_num:02d}-{last_day}\''
                    extract_expr = f'EXTRACT(MONTH FROM CAST("{date_col}" AS DATE)) = {month_num}'
                    sql = sql.replace(extract_expr, date_range_filter)
    
    # Add source_file filter if needed
    if source_file:
        try:
            conn = get_connection()
            column_check = conn.execute("SELECT name FROM pragma_table_info('sales') WHERE name = 'source_file'").fetchdf()
            if not column_check.empty:
                source_count_sql = 'SELECT COUNT(DISTINCT source_file) as count FROM sales WHERE source_file IS NOT NULL'
                source_count_df = execute_query(source_count_sql)
                if not source_count_df.empty and source_count_df.iloc[0]['count'] > 1:
                    sql_upper_check = sql.upper()
                    if 'SOURCE_FILE' not in sql_upper_check:
                        # Find WHERE, GROUP BY, ORDER BY, and LIMIT positions
                        where_pos = sql_upper_check.find(' WHERE ')
                        group_by_pos = sql_upper_check.find(' GROUP BY ')
                        order_by_pos = sql_upper_check.find(' ORDER BY ')
                        limit_pos = sql_upper_check.find(' LIMIT ')
                        
                        if where_pos >= 0:
                            # WHERE clause exists - find the first clause after WHERE
                            insertion_point = -1
                            insertion_type = None
                            
                            # Check GROUP BY first (it comes before ORDER BY and LIMIT)
                            if group_by_pos > 0 and group_by_pos > where_pos:
                                insertion_point = group_by_pos
                                insertion_type = "GROUP BY"
                            # Then ORDER BY
                            elif order_by_pos > 0 and order_by_pos > where_pos:
                                insertion_point = order_by_pos
                                insertion_type = "ORDER BY"
                            # Finally LIMIT
                            elif limit_pos > 0 and limit_pos > where_pos:
                                insertion_point = limit_pos
                                insertion_type = "LIMIT"
                            
                            if insertion_point > 0:
                                # Insert AND before the first clause after WHERE
                                sql = sql[:insertion_point] + f" AND source_file = '{source_file}' " + sql[insertion_point:]
                            else:
                                # No clauses after WHERE - add AND at end
                                sql = sql.rstrip() + f" AND source_file = '{source_file}'"
                        elif group_by_pos > 0:
                            # No WHERE but GROUP BY exists - add WHERE before GROUP BY
                            sql = sql[:group_by_pos] + f" WHERE source_file = '{source_file}' " + sql[group_by_pos:]
                        elif order_by_pos > 0:
                            # No WHERE or GROUP BY but ORDER BY exists - add WHERE before ORDER BY
                            sql = sql[:order_by_pos] + f" WHERE source_file = '{source_file}' " + sql[order_by_pos:]
                        elif limit_pos > 0:
                            sql = sql[:limit_pos] + f" WHERE source_file = '{source_file}' " + sql[limit_pos:]
                        else:
                            sql = sql.rstrip() + f" WHERE source_file = '{source_file}'"
        except Exception as e:
            logger.warning(f"Could not add source_file filter: {e}")
    
    # Final safety check
    sql_upper = sql.upper().strip()
    if not sql_upper.startswith('SELECT'):
        return None
    
    return sql.strip()


def generate_sql_with_ollama(
    prompt: str,
    question: str,
    schema: Dict[str, Any],
    source_file: Optional[str] = None,
    is_net_revenue_query: bool = False
) -> Tuple[Optional[str], Optional[str], str, float, float]:
    """
    Generate SQL using Ollama (fallback)
    
    Returns:
        Tuple of (sql_query, error_message, provider, response_time, confidence)
    """
    start_time = time.time()
    
    try:
        # Call Ollama API
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent SQL
                "num_predict": 300,   # Allow longer queries
                "top_p": 0.9,
                "top_k": 40,
            }
            },
            timeout=settings.OLLAMA_TIMEOUT,
        )
        
        if response.status_code == 200:
            result = response.json()
            sql = result.get('response', '').strip()
            
            # Extract SQL if wrapped in code blocks
            if '```sql' in sql:
                sql = sql.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql:
                parts = sql.split('```')
                if len(parts) >= 2:
                    sql = parts[1].strip()
                    # Remove language identifier if present
                    if sql.startswith('sql'):
                        sql = sql[3:].strip()
            
            # Remove any leading/trailing whitespace and newlines
            sql = sql.strip()
            
            # Remove any trailing semicolons
            if sql.endswith(';'):
                sql = sql[:-1].strip()
            
            # Remove any comments (lines starting with --)
            lines = sql.split('\n')
            sql = '\n'.join([line for line in lines if not line.strip().startswith('--')]).strip()
            
            # Basic SQL safety check
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT', 'CREATE', 'REPLACE']
            sql_upper = sql.upper().strip()
            
            # Use word boundaries to match only whole words (not substrings)
            # This prevents false positives like 'FreeReplacement' matching 'REPLACE'
            for keyword in dangerous_keywords:
                # Match keyword as whole word (not substring)
                # Pattern: word boundary, keyword, word boundary (or end of string)
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, sql_upper):
                    return None, f"Query contains dangerous keyword: {keyword}"
            
            if not sql_upper.startswith('SELECT'):
                return None, "Query must start with SELECT"
            
            # Post-process: Fix common mistakes
            # Remove currency conversion multipliers
            # Remove patterns like "* 74.99" or "*74.99" (currency conversion)
            sql = re.sub(r'\*\s*\d+\.?\d*\s*(?:--.*)?$', '', sql, flags=re.MULTILINE)
            sql = sql.strip()
            
            # Fix date function calls that don't cast VARCHAR to DATE
            # The issue: EXTRACT(MONTH FROM "Invoice Date") fails because "Invoice Date" is VARCHAR
            # Fix: EXTRACT(MONTH FROM CAST("Invoice Date" AS DATE))
            
            original_sql = sql
            
            # Pattern 1: EXTRACT(MONTH|YEAR|DAY|QUARTER|WEEK FROM "column")
            # Replace with: EXTRACT(MONTH|YEAR|DAY|QUARTER|WEEK FROM CAST("column" AS DATE))
            extract_pattern = r'EXTRACT\s*\(\s*(MONTH|YEAR|DAY|QUARTER|WEEK)\s+FROM\s+"([^"]+)"\s*\)'
            def fix_extract(match):
                extract_type = match.group(1)
                col_name = match.group(2)
                # Check if this specific match already has CAST (check the full match)
                match_str = match.group(0)
                if f'CAST("{col_name}"' not in match_str:
                    return f'EXTRACT({extract_type} FROM CAST("{col_name}" AS DATE))'
                return match.group(0)
            
            if re.search(extract_pattern, sql, re.IGNORECASE):
                sql = re.sub(extract_pattern, fix_extract, sql, flags=re.IGNORECASE)
                logger.info("Post-processed: Fixed EXTRACT calls to include CAST")
            
            # Pattern 2: DATE_PART('month', "column") or DATE_PART('month', 'column')  
            # Replace with: DATE_PART('month', CAST("column" AS DATE))
            datepart_pattern = r'DATE_PART\s*\(\s*([^,]+?)\s*,\s*"([^"]+)"\s*\)'
            def fix_datepart(match):
                part_type = match.group(1).strip()
                col_name = match.group(2)
                # Check if this specific match already has CAST
                match_str = match.group(0)
                if f'CAST("{col_name}"' not in match_str:
                    return f'DATE_PART({part_type}, CAST("{col_name}" AS DATE))'
                return match.group(0)
            
            if re.search(datepart_pattern, sql, re.IGNORECASE):
                sql = re.sub(datepart_pattern, fix_datepart, sql, flags=re.IGNORECASE)
                logger.info("Post-processed: Fixed DATE_PART calls to include CAST")
            
            # Fix month-only queries that don't specify year (CRITICAL FIX)
            # Pattern: EXTRACT(MONTH FROM ...) = 7 without year filter
            # This causes data from multiple years to be included incorrectly
            month_only_pattern = r"EXTRACT\s*\(\s*MONTH\s+FROM\s+CAST\([^)]+AS\s+DATE\)\s*\)\s*=\s*(\d+)"
            if re.search(month_only_pattern, sql, re.IGNORECASE):
                # Check if year filter already exists
                if 'EXTRACT(YEAR' not in sql.upper() and 'EXTRACT\s*\(\s*YEAR' not in sql.upper():
                    # Extract month number
                    match = re.search(month_only_pattern, sql, re.IGNORECASE)
                    if match:
                        month_num = int(match.group(1))
                        
                        # Find the date column name from the EXTRACT expression
                        date_col_match = re.search(r'EXTRACT\s*\(\s*MONTH\s+FROM\s+CAST\("([^"]+)"\s+AS\s+DATE\)', sql, re.IGNORECASE)
                        if date_col_match:
                            date_col = date_col_match.group(1)
                            
                            # Assume current year (2025) if not specified
                            current_year = datetime.now().year
                            
                            # Calculate last day of month (handle different month lengths)
                            last_day = monthrange(current_year, month_num)[1]
                            
                            # Build date range filter
                            date_range_filter = f'CAST("{date_col}" AS DATE) >= \'{current_year}-{month_num:02d}-01\' AND CAST("{date_col}" AS DATE) <= \'{current_year}-{month_num:02d}-{last_day}\''
                            
                            # Check if LIMIT clause exists - we need to preserve it
                            limit_match = re.search(r'\s+LIMIT\s+\d+', sql, re.IGNORECASE)
                            limit_clause = limit_match.group(0) if limit_match else ''
                            
                            # Replace the EXTRACT(MONTH) = X with the date range
                            # Try to replace in WHERE clause context
                            extract_expr = f'EXTRACT(MONTH FROM CAST("{date_col}" AS DATE)) = {month_num}'
                            extract_expr_spaced = f'EXTRACT(MONTH FROM CAST("{date_col}" AS DATE)) = {month_num}'
                            
                            # Replace the expression with date range filter
                            # Handle case where LIMIT might be directly after the expression
                            if extract_expr + limit_clause in sql:
                                # Replace including LIMIT, then add LIMIT back after date range
                                sql = sql.replace(extract_expr + limit_clause, date_range_filter + limit_clause)
                            elif extract_expr in sql:
                                sql = sql.replace(extract_expr, date_range_filter)
                            elif extract_expr_spaced + limit_clause in sql:
                                sql = sql.replace(extract_expr_spaced + limit_clause, date_range_filter + limit_clause)
                            elif extract_expr_spaced in sql:
                                sql = sql.replace(extract_expr_spaced, date_range_filter)
                            else:
                                # Try regex replacement - be careful not to match LIMIT as part of the expression
                                pattern = rf'EXTRACT\s*\(\s*MONTH\s+FROM\s+CAST\("{re.escape(date_col)}"\s+AS\s+DATE\)\s*\)\s*=\s*{month_num}(?=\s+LIMIT|\s*$|\))'
                                sql = re.sub(pattern, date_range_filter, sql, flags=re.IGNORECASE)
                            
                            logger.info(f"Post-processed: Replaced month-only filter (month={month_num}) with date range filter for {current_year}")
                            logger.info(f"Date range: {current_year}-{month_num:02d}-01 to {current_year}-{month_num:02d}-{last_day}")
            
            # Post-process: Add source_file filter if multiple sources exist (prevent multi-source data mixing)
            # Check if source_file column exists and if there are multiple source files
            try:
                # Quick check: does source_file column exist?
                conn = get_connection()
                column_check = conn.execute("SELECT name FROM pragma_table_info('sales') WHERE name = 'source_file'").fetchdf()
                
                if not column_check.empty:
                    # Check how many source files exist
                    source_count_sql = 'SELECT COUNT(DISTINCT source_file) as count FROM sales WHERE source_file IS NOT NULL'
                    source_count_df = execute_query(source_count_sql)
                    
                    if not source_count_df.empty and source_count_df.iloc[0]['count'] > 1:
                        # Multiple sources exist - check if source_file filter is in query
                        sql_upper_check = sql.upper()
                        if 'SOURCE_FILE' not in sql_upper_check and 'source_file' not in sql:
                            # Get which source file to use
                            target_source_file = None
                            
                            if source_file:
                                target_source_file = source_file
                                logger.info(f"Using specified source_file: {target_source_file}")
                            else:
                                # Try to infer from question (e.g., "July" -> "JulyMonthly.csv")
                                # Note: question variable is from function parameter, available in this scope
                                question_lower = question.lower() if isinstance(question, str) else str(question).lower()
                                
                                # Get all source files
                                all_sources_sql = 'SELECT DISTINCT source_file FROM sales WHERE source_file IS NOT NULL'
                                all_sources_df = execute_query(all_sources_sql)
                                
                                if not all_sources_df.empty:
                                    source_files = all_sources_df['source_file'].tolist()
                                    
                                    # Try to match question context (e.g., "july" -> "JulyMonthly.csv")
                                    matched_source = None
                                    if 'july' in question_lower or 'jul' in question_lower:
                                        matched_source = next((s for s in source_files if 'july' in s.lower() or 'jul' in s.lower()), None)
                                    elif 'august' in question_lower or 'aug' in question_lower:
                                        matched_source = next((s for s in source_files if 'aug' in s.lower()), None)
                                    elif 'september' in question_lower or 'sept' in question_lower or 'sep' in question_lower:
                                        matched_source = next((s for s in source_files if 'sep' in s.lower() or 'sept' in s.lower()), None)
                                    
                                    if matched_source:
                                        target_source_file = matched_source
                                        logger.info(f"Inferred source_file from question context: {matched_source}")
                                    else:
                                        # Fallback: Get most recent source file by loaded_at
                                        latest_source_sql = """
                                        SELECT source_file, MAX(loaded_at) as last_loaded
                                        FROM sales
                                        WHERE source_file IS NOT NULL
                                        GROUP BY source_file
                                        ORDER BY last_loaded DESC
                                        LIMIT 1
                                        """
                                        latest_source_df = execute_query(latest_source_sql)
                                        if not latest_source_df.empty:
                                            target_source_file = latest_source_df.iloc[0]['source_file']
                                            logger.info(f"Using most recent source_file: {target_source_file}")
                            
                            # Add source_file filter to SQL
                            if target_source_file:
                                # Find WHERE, GROUP BY, ORDER BY, and LIMIT clause positions
                                sql_upper = sql.upper()
                                where_pos = -1
                                for pattern in [' WHERE ', 'WHERE ']:
                                    pos = sql_upper.find(pattern)
                                    if pos != -1:
                                        where_pos = pos + len(pattern) - 1
                                        break
                                
                                group_by_pos = -1
                                for pattern in [' GROUP BY ', 'GROUP BY ']:
                                    pos = sql_upper.find(pattern)
                                    if pos != -1:
                                        group_by_pos = pos
                                        break
                                
                                order_by_pos = -1
                                for pattern in [' ORDER BY ', 'ORDER BY ']:
                                    pos = sql_upper.find(pattern)
                                    if pos != -1:
                                        order_by_pos = pos
                                        break
                                
                                limit_pos = -1
                                for pattern in [' LIMIT ', 'LIMIT ']:
                                    pos = sql_upper.find(pattern)
                                    if pos != -1:
                                        limit_pos = pos
                                        break
                                
                                # Determine insertion point: before GROUP BY, ORDER BY, or LIMIT (whichever comes first after WHERE)
                                if where_pos >= 0:
                                    # WHERE clause exists - find the first clause after WHERE
                                    insertion_point = -1
                                    insertion_type = None
                                    
                                    # Check GROUP BY first (it comes before ORDER BY and LIMIT)
                                    if group_by_pos > 0 and group_by_pos > where_pos:
                                        insertion_point = group_by_pos
                                        insertion_type = "GROUP BY"
                                    # Then ORDER BY
                                    elif order_by_pos > 0 and order_by_pos > where_pos:
                                        insertion_point = order_by_pos
                                        insertion_type = "ORDER BY"
                                    # Finally LIMIT
                                    elif limit_pos > 0 and limit_pos > where_pos:
                                        insertion_point = limit_pos
                                        insertion_type = "LIMIT"
                                    
                                    if insertion_point > 0:
                                        # Insert AND before the first clause after WHERE
                                        sql = sql[:insertion_point] + f" AND source_file = '{target_source_file}' " + sql[insertion_point:]
                                        logger.info(f"✅ Post-processed: Added source_file filter before {insertion_type} (using: {target_source_file})")
                                    else:
                                        # No clauses after WHERE - add AND at end
                                        sql = sql.rstrip() + f" AND source_file = '{target_source_file}'"
                                        logger.info(f"✅ Post-processed: Added source_file filter at end of WHERE clause (using: {target_source_file})")
                                elif group_by_pos > 0:
                                    # No WHERE but GROUP BY exists - add WHERE before GROUP BY
                                    sql = sql[:group_by_pos] + f" WHERE source_file = '{target_source_file}' " + sql[group_by_pos:]
                                    logger.info(f"✅ Post-processed: Added source_file WHERE clause before GROUP BY (using: {target_source_file})")
                                elif order_by_pos > 0:
                                    # No WHERE or GROUP BY but ORDER BY exists - add WHERE before ORDER BY
                                    sql = sql[:order_by_pos] + f" WHERE source_file = '{target_source_file}' " + sql[order_by_pos:]
                                    logger.info(f"✅ Post-processed: Added source_file WHERE clause before ORDER BY (using: {target_source_file})")
                                elif limit_pos > 0:
                                    # No WHERE, GROUP BY, or ORDER BY - add WHERE before LIMIT
                                    sql = sql[:limit_pos] + f" WHERE source_file = '{target_source_file}' " + sql[limit_pos:]
                                    logger.info(f"✅ Post-processed: Added source_file WHERE clause before LIMIT (using: {target_source_file})")
                                else:
                                    # No WHERE, GROUP BY, ORDER BY, or LIMIT - add WHERE at end
                                    sql = sql.rstrip() + f" WHERE source_file = '{target_source_file}'"
                                    logger.info(f"✅ Post-processed: Added source_file WHERE clause at end (using: {target_source_file})")
                            else:
                                logger.warning("⚠️ Multiple source files detected but could not determine which one to filter by")
                        else:
                            logger.debug("source_file filter already present in query")
                    else:
                        logger.debug(f"Single source file or no source files ({source_count_df.iloc[0]['count'] if not source_count_df.empty else 0}) - no filter needed")
            except Exception as e:
                logger.warning(f"⚠️ Could not add source_file filter (non-critical): {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
            
            # Final safety check: Fix any malformed date strings with LIMIT
            # Pattern: 'YYYY-MM-DD LIMIT -> 'YYYY-MM-DD' LIMIT
            # This fixes cases where LIMIT was accidentally placed inside the date string
            sql = re.sub(r"<=\s*'(\d{4}-\d{2}-\d{2})(\s+LIMIT\s+\d+)", r"<= '\1'\2", sql, flags=re.IGNORECASE)
            sql = re.sub(r">=\s*'(\d{4}-\d{2}-\d{2})(\s+LIMIT\s+\d+)", r">= '\1'\2", sql, flags=re.IGNORECASE)
            sql = re.sub(r"=\s*'(\d{4}-\d{2}-\d{2})(\s+LIMIT\s+\d+)", r"= '\1'\2", sql, flags=re.IGNORECASE)
            
            # Also fix any date strings that might be missing closing quotes before LIMIT
            sql = re.sub(r"('(?:\d{4}-\d{2}-\d{2}))(\s+LIMIT\s+\d+)", r"\1'\2", sql, flags=re.IGNORECASE)
            
            # Post-process: Detect and warn about incomplete net revenue queries
            # This helps debug if the AI still generates incomplete SQL despite the prompt
            # Note: is_net_revenue_query is passed as parameter
            if is_net_revenue_query:
                # Check if SQL includes all required components
                has_shipment = bool(re.search(r"transaction_type\s*=\s*['\"]Shipment['\"]", sql, re.IGNORECASE))
                has_refund = bool(re.search(r"transaction_type\s*=\s*['\"]Refund['\"]", sql, re.IGNORECASE))
                has_cancel = bool(re.search(r"transaction_type\s*=\s*['\"]Cancel['\"]", sql, re.IGNORECASE))
                has_free_repl = bool(re.search(r"transaction_type\s*=\s*['\"]FreeReplacement['\"]", sql, re.IGNORECASE))
                
                # Check if net_revenue calculation exists
                has_net_revenue_calc = bool(re.search(r'\bnet_revenue\b|\bnet_rev\b', sql, re.IGNORECASE))
                
                # Log warning if components are missing
                if has_net_revenue_calc:
                    missing_components = []
                    if not has_shipment:
                        missing_components.append('Shipment')
                    if not has_refund:
                        missing_components.append('Refund')
                    if not has_cancel:
                        missing_components.append('Cancel')
                    if not has_free_repl:
                        missing_components.append('FreeReplacement')
                    
                    if missing_components:
                        logger.warning(f"⚠️ Net revenue query detected but missing components: {', '.join(missing_components)}")
                        logger.warning(f"Generated SQL may be incomplete for net revenue calculation")
            
            if sql != original_sql:
                logger.info(f"SQL post-processed. Original: {original_sql[:150]}...")
                logger.info(f"SQL post-processed. Fixed: {sql[:150]}...")
            
            # Log the final SQL for debugging
            logger.info(f"Final generated SQL: {sql}")
            
            # Final validation check (post_process_sql is redundant here since we already did all processing inline)
            # Just ensure SQL is valid before returning
            if not sql or not sql.strip():
                response_time = time.time() - start_time
                logger.error(f"Ollama execution result: failure - SQL is empty after processing")
                logger.error(f"Question was: {question}")
                return None, "SQL generation failed - empty result", 'ollama', response_time, 0.0
            
            sql_upper = sql.upper().strip()
            if not sql_upper.startswith('SELECT'):
                response_time = time.time() - start_time
                logger.error(f"Ollama execution result: failure - SQL does not start with SELECT")
                logger.error(f"Question was: {question}")
                logger.error(f"Generated SQL (first 200 chars): {sql[:200]}")
                return None, "SQL generation failed - invalid query format", 'ollama', response_time, 0.0
            
            response_time = time.time() - start_time
            confidence = calculate_ollama_confidence(sql, response_time)
            
            # Log Ollama SQL generation and confidence
            logger.info(f"Ollama SQL generated: {sql}")
            logger.info(f"Ollama confidence: {confidence:.2f}")
            
            # Check confidence threshold (0.75 for Ollama)
            confidence_threshold = 0.75
            logger.info(f"Confidence threshold: {confidence_threshold}")
            if confidence >= confidence_threshold:
                logger.info(f"Result: accepted (confidence {confidence:.2f} >= threshold {confidence_threshold})")
                logger.info(f"Ollama execution result: success")
            else:
                logger.warning(f"Result: below threshold (confidence {confidence:.2f} < threshold {confidence_threshold})")
                logger.warning(f"Ollama execution result: low confidence (but may still be useful)")
            
            return sql, None, 'ollama', response_time, confidence
        
        else:
            response_time = time.time() - start_time
            error = f"Ollama API error: {response.status_code}"
            logger.error(f"Ollama execution result: failure - {error}")
            return None, error, 'ollama', response_time, 0.0
    
    except Exception as e:
        response_time = time.time() - start_time
        error = f"LLM generation failed: {str(e)}"
        logger.error(f"Ollama execution result: failure - {error}")
        return None, error, 'ollama', response_time, 0.0


def validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL query safety
    
    Args:
        sql: SQL query string
    
    Returns:
        Tuple of (is_safe, error_message)
    """
    logger.info(f"Validating SQL: {sql}")
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT
    if not sql_upper.startswith('SELECT'):
        error_msg = "Query must start with SELECT"
        logger.warning(f"Validation result: failed - {error_msg}")
        logger.warning(f"Validation errors: {error_msg}")
        return False, error_msg
    
    # Block dangerous keywords (using word boundaries to avoid false positives)
    # Example: 'FreeReplacement' should NOT match 'REPLACE'
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT', 'CREATE', 'REPLACE']
    for keyword in dangerous_keywords:
        # Match keyword as whole word (not substring)
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            error_msg = f"Query contains dangerous keyword: {keyword}"
            logger.warning(f"Validation result: failed - {error_msg}")
            logger.warning(f"Validation errors: {error_msg}")
            return False, error_msg
    
    logger.info(f"Validation result: passed")
    return True, None


def format_query_response(
    question: str,
    sql: str,
    result_df,
) -> str:
    """
    Format database query results into natural language response
    
    Args:
        question: Original user question
        sql: SQL query that was executed
        result_df: Query results as DataFrame
    
    Returns:
        Formatted natural language response
    """
    import pandas as pd
    
    # Simple formatting (we'll improve this later with better prompts)
    if result_df.empty:
        return "No results found for your query."
    
    # If single value result (e.g., SUM, COUNT)
    if len(result_df) == 1 and len(result_df.columns) == 1:
        value = result_df.iloc[0, 0]
        
        # Format Indian currency if it looks like revenue
        if 'revenue' in question.lower() or 'sales' in question.lower():
            formatted_value = f"₹{value:,.2f}"
        else:
            formatted_value = f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
        
        return f"The answer is: {formatted_value}"
    
    # If multiple rows (e.g., top products)
    elif len(result_df) > 1:
        # Return first few rows as text
        response = "Here are the results:\n\n"
        for idx, row in result_df.head(10).iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            response += f"- {row_text}\n"
        
        if len(result_df) > 10:
            response += f"\n(Showing top 10 of {len(result_df)} results)"
        
        return response
    
    else:
        return str(result_df.to_dict('records'))
