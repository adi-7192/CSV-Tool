"""
AI Service - Ollama LLM integration for natural language queries

Extracted from legacy/ai_assistant.py
"""
import requests
import json
import logging
from typing import Dict, Optional, Tuple
from core.config import settings
from core.database import execute_query, get_connection

logger = logging.getLogger(__name__)


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


def get_database_schema() -> Dict[str, any]:
    """
    Get database schema information (columns, sample data, statistics)
    
    Returns:
        Dictionary with schema information
    """
    try:
        conn = get_connection()
        
        # Check if sales table exists
        tables = conn.execute("SHOW TABLES").fetchdf()
        if 'sales' not in tables['name'].values:
            return {'columns': [], 'sample_data': [], 'statistics': {}}
        
        # Get column information
        columns_df = conn.execute("DESCRIBE sales").fetchdf()
        columns = columns_df['column_name'].tolist()
        
        # Get sample data (first 5 rows)
        sample_df = execute_query("SELECT * FROM sales LIMIT 5")
        sample_data = sample_df.to_dict('records') if not sample_df.empty else []
        
        # Get basic statistics
        stats = {}
        if sample_df.shape[0] > 0:
            row_count_df = execute_query("SELECT COUNT(*) as count FROM sales")
            stats['row_count'] = int(row_count_df['count'].iloc[0]) if not row_count_df.empty else 0
        
        return {
            'columns': columns,
            'sample_data': sample_data,
            'statistics': stats
        }
    except Exception as e:
        logger.error(f"Error getting database schema: {e}")
        return {'columns': [], 'sample_data': [], 'statistics': {}}


def generate_sql_from_question(
    question: str,
    schema: Optional[Dict[str, any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate SQL query from natural language question using Ollama LLM
    
    Args:
        question: Natural language question
        schema: Optional database schema information (if None, will fetch)
    
    Returns:
        Tuple of (sql_query, error_message)
        If successful: (sql_string, None)
        If failed: (None, error_message)
    """
    # Get schema if not provided
    if schema is None:
        schema = get_database_schema()
    
    columns = schema.get('columns', [])
    
    # Build prompt with schema context
    prompt = f"""You are a SQL expert. Generate a DuckDB SQL query to answer this question.
Database Schema:
Table: sales
Columns: {', '.join(columns)}

Question: {question}

Instructions:
- Return ONLY the SQL query, no explanations
- Use DuckDB syntax
- Use double quotes around column names (e.g., "Invoice Date")
- Use proper date functions (CURRENT_DATE, DATE_TRUNC, INTERVAL)
- Filter by transaction_type for revenue (use 'Shipment' for revenue)
- Format currency as Indian Rupees
- Be concise but accurate
- Only use SELECT statements (never INSERT, UPDATE, DELETE, DROP, TRUNCATE, ALTER)
- Add LIMIT clause if querying many rows

SQL Query:"""
    
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
                    "num_predict": 200,   # Limit response length
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
                sql = sql.split('```')[1].split('```')[0].strip()
            
            # Basic SQL safety check
            dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT']
            sql_upper = sql.upper()
            if any(keyword in sql_upper for keyword in dangerous_keywords):
                return None, "Query contains dangerous keywords"
            
            if not sql_upper.startswith('SELECT'):
                return None, "Query must start with SELECT"
            
            # Clean up SQL
            sql = sql.strip()
            if sql.endswith(';'):
                sql = sql[:-1]
            
            return sql, None
        
        else:
            error = f"Ollama API error: {response.status_code}"
            logger.error(error)
            return None, error
    
    except Exception as e:
        error = f"LLM generation failed: {str(e)}"
        logger.error(error)
        return None, error


def validate_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL query safety
    
    Args:
        sql: SQL query string
    
    Returns:
        Tuple of (is_safe, error_message)
    """
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT
    if not sql_upper.startswith('SELECT'):
        return False, "Query must start with SELECT"
    
    # Block dangerous keywords
    dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT', 'CREATE', 'REPLACE']
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False, f"Query contains dangerous keyword: {keyword}"
    
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
