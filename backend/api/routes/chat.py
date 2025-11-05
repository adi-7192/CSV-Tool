"""
AI Chat API Endpoints - Natural language query interface

Extracted from legacy ai_assistant.py
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import time

from core.ai_service import (
    check_ollama_connection,
    generate_sql_from_question,
    validate_sql_safety,
    get_database_schema,
    format_query_response,
)
from core.database import execute_query, table_exists

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Ask a natural language question about the data
    
    Request body:
        - question: Natural language question (e.g., "What is my total revenue?")
        - context: Optional context (date filters, etc.)
    
    Returns:
        - answer: Natural language answer
        - sql: SQL query that was executed
        - data: Query results (first 100 rows)
        - execution_time: Query execution time in seconds
    """
    
    # Check if database has data
    if not table_exists('sales'):
        return ChatResponse(
            answer="No data available. Please upload a CSV file first.",
            error="No data in database"
        )
    
    # Check Ollama connection
    if not check_ollama_connection():
        return ChatResponse(
            answer="AI service (Ollama) is not available. Please start Ollama server.",
            error="Ollama connection failed"
        )
    
    try:
        start_time = time.time()
        
        # Get database schema
        schema = get_database_schema()
        
        if not schema.get('columns'):
            return ChatResponse(
                answer="Database schema could not be retrieved.",
                error="Schema retrieval failed"
            )
        
        # Generate SQL from question
        sql, error = generate_sql_from_question(request.question, schema)
        
        if error or not sql:
            return ChatResponse(
                answer="I couldn't understand your question. Please try rephrasing it.",
                error=error or "SQL generation failed",
            )
        
        # Validate SQL safety
        is_safe, safety_error = validate_sql_safety(sql)
        if not is_safe:
            logger.warning(f"SQL safety validation failed: {safety_error}")
            return ChatResponse(
                answer="I couldn't generate a safe query for your question. Please try rephrasing.",
                sql=sql,
                error=safety_error or "SQL safety validation failed",
            )
        
        # Execute SQL query
        try:
            result_df = execute_query(sql)
            
            # Limit results for response
            if len(result_df) > 100:
                result_df = result_df.head(100)
                logger.info(f"Limited results to 100 rows (total: {len(result_df)} rows)")
            
            execution_time = time.time() - start_time
            
            # Format response
            answer = format_query_response(request.question, sql, result_df)
            
            # Prepare data response
            if result_df.empty:
                data = None
            elif len(result_df) == 1 and len(result_df.columns) == 1:
                # Single value result
                value = result_df.iloc[0, 0]
                data = {"value": float(value) if isinstance(value, (int, float)) else str(value)}
            else:
                # Multiple rows
                data = {
                    "rows": result_df.to_dict('records'),
                    "count": len(result_df)
                }
            
            return ChatResponse(
                answer=answer,
                sql=sql,
                data=data,
                execution_time=round(execution_time, 3),
            )
        
        except Exception as query_error:
            logger.error(f"Query execution failed: {str(query_error)}")
            return ChatResponse(
                answer="Sorry, I generated a query but it failed to execute. Please try a different question.",
                sql=sql,
                error=str(query_error),
            )
    
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_suggested_questions():
    """
    Get suggested questions user can ask
    """
    suggestions = [
        "What is my total revenue?",
        "Show me top 10 products by revenue",
        "Which region has the highest sales?",
        "What were my refunds last month?",
        "Compare August vs September revenue",
        "Show me products with declining sales",
        "What is my average order value?",
        "How many orders did I have last week?",
        "What is the total number of shipments?",
        "Which products had the most units sold?",
    ]
    
    return {"suggestions": suggestions}
