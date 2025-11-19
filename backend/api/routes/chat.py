"""
AI Chat API Endpoints - Natural language query interface

Extracted from legacy ai_assistant.py
"""
from fastapi import APIRouter, HTTPException, Header
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
    provider: Optional[str] = None  # 'openai', 'anthropic', 'ollama', or None
    confidence: Optional[float] = None  # Confidence score (0.0-1.0)
    suggestion: Optional[str] = None  # Suggestion message (e.g., to add API key)


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
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
    
    # Get user ID (optional for now, but required for external LLM)
    current_user_id = x_user_id.strip() if x_user_id and x_user_id.strip() else None
    
    # Check API key status for logging
    has_openai_key = False
    has_anthropic_key = False
    if current_user_id:
        try:
            from services.api_key_service import APIKeyService
            has_openai_key = APIKeyService.get_api_key(current_user_id, 'openai', decrypt=False) is not None
            has_anthropic_key = APIKeyService.get_api_key(current_user_id, 'anthropic', decrypt=False) is not None
        except Exception as e:
            logger.warning(f"Error checking API keys for user_id={current_user_id}: {e}")
    
    logger.info(f"chat_ask: user_id={current_user_id}, has_openai_key={has_openai_key}, has_anthropic_key={has_anthropic_key}")
    
    # Check Ollama connection (needed as fallback)
    ollama_available = check_ollama_connection()
    if not ollama_available:
        # Only fail if user has no API keys
        if current_user_id:
            if not has_openai_key and not has_anthropic_key:
                logger.warning(f"chat_ask: Ollama unavailable and no API keys for user_id={current_user_id}")
                return ChatResponse(
                    answer="AI service (Ollama) is not available and no external API keys found. Please start Ollama server or add an API key in Settings.",
                    error="Ollama connection failed and no API keys available"
                )
        else:
            logger.warning("chat_ask: Ollama unavailable and no user_id provided")
            return ChatResponse(
                answer="AI service (Ollama) is not available. Please start Ollama server or add an API key in Settings.",
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
        
        # Generate SQL from question (tries external LLMs first, then Ollama)
        sql, error, provider, sql_gen_time, confidence = await generate_sql_from_question(
            request.question,
            schema,
            user_id=current_user_id
        )
        
        if error or not sql:
            return ChatResponse(
                answer="I couldn't understand your question. Please try rephrasing it.",
                error=error or "SQL generation failed",
                provider=provider,
                confidence=confidence,
            )
        
        # Validate SQL safety
        is_safe, safety_error = validate_sql_safety(sql)
        if not is_safe:
            logger.warning(f"SQL safety validation failed: {safety_error}")
            return ChatResponse(
                answer="I couldn't generate a safe query for your question. Please try rephrasing.",
                sql=sql,
                error=safety_error or "SQL safety validation failed",
                provider=provider,
                confidence=confidence,
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
            
            # Generate suggestion if confidence is low or response time is high
            suggestion = None
            if provider == 'ollama':
                if confidence < 0.7 or sql_gen_time > 30:
                    suggestion = "For better AI responses, consider adding an OpenAI or Anthropic API key in Settings."
            
            # Log metrics (masked API key if used)
            if provider in ['openai', 'anthropic']:
                from services.api_key_service import APIKeyService
                key_info = APIKeyService.get_api_key(current_user_id, provider, decrypt=False)
                masked_key = key_info['key'] if key_info else "N/A"
                logger.info(f"✅ Used {provider} API key: {masked_key} (response_time: {sql_gen_time:.2f}s, confidence: {confidence:.2f})")
            else:
                logger.info(f"✅ Used {provider} (response_time: {sql_gen_time:.2f}s, confidence: {confidence:.2f})")
            
            return ChatResponse(
                answer=answer,
                sql=sql,
                data=data,
                execution_time=round(execution_time, 3),
                provider=provider,
                confidence=round(confidence, 2),
                suggestion=suggestion,
            )
        
        except Exception as query_error:
            error_str = str(query_error)
            logger.error(f"Query execution failed: {error_str}")
            logger.error(f"Failed SQL query: {sql}")
            
            # Parse error message to provide user-friendly feedback
            user_friendly_message = "Sorry, I generated a query but it failed to execute. Please try a different question."
            
            # Check for common SQL errors and provide better messages
            if "syntax error" in error_str.lower() or "parser error" in error_str.lower():
                user_friendly_message = "I generated a query with a syntax error. Please try rephrasing your question or ask something simpler."
            elif "column" in error_str.lower() and "not found" in error_str.lower():
                user_friendly_message = "The query referenced a column that doesn't exist in your data. Please try a different question."
            elif "type mismatch" in error_str.lower() or "cannot cast" in error_str.lower():
                user_friendly_message = "There was a data type issue with the query. Please try rephrasing your question."
            
            return ChatResponse(
                answer=user_friendly_message,
                sql=sql,
                error=error_str,
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
