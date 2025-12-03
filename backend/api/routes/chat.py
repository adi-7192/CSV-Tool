"""
AI Chat API Endpoints - Natural language query interface

Hybrid Architecture with ChromaDB RAG:
1. Intent classification (METRIC, ADVISORY, EXPLORATION, CONVERSATIONAL)
2. ChromaDB semantic search for relevant business rules and schema
3. METRIC: Safe SQL templates with dynamic column detection
4. ADVISORY: Comprehensive data + RAG context + Gemini analysis
5. EXPLORATION: RAG-enhanced LLM SQL generation with validation
6. Response generator with data grounding (no hallucination)
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
)
from core.database import execute_query, table_exists
from services.intent_classifier import classify_intent, get_intent_summary
from services.metric_queries import (
    execute_metric_query,
    query_advisory_data,
    query_net_revenue,
    QueryResult,
)
from services.response_generator import (
    generate_response_with_gemini,
    generate_conversational_response,
    format_metric_response,
    format_advisory_response,
)
from services.rag_service import retrieve_context, enhance_prompt_with_rag
from services.sql_validator import validate_sql_columns, auto_correct_sql
from services.embedding_service import index_all_documents, get_collection
from services.metrics_registry import find_matching_function, get_function_call_params, call_metric_function

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
    provider: Optional[str] = None
    confidence: Optional[float] = None
    suggestion: Optional[str] = None


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
):
    """
    Ask a natural language question about the data.
    
    Flow:
    1. Classify intent (METRIC, ADVISORY, EXPLORATION)
    2. METRIC: Use safe SQL templates (DEFAULT - no free-form SQL)
    3. ADVISORY: Query comprehensive data, use Gemini for analysis
    4. EXPLORATION: Fall back to LLM SQL generation
    5. Generate natural language response
    """
    
    # Check if database has data
    if not table_exists('sales'):
        return ChatResponse(
            answer="No data available. Please upload a CSV file first.",
            error="No data in database"
        )
    
    # Get user ID
    current_user_id = x_user_id.strip() if x_user_id and x_user_id.strip() else 'user-123'
    
    # Check API key status
    has_gemini_key = False
    if current_user_id:
        try:
            from services.api_key_service import APIKeyService
            gemini_key = APIKeyService.get_api_key(current_user_id, 'gemini', decrypt=False)
            has_gemini_key = gemini_key is not None and gemini_key.get('enabled', True)
        except Exception as e:
            logger.warning(f"Error checking API keys: {e}")
    
    # Extract date range from context
    start_date = None
    end_date = None
    if request.context:
        start_date = request.context.get('start_date')
        end_date = request.context.get('end_date')
        if start_date and end_date:
            logger.info(f"Date context: {start_date} to {end_date}")
    
        try:
            start_time = time.time()

            # Step 0: Validate question before processing
            from services.intent_classifier import is_valid_question
            is_valid, validation_error = is_valid_question(request.question)
            if not is_valid:
                logger.warning(f"Invalid question detected: '{request.question}' - {validation_error}")
                return ChatResponse(
                    answer=validation_error or "Your question doesn't make sense. Please ask a meaningful question about your sales data.",
                    sql="",
                    data=None,
                    execution_time=0.001,
                    provider="validation",
                    confidence=1.0,
                    error=validation_error or "Invalid question"
                )

            # Step 1: Classify intent
            intent = classify_intent(request.question)
            logger.info(f"Intent: {get_intent_summary(intent)}")
            
            # Step 1.5: Check metrics registry FIRST (before SQL generation)
            # This ensures we use proven backend functions instead of generating SQL
            matched_function = find_matching_function(request.question)
            
            if matched_function:
                logger.info(f"✅ Registry match found: {matched_function.function_name}")
                
                # Build function parameters
                func_params = get_function_call_params(
                    matched_function,
                    request.question,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Check if we have required parameters
                if func_params is None:
                    logger.warning(f"Could not build required params for {matched_function.function_name}, falling back to normal flow")
                else:
                    missing_params = [p for p in matched_function.required_params if p not in func_params]
                    if missing_params:
                        logger.warning(f"Missing required params {missing_params} for {matched_function.function_name}, falling back to normal flow")
                    else:
                        # Call the function
                        try:
                            function_result = call_metric_function(matched_function.function_name, func_params)
                            
                            if 'error' in function_result:
                                logger.warning(f"Function call failed: {function_result['error']}, falling back to normal flow")
                            else:
                                # Success! Format response with LLM
                                logger.info(f"✅ Function call successful, formatting response")
                                
                                # Create QueryResult-like structure for response generator
                                query_result = QueryResult(
                                    success=True,
                                    data=function_result,
                                    sql=f"-- Called function: {matched_function.function_name}"
                                )
                                
                                # Generate natural language response
                                if has_gemini_key:
                                    answer = await generate_response_with_gemini(
                                        request.question,
                                        query_result,
                                        "METRIC",
                                        current_user_id
                                    )
                                    provider = "gemini"
                                else:
                                    answer = format_metric_response(
                                        request.question,
                                        query_result,
                                        "METRIC",
                                        matched_function.return_type
                                    )
                                    provider = "function"
                                
                                execution_time = time.time() - start_time
                                
                                return ChatResponse(
                                    answer=answer,
                                    sql=query_result.sql,
                                    data=function_result,
                                    execution_time=round(execution_time, 3),
                                    provider=provider,
                                    confidence=1.0,  # Functions are always accurate
                                )
                                
                        except Exception as func_error:
                            logger.error(f"Error calling metric function: {func_error}", exc_info=True)
                            # Fall through to normal flow
            
            # Step 2: Route based on intent type (if no registry match or function call failed)
            
            # CONVERSATIONAL: Handle greetings and general chat
            if intent.type == "CONVERSATIONAL":
                logger.info("Using CONVERSATIONAL path")
                
                # Use Gemini for conversational response if available
                if has_gemini_key:
                    answer = await generate_conversational_response(
                        request.question,
                        current_user_id
                    )
                    provider = "gemini"
                else:
                    # Simple fallback responses
                    question_lower = request.question.lower().strip()
                if any(g in question_lower for g in ['hello', 'hi', 'hey', 'hola', 'howdy']):
                    answer = "Hello! I'm your AI data analyst. I can help you explore your sales data. Try asking me things like:\n\n• What's my total revenue?\n• Show me top 10 products\n• Which items have high refund rates?\n• Where should I focus my business?"
                elif 'what can you do' in question_lower or 'help' in question_lower:
                    answer = "I can help you analyze your sales data! Here's what I can do:\n\n📊 **Metrics**: Revenue, refunds, order counts, averages\n📈 **Rankings**: Top products, best regions, worst performers\n🔍 **Analysis**: Trends, comparisons, problem areas\n💡 **Advice**: Business recommendations based on your data\n\nJust ask me in plain English!"
                elif any(g in question_lower for g in ['thank', 'thanks', 'thx']):
                    answer = "You're welcome! Let me know if you have any more questions about your data."
                elif any(g in question_lower for g in ['bye', 'goodbye', 'see you']):
                    answer = "Goodbye! Come back anytime you need help analyzing your data."
                else:
                    answer = "I'm your AI data analyst! I'm here to help you understand your sales data. What would you like to know?"
                provider = "template"
                
                execution_time = time.time() - start_time
                return ChatResponse(
                    answer=answer,
                    execution_time=round(execution_time, 3),
                    provider=provider,
                    confidence=1.0,
                )
            
            # METRIC: Use safe SQL templates (DEFAULT for data questions)
            if intent.type == "METRIC":
                # DEFAULT PATH: Use safe SQL templates
                logger.info("Using METRIC path with safe SQL templates")
                
                # Let execute_metric_query handle all metric queries, including net revenue
                # It will extract entities (region, product) from intent and route appropriately
                query_result = execute_metric_query(intent, start_date, end_date, request.question)
                
                if not query_result.success:
                    return ChatResponse(
                        answer=f"Sorry, I couldn't retrieve the data: {query_result.error}",
                        sql=query_result.sql,
                        error=query_result.error,
                        provider="template",
                        confidence=1.0,
                    )
                
                # Generate response with Gemini if available
                if has_gemini_key:
                    answer = await generate_response_with_gemini(
                        request.question,
                        query_result,
                        "METRIC",
                        current_user_id
                    )
                    provider = "gemini"
                else:
                    answer = format_metric_response(
                        request.question,
                        query_result,
                        "METRIC",
                        intent.metric_type
                    )
                    provider = "template"
                
                execution_time = time.time() - start_time
                
                return ChatResponse(
                    answer=answer,
                    sql=query_result.sql,
                    data=query_result.data,
                    execution_time=round(execution_time, 3),
                    provider=provider,
                    confidence=1.0,  # Templates are always accurate
                )
            
            elif intent.type == "ADVISORY":
                # ADVISORY PATH: Get comprehensive data + Gemini analysis
                logger.info("Using ADVISORY path with comprehensive data")
                
                # Get comprehensive data for analysis
                query_result = query_advisory_data(start_date, end_date)
                
                if not query_result.success:
                    return ChatResponse(
                        answer=f"Sorry, I couldn't retrieve the data for analysis: {query_result.error}",
                        sql=query_result.sql,
                        error=query_result.error,
                        provider="template",
                        confidence=0.8,
                    )
                
                # Get RAG context for additional insights
                rag_context = None
                try:
                    rag_context = retrieve_context(request.question)
                except Exception as e:
                    logger.warning(f"RAG context retrieval failed: {e}")
                
                # Generate advisory response with Gemini
                if has_gemini_key:
                    answer = await generate_response_with_gemini(
                        request.question,
                        query_result,
                        "ADVISORY",
                        current_user_id,
                        rag_context
                    )
                    provider = "gemini"
                else:
                    answer = format_advisory_response(request.question, query_result)
                    provider = "template"
                
                execution_time = time.time() - start_time
                
                return ChatResponse(
                    answer=answer,
                    sql=query_result.sql,
                    data=query_result.data,
                    execution_time=round(execution_time, 3),
                    provider=provider,
                    confidence=0.9,
                )
            
            else:
                # EXPLORATION PATH: RAG-enhanced LLM SQL generation with validation
                logger.info("Using EXPLORATION path with RAG-enhanced SQL generation")
                
                # Check Ollama availability (needed as fallback)
                ollama_available = check_ollama_connection()
                if not ollama_available and not has_gemini_key:
                    return ChatResponse(
                        answer="AI service is not available. Please add a Gemini API key in Settings.",
                        error="No LLM available"
                    )
                
                # Get schema for SQL generation
                schema = get_database_schema()
                if not schema.get('columns'):
                    return ChatResponse(
                        answer="Database schema could not be retrieved.",
                        error="Schema retrieval failed"
                    )
                
                # STEP 1: Retrieve RAG context (ChromaDB semantic search)
                try:
                    rag_context = retrieve_context(request.question, schema)
                    logger.info(f"RAG: Retrieved context with {len(rag_context.get('semantic_matches', []))} semantic matches")
                except Exception as rag_error:
                    logger.warning(f"RAG context retrieval failed: {rag_error}")
                    rag_context = {'column_mappings': {}, 'semantic_matches': [], 'business_rules': []}
                
                # Generate SQL with LLM (will be enhanced with RAG context internally)
                date_context = {'start_date': start_date, 'end_date': end_date} if start_date and end_date else None
                sql, error, provider, sql_gen_time, confidence = await generate_sql_from_question(
                    request.question,
                    schema,
                    user_id=current_user_id,
                    date_context=date_context,
                    rag_context=rag_context  # Pass RAG context to SQL generator
                )
                
                if error or not sql:
                    return ChatResponse(
                        answer=f"I couldn't understand your question. Try asking something like 'Show me top 10 products' or 'What was my revenue?'",
                        error=error or "SQL generation failed",
                        provider=provider,
                        confidence=confidence,
                    )
                
                # STEP 2: Validate and auto-correct SQL column names
                is_valid, validation_error, corrected_sql = validate_sql_columns(sql)
                if corrected_sql and corrected_sql != sql:
                    logger.info(f"SQL auto-corrected: {sql[:100]}... -> {corrected_sql[:100]}...")
                    sql = corrected_sql
                
                if not is_valid:
                    logger.warning(f"SQL validation failed: {validation_error}")
                    # Try auto-correction
                    sql = auto_correct_sql(sql)
                
                # Validate SQL safety
                is_safe, safety_error = validate_sql_safety(sql)
                if not is_safe:
                    return ChatResponse(
                        answer="I couldn't generate a safe query. Please try rephrasing.",
                        sql=sql,
                        error=safety_error,
                        provider=provider,
                        confidence=confidence,
                    )
                
                # Execute SQL
                try:
                    result_df = execute_query(sql)
                    if len(result_df) > 100:
                        result_df = result_df.head(100)
                    
                    # Create QueryResult for response generator
                    if result_df.empty:
                        data = {}
                    elif len(result_df) == 1 and len(result_df.columns) == 1:
                        data = {"value": result_df.iloc[0, 0]}
                    else:
                        data = {"rows": result_df.to_dict('records'), "count": len(result_df)}
                    
                    query_result = QueryResult(success=True, data=data, sql=sql)
                    
                    # STEP 3: Generate response with data grounding
                    if has_gemini_key:
                        answer = await generate_response_with_gemini(
                            request.question,
                            query_result,
                            "EXPLORATION",
                            current_user_id,
                            rag_context  # Pass RAG context for grounded response
                        )
                    else:
                        answer = format_metric_response(request.question, query_result, "EXPLORATION")
                    
                    execution_time = time.time() - start_time
                    
                    return ChatResponse(
                        answer=answer,
                        sql=sql,
                        data=data,
                        execution_time=round(execution_time, 3),
                        provider=provider,
                        confidence=round(confidence, 2),
                    )
                    
                except Exception as query_error:
                    error_msg = str(query_error)
                    logger.error(f"Query execution failed: {error_msg}")
                    
                    # If column error, provide helpful message
                    if "not found" in error_msg.lower() or "binder error" in error_msg.lower():
                        return ChatResponse(
                            answer="I had trouble with the database query. The column names might have changed. Please try a simpler question like 'What is my total revenue?'",
                            sql=sql,
                            error=error_msg,
                            provider=provider,
                            confidence=confidence,
                            suggestion="Try: 'What is my revenue?' or 'Show me top products'"
                        )
                    
                    return ChatResponse(
                        answer="Sorry, the query failed to execute. Please try a different question.",
                        sql=sql,
                        error=error_msg,
                        provider=provider,
                        confidence=confidence,
                    )
        
        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_suggested_questions():
    """Get suggested questions categorized by type"""
    return {
        "suggestions": [
            # Metric questions
            "What is my total revenue?",
            "What is my net revenue?",
            "Show me top 10 products by revenue",
            "What is my refund rate?",
            "Revenue by region",
            # Advisory questions
            "Where should I focus my business?",
            "Are there any worrying trends?",
            "Which products have high refund rates?",
            # Exploration questions
            "How many orders did I have?",
            "Compare Mumbai vs Delhi revenue",
        ]
    }
