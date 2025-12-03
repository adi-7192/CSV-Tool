# Async Error Debug - Configuration Files

## File 1: backend/core/ai_service.py

```python
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
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from calendar import monthrange
from core.config import settings
from core.database import execute_query, get_connection
from utils.external_llm import generate_sql_with_external_llm, calculate_ollama_confidence
from utils.encryption import APIKeyEncryption

logger = logging.getLogger(__name__)


def generate_sql_from_question(
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
            sql, error, provider, response_time, confidence = asyncio.run(
                generate_sql_with_external_llm(prompt, user_id)
            )
            
            if sql and not error:
                # Post-process external LLM SQL (same as Ollama)
                sql = post_process_sql(sql, question, schema, source_file)
                if sql:
                    logger.info(f"✅ External LLM ({provider}) SQL generation successful")
                    return sql, None, provider, response_time, confidence
                else:
                    logger.warning(f"External LLM SQL post-processing failed, falling back to Ollama")
            else:
                logger.warning(f"External LLM failed: {error}, falling back to Ollama")
        except Exception as e:
            logger.warning(f"External LLM call failed: {e}, falling back to Ollama")
    
    # Fallback to Ollama
    return generate_sql_with_ollama(prompt, question, schema, source_file, is_net_revenue_query)
```

**Key async call:**
- Line 307: `asyncio.run(generate_sql_with_external_llm(prompt, user_id))` - This is called from a synchronous function `generate_sql_from_question()`


## File 2: backend/api/routes/chat.py

```python
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
    current_user_id = x_user_id.strip() if x_user_id else None
    
    # Check Ollama connection (needed as fallback)
    ollama_available = check_ollama_connection()
    if not ollama_available:
        # Only fail if user has no API keys
        if current_user_id:
            from services.api_key_service import APIKeyService
            has_openai = APIKeyService.get_api_key(current_user_id, 'openai', decrypt=False) is not None
            has_anthropic = APIKeyService.get_api_key(current_user_id, 'anthropic', decrypt=False) is not None
            
            if not has_openai and not has_anthropic:
                return ChatResponse(
                    answer="AI service (Ollama) is not available and no external API keys found. Please start Ollama server or add an API key in Settings.",
                    error="Ollama connection failed and no API keys available"
                )
        else:
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
        sql, error, provider, sql_gen_time, confidence = generate_sql_from_question(
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
            logger.error(f"Query execution failed: {str(query_error)}")
            return ChatResponse(
                answer="Sorry, I generated a query but it failed to execute. Please try a different question.",
                sql=sql,
                error=str(query_error),
            )
    
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Key call:**
- Line 103: `generate_sql_from_question()` is called from an async endpoint `ask_question()`


## File 3: backend/main.py

```python
"""
FastAPI Backend for Analytics Dashboard

Inspired by Metabase architecture - Clean, modular, API-first design
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, upload, metrics, charts, chat, verification, data_status, data_routes, api_keys, user_api_keys
from core.config import settings
from core.database import init_database
from utils.logger import setup_logger, app_logger

# Initialize logging
logger = setup_logger('main')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Analytics API...")
    logger.info(f"Database path: {settings.DATABASE_PATH}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    init_database()  # Initialize database connections, create tables if needed
    logger.info("✅ Database initialized")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Analytics API...")


# Initialize FastAPI app
app = FastAPI(
    title="Analytics Dashboard API",
    description="FastAPI backend for business analytics with AI-powered insights",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# CORS middleware (allow React frontend to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        settings.FRONTEND_URL,    # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(data_status.router, prefix="/api/data", tags=["Data Status"])
app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(api_keys.router, tags=["API Keys"])
app.include_router(user_api_keys.router, tags=["User API Keys"])
```

**Router registration:**
- Line 65: `app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])` - Chat router registered
- Line 70: `app.include_router(user_api_keys.router, tags=["User API Keys"])` - User API keys router registered


## File 4: backend/models/api_key_responses.py

```python
"""
API Key Response Models

Pydantic models for API key API responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class APIKeyInfo(BaseModel):
    """API Key information (masked)"""
    id: str
    user_id: str
    provider: str  # 'openai' or 'anthropic'
    masked_key: str  # Masked version (e.g., "sk-...wxyz")
    created_at: datetime
    updated_at: datetime


class APIKeyResponse(BaseModel):
    """Response wrapper for API key operations"""
    success: bool
    data: Optional[APIKeyInfo] = None
    message: Optional[str] = None


class APIKeyListResponse(BaseModel):
    """Response for listing all API keys"""
    success: bool
    data: list[APIKeyInfo] = []
    message: Optional[str] = None


class EncryptionStatusResponse(BaseModel):
    """Response for encryption verification"""
    encryption_available: bool
    message: str
```


## File 5: backend/api/routes/user_api_keys.py

```python
"""
User API Key Management Endpoints

Endpoints for users to manage their own LLM API keys with validation and authentication.
"""

from fastapi import APIRouter, HTTPException, status, Query, Depends, Body
from typing import Optional
from pydantic import BaseModel, Field
import logging

from models.api_key_responses import APIKeyResponse, APIKeyInfo
from services.api_key_service import APIKeyService
from utils.encryption import EncryptionError
from utils.api_key_validator import validate_and_test_api_key, APIKeyValidationError
from utils.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user/api-key", tags=["User API Keys"])


class CreateUserAPIKeyRequest(BaseModel):
    """Request model for creating/updating user API key"""
    provider: str = Field(..., description="API provider: 'openai' or 'anthropic'")
    api_key: str = Field(..., min_length=1, description="The API key to store (will be validated and encrypted)")


@router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(
    request: CreateUserAPIKeyRequest = Body(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Create or update an API key for the authenticated user
    
    - Validates API key format (OpenAI: starts with "sk-", Anthropic: starts with "sk-ant-")
    - Tests the key by making a small API call to verify it's valid
    - Encrypts and stores the key if valid
    - Returns success or specific error message
    
    Protected with authentication - users can only manage their own keys.
    """
    provider = request.provider
    api_key = request.api_key
    
    # Validate provider
    if provider not in ['openai', 'anthropic']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai' or 'anthropic'"
        )
    
    # Validate and test API key
    try:
        is_valid, error_message = await validate_and_test_api_key(provider, api_key)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message or "API key validation failed"
            )
    
    except APIKeyValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate API key. Please try again."
        )
    
    # If validation passed, encrypt and store
    try:
        result = APIKeyService.create_api_key(
            user_id=current_user_id,
            provider=provider,
            api_key=api_key
        )
        
        return APIKeyResponse(
            success=True,
            data=APIKeyInfo(
                id=result['id'],
                user_id=result['user_id'],
                provider=result['provider'],
                masked_key=result['masked_key'],
                created_at=result['created_at'],
                updated_at=result['updated_at'],
            ),
            message=f"API key for {provider} saved successfully",
        )
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except EncryptionError as e:
        logger.error(f"Encryption error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Encryption service unavailable. Please check API_KEY_ENCRYPTION_KEY environment variable."
        )
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save API key"
        )
```

**Key async call:**
- Line 56: `await validate_and_test_api_key(provider, api_key)` - Async validation call


## Summary

**LLM Providers:**
- OpenAI (GPT-4o) - via `generate_sql_with_openai()` in `utils/external_llm.py`
- Anthropic (Claude 3 Sonnet) - via `generate_sql_with_anthropic()` in `utils/external_llm.py`
- Ollama (local fallback) - via `generate_sql_with_ollama()` in `core/ai_service.py`

**Async/Sync Pattern:**
- `ask_question()` endpoint (async) → calls `generate_sql_from_question()` (sync) → uses `asyncio.run()` to call `generate_sql_with_external_llm()` (async)
- **Issue**: `asyncio.run()` cannot be called from an async context (FastAPI endpoint). This will cause a "RuntimeError: asyncio.run() cannot be called from a running event loop"

**Endpoint Structure:**
- `/api/chat/ask` - POST endpoint (async) in `backend/api/routes/chat.py`
- `/api/user/api-key` - POST endpoint (async) in `backend/api/routes/user_api_keys.py`

**asyncio.run() Usage:**
- Line 307 in `backend/core/ai_service.py`: `asyncio.run(generate_sql_with_external_llm(prompt, user_id))`
- **Problem**: This is called from a sync function that is called from an async endpoint, which will fail if an event loop is already running.

