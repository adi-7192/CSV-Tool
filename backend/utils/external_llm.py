"""
External LLM Integration Utilities

Integration with OpenAI and Anthropic APIs for SQL generation.
"""

import httpx
import logging
import time
from typing import Tuple, Optional, Dict, Any
from services.api_key_service import APIKeyService
from utils.encryption import APIKeyEncryption

logger = logging.getLogger(__name__)


class ExternalLLMError(Exception):
    """Custom exception for external LLM errors"""
    pass


async def generate_sql_with_openai(
    prompt: str,
    api_key: str,
    timeout: int = 30
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Generate SQL using OpenAI GPT-4 Turbo
    
    Args:
        prompt: The prompt for SQL generation
        api_key: OpenAI API key (decrypted)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (sql_query, error_message, response_time)
    """
    start_time = time.time()
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "gpt-4o",  # Using GPT-4o (latest model, faster and cheaper)
            "messages": [
                {
                    "role": "system",
                    "content": "You are a SQL expert. Generate DuckDB SQL queries. Return ONLY the SQL query, no explanations, no markdown, no code blocks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                sql = result['choices'][0]['message']['content'].strip()
                
                # Extract SQL if wrapped in code blocks
                if '```sql' in sql:
                    sql = sql.split('```sql')[1].split('```')[0].strip()
                elif '```' in sql:
                    parts = sql.split('```')
                    if len(parts) >= 2:
                        sql = parts[1].strip()
                        if sql.startswith('sql'):
                            sql = sql[3:].strip()
                
                # Remove trailing semicolons and comments
                sql = sql.strip().rstrip(';')
                lines = sql.split('\n')
                sql = '\n'.join([line for line in lines if not line.strip().startswith('--')]).strip()
                
                logger.info(f"OpenAI SQL generation successful (time: {response_time:.2f}s)")
                return sql, None, response_time
            
            elif response.status_code == 401:
                error_msg = "Invalid OpenAI API key"
                logger.error(f"OpenAI API error: {error_msg}")
                return None, error_msg, response_time
            
            elif response.status_code == 429:
                error_msg = "OpenAI rate limit exceeded"
                logger.error(f"OpenAI API error: {error_msg}")
                return None, error_msg, response_time
            
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                error_msg = f"OpenAI API error: {response.status_code} - {error_text}"
                logger.error(error_msg)
                return None, error_msg, response_time
    
    except httpx.TimeoutException:
        response_time = time.time() - start_time
        error_msg = "OpenAI request timed out"
        logger.error(error_msg)
        return None, error_msg, response_time
    
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"OpenAI API error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg, response_time


async def generate_sql_with_anthropic(
    prompt: str,
    api_key: str,
    timeout: int = 30
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Generate SQL using Anthropic Claude 3 Sonnet
    
    Args:
        prompt: The prompt for SQL generation
        api_key: Anthropic API key (decrypted)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (sql_query, error_message, response_time)
    """
    start_time = time.time()
    
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are a SQL expert. Generate DuckDB SQL queries. Return ONLY the SQL query, no explanations, no markdown, no code blocks.

{prompt}"""
                }
            ],
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                sql = result['content'][0]['text'].strip()
                
                # Extract SQL if wrapped in code blocks
                if '```sql' in sql:
                    sql = sql.split('```sql')[1].split('```')[0].strip()
                elif '```' in sql:
                    parts = sql.split('```')
                    if len(parts) >= 2:
                        sql = parts[1].strip()
                        if sql.startswith('sql'):
                            sql = sql[3:].strip()
                
                # Remove trailing semicolons and comments
                sql = sql.strip().rstrip(';')
                lines = sql.split('\n')
                sql = '\n'.join([line for line in lines if not line.strip().startswith('--')]).strip()
                
                logger.info(f"Anthropic SQL generation successful (time: {response_time:.2f}s)")
                return sql, None, response_time
            
            elif response.status_code == 401:
                error_msg = "Invalid Anthropic API key"
                logger.error(f"Anthropic API error: {error_msg}")
                return None, error_msg, response_time
            
            elif response.status_code == 429:
                error_msg = "Anthropic rate limit exceeded"
                logger.error(f"Anthropic API error: {error_msg}")
                return None, error_msg, response_time
            
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                error_msg = f"Anthropic API error: {response.status_code} - {error_text}"
                logger.error(error_msg)
                return None, error_msg, response_time
    
    except httpx.TimeoutException:
        response_time = time.time() - start_time
        error_msg = "Anthropic request timed out"
        logger.error(error_msg)
        return None, error_msg, response_time
    
    except Exception as e:
        response_time = time.time() - start_time
        error_msg = f"Anthropic API error: {str(e)}"
        logger.error(error_msg)
        return None, error_msg, response_time


async def generate_sql_with_gemini(
    prompt: str,
    api_key: str,
    timeout: int = 30
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Generate SQL using Google Gemini API.
    
    SIMPLIFIED version - uses the same endpoint that works in validation.
    
    Args:
        prompt: The prompt for SQL generation
        api_key: Gemini API key (decrypted)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (sql_query, error_message, response_time)
    """
    start_time = time.time()
    
    if not api_key or len(api_key.strip()) < 20:
        response_time = time.time() - start_time
        return None, "Invalid Gemini API key format", response_time
    
    # Use the same endpoint that works in validation
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": f"""You are a SQL expert. Generate DuckDB SQL queries. Return ONLY the SQL query, no explanations, no markdown, no code blocks.

{prompt}"""
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 500,
            "temperature": 0.1,
        }
    }
    
    try:
        logger.info("Calling Gemini API for SQL generation...")
        
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            
            response_time = time.time() - start_time
            logger.info(f"Gemini response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract text from response
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0:
                            sql = parts[0].get('text', '').strip()
                            
                            # Extract SQL if wrapped in code blocks
                            if '```sql' in sql:
                                sql = sql.split('```sql')[1].split('```')[0].strip()
                            elif '```' in sql:
                                code_parts = sql.split('```')
                                if len(code_parts) >= 2:
                                    sql = code_parts[1].strip()
                                    if sql.startswith('sql'):
                                        sql = sql[3:].strip()
                            
                            # Clean up
                            sql = sql.strip().rstrip(';')
                            lines = sql.split('\n')
                            sql = '\n'.join([line for line in lines if not line.strip().startswith('--')]).strip()
                            
                            if sql:
                                logger.info(f"✅ Gemini SQL generation successful (time: {response_time:.2f}s)")
                                return sql, None, response_time
                
                return None, "Empty or invalid response from Gemini", response_time
            
            # Handle errors
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "API error")
            except:
                error_msg = response.text[:200] if response.text else "Unknown error"
            
            if response.status_code == 401 or response.status_code == 403:
                return None, "Invalid Gemini API key", response_time
            elif response.status_code == 429:
                return None, "Rate limit exceeded. Please wait and try again.", response_time
            else:
                return None, f"Gemini API error ({response.status_code}): {error_msg}", response_time
                
    except httpx.TimeoutException:
        response_time = time.time() - start_time
        return None, f"Request timed out after {timeout} seconds", response_time
    except httpx.RequestError as e:
        response_time = time.time() - start_time
        return None, f"Network error: {str(e)}", response_time
    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Unexpected error: {e}")
        return None, f"Unexpected error: {str(e)}", response_time


def calculate_ollama_confidence(
    sql: str,
    response_time: float,
    error: Optional[str] = None
) -> float:
    """
    Calculate confidence score for Ollama-generated SQL
    
    Factors:
    - Response time (faster = higher confidence)
    - SQL quality indicators (SELECT, proper syntax)
    - Error presence (errors = lower confidence)
    
    Args:
        sql: Generated SQL query
        response_time: Time taken to generate SQL
        error: Optional error message
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if error:
        return 0.3  # Low confidence if there was an error
    
    if not sql:
        return 0.0  # No SQL = no confidence
    
    confidence = 0.7  # Base confidence for Ollama
    
    # Time-based confidence (faster = better)
    if response_time < 5:
        confidence += 0.1
    elif response_time > 20:
        confidence -= 0.1
    elif response_time > 30:
        confidence -= 0.2
    
    # SQL quality indicators
    sql_upper = sql.upper()
    if sql_upper.startswith('SELECT'):
        confidence += 0.1
    
    # Check for common SQL patterns (indicates good quality)
    if any(keyword in sql_upper for keyword in ['WHERE', 'FROM', 'SUM', 'COUNT', 'GROUP BY']):
        confidence += 0.05
    
    # Check for potential issues (reduce confidence)
    if any(keyword in sql_upper for keyword in ['LIMIT 1', 'LIMIT 0']):
        confidence -= 0.05
    
    # Clamp between 0.0 and 1.0
    confidence = max(0.0, min(1.0, confidence))
    
    return confidence


async def generate_sql_with_external_llm(
    prompt: str,
    user_id: Optional[str],
    provider_preference: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str, float, float]:
    """
    Generate SQL using external LLM (OpenAI, Anthropic, or Gemini) with fallback
    
    Args:
        prompt: The prompt for SQL generation
        user_id: User ID to fetch API keys (can be None)
        provider_preference: Preferred provider ('openai', 'anthropic', or 'gemini'), None for auto
        
    Returns:
        Tuple of (sql_query, error_message, provider_used, response_time, confidence)
        confidence is always 1.0 for external LLMs (high quality)
    """
    # Check user_id first
    if not user_id:
        logger.info(f"generate_sql_with_external_llm: user_id=None, skipping external LLM")
        return None, "No user_id provided, cannot lookup API keys", 'none', 0.0, 0.0
    
    # Check API key existence and enabled status
    openai_key = None
    anthropic_key = None
    gemini_key = None
    
    try:
        openai_key = APIKeyService.get_api_key(user_id, 'openai', decrypt=True)
        anthropic_key = APIKeyService.get_api_key(user_id, 'anthropic', decrypt=True)
        gemini_key = APIKeyService.get_api_key(user_id, 'gemini', decrypt=True)
    except Exception as e:
        logger.error(f"Error checking API keys for user_id={user_id}: {e}")
        return None, f"Error checking API keys: {str(e)}", 'none', 0.0, 0.0
    
    # Only consider keys that exist, are decrypted, and are enabled
    has_openai_key = (openai_key is not None and 
                      openai_key.get('key') is not None and 
                      openai_key.get('enabled', True))
    has_anthropic_key = (anthropic_key is not None and 
                        anthropic_key.get('key') is not None and 
                        anthropic_key.get('enabled', True))
    has_gemini_key = (gemini_key is not None and 
                     gemini_key.get('key') is not None and 
                     gemini_key.get('enabled', True))
    
    logger.info(f"generate_sql_with_external_llm: user_id={user_id}, has_openai_key={has_openai_key}, has_anthropic_key={has_anthropic_key}, has_gemini_key={has_gemini_key}")
    
    # Track errors for better error messages
    openai_error = None
    anthropic_error = None
    gemini_error = None
    
    # Build list of enabled providers to try (only enabled ones)
    enabled_providers = []
    if has_openai_key:
        enabled_providers.append(('openai', openai_key))
    if has_anthropic_key:
        enabled_providers.append(('anthropic', anthropic_key))
    if has_gemini_key:
        enabled_providers.append(('gemini', gemini_key))
    
    # If provider_preference is set, prioritize that provider
    if provider_preference:
        preferred_idx = next((i for i, (p, _) in enumerate(enabled_providers) if p == provider_preference), None)
        if preferred_idx is not None:
            # Move preferred provider to front
            enabled_providers.insert(0, enabled_providers.pop(preferred_idx))
    
    # Try each enabled provider in order
    for provider_name, provider_key in enabled_providers:
        masked_key = APIKeyEncryption.mask_key(provider_key['key'])
        logger.info(f"Attempting {provider_name} API call with key: {masked_key}")
        
        try:
            if provider_name == 'openai':
                sql, error, response_time = await generate_sql_with_openai(
                    prompt, provider_key['key'], timeout=30
                )
                if sql and not error:
                    logger.info(f"✅ OpenAI SQL generation successful (response_time: {response_time:.2f}s)")
                    return sql, None, 'openai', response_time, 1.0
                openai_error = error
            elif provider_name == 'anthropic':
                sql, error, response_time = await generate_sql_with_anthropic(
                    prompt, provider_key['key'], timeout=30
                )
                if sql and not error:
                    logger.info(f"✅ Anthropic SQL generation successful (response_time: {response_time:.2f}s)")
                    return sql, None, 'anthropic', response_time, 1.0
                anthropic_error = error
            elif provider_name == 'gemini':
                sql, error, response_time = await generate_sql_with_gemini(
                    prompt, provider_key['key'], timeout=30
                )
                if sql and not error:
                    logger.info(f"✅ Gemini SQL generation successful (response_time: {response_time:.2f}s)")
                    return sql, None, 'gemini', response_time, 1.0
                gemini_error = error
            
            # If we got here, this provider failed - log and continue to next
            if error:
                if "Invalid" in error or "validation" in error.lower():
                    logger.warning(f"{provider_name} API key validation failed: {error}")
                elif "rate limit" in error.lower():
                    logger.warning(f"{provider_name} rate limit exceeded: {error}")
                else:
                    logger.warning(f"{provider_name} API call failed: {error}")
        except Exception as e:
            error_msg = f"{provider_name} API call failed: {str(e)}"
            logger.error(error_msg)
            if provider_name == 'openai':
                openai_error = str(e)
            elif provider_name == 'anthropic':
                anthropic_error = str(e)
            elif provider_name == 'gemini':
                gemini_error = str(e)
    
    # If we get here, all enabled providers failed or none were enabled
    # Determine appropriate error message based on what happened
    if not has_openai_key and not has_anthropic_key and not has_gemini_key:
        error_message = f"No API key found for user {user_id}. Add one in Settings."
        logger.info(f"generate_sql_with_external_llm: {error_message}")
    elif has_openai_key or has_anthropic_key or has_gemini_key:
        # Keys exist but all failed - prioritize validation errors
        if openai_error and "Invalid" in openai_error:
            error_message = f"API key validation failed: {openai_error}. Please check your key in Settings."
        elif anthropic_error and "Invalid" in anthropic_error:
            error_message = f"API key validation failed: {anthropic_error}. Please check your key in Settings."
        elif gemini_error and "Invalid" in gemini_error:
            error_message = f"API key validation failed: {gemini_error}. Please check your key in Settings."
        elif openai_error:
            error_message = f"OpenAI API call failed: {openai_error}. Check API key and quota."
        elif anthropic_error:
            error_message = f"Anthropic API call failed: {anthropic_error}. Check API key and quota."
        elif gemini_error:
            error_message = f"Gemini API call failed: {gemini_error}. Check API key and quota."
        else:
            error_message = "All external LLM API calls failed. Please check your API keys and quota in Settings."
        logger.warning(f"generate_sql_with_external_llm: {error_message}")
    else:
        error_message = "No external LLM API keys available or all failed"
        logger.warning(f"generate_sql_with_external_llm: {error_message}")
    
    return None, error_message, 'none', 0.0, 0.0

