"""
API Key Validation Utilities

Simplified validation focusing on Gemini API.
Validates API key formats and tests keys by making API calls.
"""

import httpx
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class APIKeyValidationError(Exception):
    """Custom exception for API key validation errors"""
    pass


def validate_gemini_key_format(api_key: str) -> bool:
    """
    Validate Gemini API key format
    
    Gemini API keys:
    - Start with "AIza" typically
    - Are 39 characters long
    - Alphanumeric with underscores/hyphens
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if format is valid
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    api_key = api_key.strip()
    
    # Must be at least 20 chars
    if len(api_key) < 20:
        return False
    
    # Should not look like other provider keys
    if api_key.startswith('sk-'):
        return False
    
    return True


async def test_gemini_key(api_key: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Test Google Gemini API key by making a minimal API call.
    
    This is a SIMPLIFIED version that matches the working test script exactly.
    
    Args:
        api_key: The API key to test
        timeout: Request timeout in seconds (default 10)
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not api_key or not isinstance(api_key, str):
        return False, "API key cannot be empty"
    
    api_key = api_key.strip()
    if len(api_key) < 20:
        return False, "API key appears to be too short"
    
    logger.info(f"Testing Gemini API key (length: {len(api_key)}, starts with: {api_key[:10]}...)")
    
    # Use the exact same endpoint that works in the test script
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": "Hi"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1
        }
    }
    
    try:
        logger.info(f"Calling Gemini API endpoint...")
        
        # Simple timeout - exactly like test script
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            response = await client.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("✅ Gemini API key validation successful!")
                return True, None
            
            # Handle errors
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "API error")
            except:
                error_msg = response.text[:200] if response.text else "Unknown error"
            
            if response.status_code == 400:
                logger.error(f"Gemini API 400 error: {error_msg}")
                return False, f"Invalid request: {error_msg}"
            elif response.status_code == 401:
                logger.error(f"Gemini API 401 error: {error_msg}")
                return False, "Invalid API key. Authentication failed."
            elif response.status_code == 403:
                logger.error(f"Gemini API 403 error: {error_msg}")
                return False, f"Access forbidden: {error_msg}"
            elif response.status_code == 429:
                logger.warning(f"Gemini API 429 rate limit")
                return False, "Rate limit exceeded. Please wait and try again."
            else:
                logger.error(f"Gemini API error {response.status_code}: {error_msg}")
                return False, f"API error ({response.status_code}): {error_msg}"
                
    except httpx.TimeoutException:
        logger.error("Gemini API request timed out")
        return False, f"Request timed out after {timeout} seconds. Check your internet connection."
    except httpx.RequestError as e:
        logger.error(f"Network error: {e}")
        return False, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error testing Gemini key: {e}")
        return False, f"Unexpected error: {str(e)}"


async def validate_and_test_api_key(provider: str, api_key: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Validate API key format and test it with a real API call.
    
    Currently only supports Gemini. OpenAI and Anthropic are disabled.
    
    Args:
        provider: 'gemini' (only gemini supported for now)
        api_key: The API key to validate and test
        timeout: Request timeout in seconds
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if provider == "gemini":
        # Validate format first
        if not validate_gemini_key_format(api_key):
            return False, "Invalid Gemini API key format. Key should be at least 20 characters."
        
        # Test the key
        return await test_gemini_key(api_key, timeout=timeout)
    
    elif provider in ["openai", "anthropic"]:
        # Temporarily disabled - just accept the key without testing
        logger.info(f"{provider} validation temporarily disabled - accepting key")
        return True, None
    
    else:
        raise APIKeyValidationError(f"Invalid provider: {provider}. Must be 'gemini'")
