"""
User API Key Management Endpoints

Endpoints for users to manage their own LLM API keys with validation and authentication.
"""

from fastapi import APIRouter, HTTPException, status, Query, Depends, Body
from typing import Optional
from pydantic import BaseModel, Field
import logging
import time

from models.api_key_responses import APIKeyResponse, APIKeyInfo
from services.api_key_service import APIKeyService
from utils.encryption import EncryptionError
from utils.api_key_validator import validate_and_test_api_key, APIKeyValidationError
from utils.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user/api-key", tags=["User API Keys"])


@router.get("/ping")
async def ping():
    """Simple ping endpoint to verify API is reachable"""
    return {"status": "ok", "message": "API key service is running"}


class CreateUserAPIKeyRequest(BaseModel):
    """Request model for creating/updating user API key"""
    provider: str = Field(..., description="API provider: 'openai', 'anthropic', or 'gemini'")
    api_key: str = Field(..., min_length=1, description="The API key to store (will be validated and encrypted)")


class UpdateEnabledStatusRequest(BaseModel):
    """Request model for updating enabled status"""
    provider: str = Field(..., description="API provider: 'openai', 'anthropic', or 'gemini'")
    enabled: bool = Field(..., description="Whether to enable or disable this API key")


@router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(
    request: CreateUserAPIKeyRequest = Body(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Create or update an API key for the authenticated user
    
    - Validates API key format (OpenAI: starts with "sk-", Anthropic: starts with "sk-ant-", Gemini: alphanumeric, min 20 chars)
    - Tests the key by making a small API call to verify it's valid
    - Encrypts and stores the key if valid
    - Returns success or specific error message
    
    Protected with authentication - users can only manage their own keys.
    """
    provider = request.provider
    api_key = request.api_key
    
    # Validate provider
    if provider not in ['openai', 'anthropic', 'gemini']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai', 'anthropic', or 'gemini'"
        )
    
    # Validate and test API key
    try:
        logger.info(f"Validating {provider} API key for user {current_user_id}")
        logger.info(f"API key length: {len(api_key)}, preview: {api_key[:10]}...")
        
        # Simple 10 second timeout - matches test script
        start_time = time.time()
        is_valid, error_message = await validate_and_test_api_key(provider, api_key, timeout=10)
        elapsed = time.time() - start_time
        logger.info(f"Validation completed in {elapsed:.2f}s, valid={is_valid}")
        
        if not is_valid:
            logger.warning(f"API key validation failed: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message or "API key validation failed"
            )
        
        logger.info(f"✅ API key validation successful for {provider}")
    
    except HTTPException:
        raise
    except APIKeyValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate API key: {str(e)}"
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
                enabled=result.get('enabled', True),
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


@router.get("", response_model=APIKeyResponse)
async def get_user_api_key(
    provider: str = Query(..., description="API provider: 'openai', 'anthropic', or 'gemini'"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Get the authenticated user's API key (masked version only)
    
    Returns whether user has a key and a masked version (never full key).
    Mask shows only first 6 and last 4 characters (e.g., "sk-...wxyz").
    
    Protected with authentication.
    """
    # Validate provider
    if provider not in ['openai', 'anthropic', 'gemini']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai', 'anthropic', or 'gemini'"
        )
    
    try:
        key_info = APIKeyService.get_api_key(current_user_id, provider, decrypt=False)
        
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No API key found for {provider}"
            )
        
        return APIKeyResponse(
            success=True,
            data=APIKeyInfo(
                id=key_info['id'],
                user_id=key_info['user_id'],
                provider=key_info['provider'],
                masked_key=key_info['key'],  # Already masked
                enabled=key_info.get('enabled', True),
                created_at=key_info['created_at'],
                updated_at=key_info['updated_at'],
            ),
            message="API key retrieved successfully",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key"
        )


@router.delete("", response_model=APIKeyResponse)
async def delete_user_api_key(
    provider: str = Query(..., description="API provider: 'openai', 'anthropic', or 'gemini'"),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Delete the authenticated user's API key
    
    Removes the API key from database for that user.
    Returns success message.
    
    Protected with authentication.
    """
    # Validate provider
    if provider not in ['openai', 'anthropic', 'gemini']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai', 'anthropic', or 'gemini'"
        )
    
    try:
        # Check if key exists first
        key_info = APIKeyService.get_api_key(current_user_id, provider, decrypt=False)
        
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No API key found for {provider}"
            )
        
        # Delete the key
        result = APIKeyService.delete_api_key(current_user_id, provider)
        
        if result['success']:
            return APIKeyResponse(
                success=True,
                data=None,
                message=result['message'],
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('message', 'Failed to delete API key')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )


@router.patch("/enable", response_model=APIKeyResponse)
async def update_enabled_status(
    request: UpdateEnabledStatusRequest = Body(...),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Update the enabled status of an API key
    
    - Only enabled API keys will be used for AI chat
    - Users can enable/disable keys without deleting them
    - Protected with authentication
    
    Protected with authentication.
    """
    provider = request.provider
    enabled = request.enabled
    
    # Validate provider
    if provider not in ['openai', 'anthropic', 'gemini']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai', 'anthropic', or 'gemini'"
        )
    
    try:
        # Check if key exists first
        key_info = APIKeyService.get_api_key(current_user_id, provider, decrypt=False)
        
        if not key_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No API key found for {provider}. Please add an API key first."
            )
        
        # Update enabled status
        result = APIKeyService.update_enabled_status(current_user_id, provider, enabled)
        
        if result['success']:
            # Get updated key info
            updated_key_info = APIKeyService.get_api_key(current_user_id, provider, decrypt=False)
            
            return APIKeyResponse(
                success=True,
                data=APIKeyInfo(
                    id=updated_key_info['id'],
                    user_id=updated_key_info['user_id'],
                    provider=updated_key_info['provider'],
                    masked_key=updated_key_info['key'],
                    enabled=updated_key_info.get('enabled', True),
                    created_at=updated_key_info['created_at'],
                    updated_at=updated_key_info['updated_at'],
                ),
                message=f"API key for {provider} {'enabled' if enabled else 'disabled'} successfully",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get('message', 'Failed to update enabled status')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update enabled status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update enabled status"
        )

