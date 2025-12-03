"""
API Routes for managing user API keys

Endpoints for creating, reading, updating, and deleting encrypted API keys.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from models.api_key_requests import (
    CreateAPIKeyRequest,
    GetAPIKeyRequest,
    DeleteAPIKeyRequest,
)
from models.api_key_responses import (
    APIKeyResponse,
    APIKeyListResponse,
    APIKeyInfo,
    EncryptionStatusResponse,
)
from services.api_key_service import APIKeyService
from utils.encryption import EncryptionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


@router.post("/", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(request: CreateAPIKeyRequest):
    """
    Create or update an API key for a user
    
    The API key will be encrypted before storage.
    Only a masked version will be returned.
    """
    try:
        result = APIKeyService.create_api_key(
            user_id=request.user_id,
            provider=request.provider,
            api_key=request.api_key
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
            message=f"API key for {request.provider} saved successfully",
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


@router.get("/{user_id}/{provider}", response_model=APIKeyResponse)
async def get_api_key(user_id: str, provider: str):
    """
    Get an API key for a user (returns masked version only)
    
    The key is never returned in full - only masked (first 6, last 4 chars).
    """
    if provider not in ['openai', 'anthropic']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai' or 'anthropic'"
        )
    
    try:
        key_info = APIKeyService.get_api_key(user_id, provider, decrypt=False)
        
        if not key_info:
            return APIKeyResponse(
                success=False,
                data=None,
                message=f"No API key found for {provider}",
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
    
    except Exception as e:
        logger.error(f"Failed to get API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key"
        )


@router.get("/{user_id}", response_model=APIKeyListResponse)
async def get_all_api_keys(user_id: str):
    """
    Get all API keys for a user (all masked)
    """
    try:
        keys = APIKeyService.get_all_api_keys(user_id)
        
        return APIKeyListResponse(
            success=True,
            data=[
                APIKeyInfo(
                    id=key['id'],
                    user_id=key['user_id'],
                    provider=key['provider'],
                    masked_key=key['masked_key'],
                    enabled=key.get('enabled', True),
                    created_at=key['created_at'],
                    updated_at=key['updated_at'],
                )
                for key in keys
            ],
            count=len(keys),
        )
    
    except Exception as e:
        logger.error(f"Failed to get API keys: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys"
        )


@router.delete("/{user_id}/{provider}", response_model=APIKeyResponse)
async def delete_api_key(user_id: str, provider: str):
    """
    Delete an API key for a user
    """
    if provider not in ['openai', 'anthropic']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid provider. Must be 'openai' or 'anthropic'"
        )
    
    try:
        result = APIKeyService.delete_api_key(user_id, provider)
        
        if result['success']:
            return APIKeyResponse(
                success=True,
                data=None,
                message=result['message'],
            )
        else:
            return APIKeyResponse(
                success=False,
                data=None,
                message=result.get('message', 'Failed to delete API key'),
            )
    
    except Exception as e:
        logger.error(f"Failed to delete API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key"
        )


@router.get("/verify/encryption", response_model=EncryptionStatusResponse)
async def verify_encryption():
    """
    Verify that encryption is working correctly
    
    This endpoint can be used to check if the encryption service is properly configured.
    """
    try:
        is_working = APIKeyService.verify_encryption()
        
        if is_working:
            return EncryptionStatusResponse(
                encryption_working=True,
                message="Encryption service is working correctly"
            )
        else:
            return EncryptionStatusResponse(
                encryption_working=False,
                message="Encryption service is not working. Check API_KEY_ENCRYPTION_KEY environment variable."
            )
    
    except Exception as e:
        logger.error(f"Encryption verification failed: {e}")
        return EncryptionStatusResponse(
            encryption_working=False,
            message=f"Encryption verification failed: {str(e)}"
        )

