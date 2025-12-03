"""
Authentication utilities

Simple authentication dependency for protecting endpoints.
Uses JWT tokens from Authorization header.
"""

from fastapi import HTTPException, status, Header, Depends
from typing import Optional
import logging
from services.auth_service import decode_access_token

logger = logging.getLogger(__name__)


async def get_current_user_id(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),  # Fallback for backward compatibility
) -> str:
    """
    Get current user ID from JWT token in Authorization header
    
    Priority:
    1. JWT token from Authorization header (Bearer token)
    2. X-User-ID header (fallback for backward compatibility)
    
    Args:
        authorization: Authorization header (Bearer token)
        x_user_id: User ID from X-User-ID header (fallback)
        
    Returns:
        str: User ID
        
    Raises:
        HTTPException: 401 if user is not authenticated
    """
    # Priority 1: Extract from JWT token
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "").strip()
            payload = decode_access_token(token)
            # JWT uses "sub" (subject) for user_id
            user_id = str(payload.get('sub') or payload.get('user_id'))
            if user_id:
                logger.debug(f"Extracted user_id {user_id} from JWT token")
                return user_id
        except Exception as e:
            logger.warning(f"Failed to decode JWT token: {e}")
            # Fall through to check X-User-ID header
    
    # Priority 2: X-User-ID header (fallback for backward compatibility)
    if x_user_id:
        logger.debug(f"Using X-User-ID header: {x_user_id}")
        return x_user_id.strip()
    
    # If no authentication provided, raise 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide a valid Authorization token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_user_access(user_id: str, resource_user_id: str) -> None:
    """
    Verify that a user has access to a resource
    
    Args:
        user_id: The authenticated user's ID
        resource_user_id: The resource owner's user ID
        
    Raises:
        HTTPException: 403 if user doesn't have access
    """
    if user_id != resource_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this resource."
        )

