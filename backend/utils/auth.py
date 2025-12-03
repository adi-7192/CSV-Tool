"""
Authentication utilities

Simple authentication dependency for protecting endpoints.
Can be extended to use JWT tokens, OAuth, etc.
"""

from fastapi import HTTPException, status, Header
from typing import Optional
import logging

logger = logging.getLogger(__name__)


async def get_current_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Get current user ID from request headers
    
    This is a simple authentication mechanism. In production, you should:
    - Use JWT tokens
    - Validate tokens against a user database
    - Extract user_id from token payload
    
    For now, we support:
    1. X-User-ID header (for development/testing)
    2. Authorization header (Bearer token - can be extended)
    
    Args:
        x_user_id: User ID from X-User-ID header
        authorization: Authorization header (Bearer token)
        
    Returns:
        str: User ID
        
    Raises:
        HTTPException: 401 if user is not authenticated
    """
    # Priority: X-User-ID header (for development)
    if x_user_id:
        return x_user_id.strip()
    
    # TODO: Extract user_id from JWT token in Authorization header
    # For now, if Authorization is provided, we'd need to decode it
    # Example:
    # if authorization and authorization.startswith("Bearer "):
    #     token = authorization.replace("Bearer ", "")
    #     # Decode JWT and extract user_id
    #     user_id = decode_jwt_token(token)
    #     return user_id
    
    # If no authentication provided, raise 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Please provide X-User-ID header or valid Authorization token.",
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

