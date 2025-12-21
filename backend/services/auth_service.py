"""
Authentication service - Password hashing and JWT token management
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import bcrypt
from jose import JWTError, jwt
from core.config import settings
import logging

logger = logging.getLogger(__name__)


def hash_password(plain_password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    
    Args:
        plain_password: Plain text password
        
    Returns:
        Hashed password string
    """
    password_bytes = plain_password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        password_bytes = plain_password.encode('utf-8')
        hashed_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception as e:
        logger.warning(f"Password verification error: {e}")
        return False


def create_access_token(
    user_id: int,
    email: str,
    role: str,
    tenant_id: Optional[str] = None,
    token_version: int = 0
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User ID
        email: User email
        role: User role ("user" or "admin")
        tenant_id: Optional tenant ID for multi-tenant support
        token_version: Token version for session invalidation (incremented on password change)
        
    Returns:
        Encoded JWT token string
    """
    expire = datetime.utcnow() + timedelta(seconds=settings.JWT_EXPIRES_IN)
    
    payload: Dict[str, Any] = {
        "sub": str(user_id),  # Subject (user ID)
        "email": email,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "tv": token_version,  # Token version - invalidates old sessions on password change
    }
    
    if tenant_id:
        payload["tenant_id"] = tenant_id
    
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    logger.debug(f"Created access token for user {user_id} (role: {role}, token_version: {token_version})")
    
    return token


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT access token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        JWTError: If token is invalid, expired, or malformed
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise

