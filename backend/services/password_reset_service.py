"""
Password Reset Service - Secure token generation and validation.

Security features:
- Cryptographically secure random token generation
- SHA-256 hashed token storage (never store plaintext)
- Token expiration (configurable, default 15 minutes)
- Single-use tokens (marked as used after successful reset)
- Request metadata logging (IP, user agent)
"""
import secrets
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import logging

from core.database import get_connection, execute_query
from core.config import settings
from services.user_service import get_user_by_email, get_user_by_id

logger = logging.getLogger(__name__)


def _hash_token(token: str) -> str:
    """
    Hash a token using SHA-256.
    
    Args:
        token: Plain text token
        
    Returns:
        Hex-encoded SHA-256 hash
    """
    return hashlib.sha256(token.encode()).hexdigest()


def _generate_secure_token() -> str:
    """
    Generate a cryptographically secure random token.
    
    Uses secrets.token_urlsafe which is suitable for password reset tokens.
    
    Returns:
        URL-safe base64-encoded token (32 bytes = 43 characters)
    """
    return secrets.token_urlsafe(32)


def create_password_reset_token(
    user_id: int,
    request_ip: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    """
    Create a password reset token for a user.
    
    Args:
        user_id: User ID to create token for
        request_ip: Client IP address (for logging)
        user_agent: Client user agent (for logging)
        
    Returns:
        Plain text token to be sent to user (NOT the hash)
    """
    conn = get_connection()
    
    # Generate secure token
    token = _generate_secure_token()
    token_hash = _hash_token(token)
    token_id = str(uuid.uuid4())
    
    # Calculate expiration
    try:
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=settings.PASSWORD_RESET_TOKEN_EXPIRES_MINUTES
        )
    except ImportError:
        expires_at = datetime.utcnow() + timedelta(
            minutes=settings.PASSWORD_RESET_TOKEN_EXPIRES_MINUTES
        )
    
    try:
        now = datetime.now(timezone.utc)
    except ImportError:
        now = datetime.utcnow()
    
    # Invalidate any existing unused tokens for this user (optional security measure)
    conn.execute(
        """
        UPDATE password_reset_tokens 
        SET used_at = ? 
        WHERE user_id = ? AND used_at IS NULL
        """,
        [now, user_id]
    )
    
    # Insert new token (store hash, not plaintext)
    conn.execute(
        """
        INSERT INTO password_reset_tokens 
        (id, user_id, token_hash, expires_at, created_at, request_ip, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            token_id,
            user_id,
            token_hash,  # Never store plaintext token!
            expires_at,
            now,
            request_ip,
            user_agent
        ]
    )
    
    logger.info(f"Created password reset token for user {user_id}")
    
    # Return the plaintext token (to be sent via email)
    return token


def validate_password_reset_token(token: str) -> Tuple[bool, Optional[int], str]:
    """
    Validate a password reset token.
    
    Args:
        token: Plain text token from user
        
    Returns:
        Tuple of (is_valid: bool, user_id: Optional[int], error_message: str)
    """
    conn = get_connection()
    
    # Hash the provided token to compare with stored hash
    token_hash = _hash_token(token)
    
    try:
        now = datetime.now(timezone.utc)
    except ImportError:
        now = datetime.utcnow()
    
    # Find the token
    result = conn.execute(
        """
        SELECT id, user_id, expires_at, used_at
        FROM password_reset_tokens
        WHERE token_hash = ?
        """,
        [token_hash]
    ).fetchdf()
    
    if result.empty:
        logger.warning("Password reset token not found")
        return False, None, "INVALID_TOKEN"
    
    row = result.iloc[0]
    user_id = int(row['user_id'])
    expires_at = row['expires_at']
    used_at = row['used_at']
    
    # Check if token has been used
    if used_at is not None:
        logger.warning(f"Password reset token already used for user {user_id}")
        return False, None, "TOKEN_ALREADY_USED"
    
    # Check if token has expired
    # Handle timezone-aware and naive datetime comparison
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
    
    # Make comparison timezone-aware or naive depending on what we have
    if hasattr(expires_at, 'tzinfo') and expires_at.tzinfo is not None:
        now = datetime.now(timezone.utc)
    else:
        now = datetime.utcnow()
    
    if expires_at < now:
        logger.warning(f"Password reset token expired for user {user_id}")
        return False, None, "TOKEN_EXPIRED"
    
    logger.info(f"Password reset token validated for user {user_id}")
    return True, user_id, ""


def mark_token_as_used(token: str) -> bool:
    """
    Mark a password reset token as used.
    
    Args:
        token: Plain text token
        
    Returns:
        True if token was marked as used
    """
    conn = get_connection()
    token_hash = _hash_token(token)
    
    try:
        now = datetime.now(timezone.utc)
    except ImportError:
        now = datetime.utcnow()
    
    conn.execute(
        """
        UPDATE password_reset_tokens
        SET used_at = ?
        WHERE token_hash = ?
        """,
        [now, token_hash]
    )
    
    logger.info("Password reset token marked as used")
    return True


def update_user_password(user_id: int, new_password_hash: str) -> bool:
    """
    Update a user's password and password_changed_at timestamp.
    
    Args:
        user_id: User ID
        new_password_hash: New bcrypt password hash
        
    Returns:
        True if password was updated
    """
    conn = get_connection()
    
    try:
        now = datetime.now(timezone.utc)
    except ImportError:
        now = datetime.utcnow()
    
    conn.execute(
        """
        UPDATE users
        SET password_hash = ?, password_changed_at = ?
        WHERE id = ?
        """,
        [new_password_hash, now, user_id]
    )
    
    logger.info(f"Password updated for user {user_id}")
    return True


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password meets minimum requirements.
    
    Requirements:
    - Minimum length (from settings)
    - At least one letter
    - At least one number
    
    Args:
        password: Plain text password
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    min_length = settings.PASSWORD_MIN_LENGTH
    
    if len(password) < min_length:
        return False, f"Password must be at least {min_length} characters long"
    
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    return True, ""


def cleanup_expired_tokens() -> int:
    """
    Remove expired and used tokens older than 24 hours.
    
    Call periodically to clean up old tokens.
    
    Returns:
        Number of tokens deleted
    """
    conn = get_connection()
    
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    except ImportError:
        cutoff = datetime.utcnow() - timedelta(hours=24)
    
    # Count before delete
    count_result = conn.execute(
        """
        SELECT COUNT(*) as count FROM password_reset_tokens
        WHERE (used_at IS NOT NULL AND used_at < ?)
           OR (expires_at < ?)
        """,
        [cutoff, cutoff]
    ).fetchdf()
    
    count = int(count_result.iloc[0]['count']) if not count_result.empty else 0
    
    # Delete old tokens
    conn.execute(
        """
        DELETE FROM password_reset_tokens
        WHERE (used_at IS NOT NULL AND used_at < ?)
           OR (expires_at < ?)
        """,
        [cutoff, cutoff]
    )
    
    if count > 0:
        logger.info(f"Cleaned up {count} expired password reset tokens")
    
    return count

