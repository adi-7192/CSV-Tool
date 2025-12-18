"""
In-memory rate limiter for password reset endpoints.

Simple sliding window rate limiter using in-memory storage.
For production, consider using Redis for distributed rate limiting.
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple
from collections import defaultdict
import threading
import logging

from core.config import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter with sliding window.
    
    Thread-safe implementation using locks.
    Tracks requests per key (IP or email) within a time window.
    """
    
    def __init__(self):
        # Dictionary: key -> list of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        
    def _clean_old_requests(self, key: str, window_seconds: int) -> None:
        """Remove requests older than the window."""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]
    
    def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_minutes: int
    ) -> Tuple[bool, int]:
        """
        Check if key is rate limited.
        
        Args:
            key: Identifier (e.g., IP address or email)
            max_requests: Maximum requests allowed in window
            window_minutes: Time window in minutes
            
        Returns:
            Tuple of (is_limited: bool, requests_remaining: int)
        """
        window_seconds = window_minutes * 60
        
        with self._lock:
            self._clean_old_requests(key, window_seconds)
            current_count = len(self._requests[key])
            
            if current_count >= max_requests:
                logger.warning(f"Rate limit exceeded for key: {key[:20]}...")
                return True, 0
            
            return False, max_requests - current_count
    
    def record_request(self, key: str) -> None:
        """
        Record a request for the given key.
        
        Args:
            key: Identifier (e.g., IP address or email)
        """
        with self._lock:
            self._requests[key].append(datetime.utcnow())
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a key (useful for testing)."""
        with self._lock:
            if key in self._requests:
                del self._requests[key]
    
    def cleanup(self) -> None:
        """Remove all expired entries to free memory."""
        # Use the default window for cleanup
        window_seconds = settings.RATE_LIMIT_WINDOW_MINUTES * 60
        
        with self._lock:
            keys_to_clean = list(self._requests.keys())
            for key in keys_to_clean:
                self._clean_old_requests(key, window_seconds)
                # Remove empty keys
                if not self._requests[key]:
                    del self._requests[key]


# Global rate limiter instance
_rate_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def check_forgot_password_rate_limit(
    ip_address: str,
    email: str
) -> Tuple[bool, str]:
    """
    Check rate limits for forgot-password endpoint.
    
    Checks both IP and email rate limits.
    
    Args:
        ip_address: Client IP address
        email: Email address being requested
        
    Returns:
        Tuple of (is_allowed: bool, error_message: str or None)
    """
    limiter = get_rate_limiter()
    
    # Check IP rate limit
    ip_key = f"forgot_password:ip:{ip_address}"
    ip_limited, _ = limiter.is_rate_limited(
        ip_key,
        settings.RATE_LIMIT_FORGOT_PASSWORD_PER_IP,
        settings.RATE_LIMIT_WINDOW_MINUTES
    )
    
    if ip_limited:
        logger.warning(f"IP rate limit hit for forgot-password: {ip_address}")
        return False, "Too many requests. Please try again later."
    
    # Check email rate limit
    email_key = f"forgot_password:email:{email.lower()}"
    email_limited, _ = limiter.is_rate_limited(
        email_key,
        settings.RATE_LIMIT_FORGOT_PASSWORD_PER_EMAIL,
        settings.RATE_LIMIT_WINDOW_MINUTES
    )
    
    if email_limited:
        logger.warning(f"Email rate limit hit for forgot-password: {email[:20]}...")
        return False, "Too many requests for this email. Please try again later."
    
    # Record the requests
    limiter.record_request(ip_key)
    limiter.record_request(email_key)
    
    return True, None

