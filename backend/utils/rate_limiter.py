"""
Rate limiter for password reset endpoints.

Supports both Redis (distributed) and in-memory (fallback) rate limiting.
Automatically falls back to in-memory if Redis is not available.
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from collections import defaultdict
import threading
import logging

from core.config import settings

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory rate limiter")


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


class RedisRateLimiter:
    """
    Redis-based rate limiter for distributed rate limiting.
    
    Uses Redis INCR + EXPIRE for atomic rate limiting across multiple processes.
    Falls back to in-memory limiter if Redis is unavailable.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.redis_available = False
        
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"Redis rate limiter initialized with URL: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory limiter.")
                self.redis_available = False
        else:
            if not REDIS_AVAILABLE:
                logger.warning("Redis library not installed. Using in-memory limiter.")
            else:
                logger.info("REDIS_URL not set. Using in-memory limiter.")
    
    def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_minutes: int
    ) -> Tuple[bool, int]:
        """
        Check if key is rate limited using Redis.
        
        Uses atomic INCR + EXPIRE pattern for thread-safe rate limiting.
        
        Args:
            key: Identifier (e.g., IP address or email)
            max_requests: Maximum requests allowed in window
            window_minutes: Time window in minutes
            
        Returns:
            Tuple of (is_limited: bool, requests_remaining: int)
        """
        if not self.redis_available or not self.redis_client:
            # Fallback to in-memory
            return _in_memory_limiter.is_rate_limited(key, max_requests, window_minutes)
        
        try:
            redis_key = f"ratelimit:{key}"
            window_seconds = window_minutes * 60
            
            # Atomic increment
            current_count = self.redis_client.incr(redis_key)
            
            # Set expiration on first request
            if current_count == 1:
                self.redis_client.expire(redis_key, window_seconds)
            
            # Check if limit exceeded
            if current_count > max_requests:
                logger.warning(f"Rate limit exceeded for key: {key[:20]}... (count: {current_count})")
                return True, 0
            
            requests_remaining = max(0, max_requests - current_count)
            return False, requests_remaining
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}. Falling back to in-memory.")
            # Fallback to in-memory
            return _in_memory_limiter.is_rate_limited(key, max_requests, window_minutes)
    
    def record_request(self, key: str) -> None:
        """
        Record a request for the given key.
        
        Note: In Redis implementation, recording is done in is_rate_limited()
        via INCR, so this is a no-op for Redis but kept for API compatibility.
        """
        if not self.redis_available or not self.redis_client:
            # Fallback to in-memory
            _in_memory_limiter.record_request(key)
    
    def reset(self, key: str) -> None:
        """Reset rate limit for a key (useful for testing)."""
        if self.redis_available and self.redis_client:
            try:
                redis_key = f"ratelimit:{key}"
                self.redis_client.delete(redis_key)
            except Exception as e:
                logger.error(f"Redis reset failed: {e}")
        else:
            _in_memory_limiter.reset(key)


# Global rate limiter instances
_rate_limiter: Optional[RateLimiter] = None
_redis_rate_limiter: Optional[RedisRateLimiter] = None
_in_memory_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Returns RedisRateLimiter if Redis is available, otherwise RateLimiter (in-memory).
    """
    global _rate_limiter, _redis_rate_limiter, _in_memory_limiter
    
    # Initialize in-memory limiter (always available as fallback)
    if _in_memory_limiter is None:
        _in_memory_limiter = RateLimiter()
    
    # Try Redis if URL is configured
    if settings.REDIS_URL:
        if _redis_rate_limiter is None:
            _redis_rate_limiter = RedisRateLimiter(redis_url=settings.REDIS_URL)
        
        # Use Redis if available, otherwise fallback to in-memory
        if _redis_rate_limiter.redis_available:
            _rate_limiter = _redis_rate_limiter
            return _rate_limiter
    
    # Use in-memory limiter
    _rate_limiter = _in_memory_limiter
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

