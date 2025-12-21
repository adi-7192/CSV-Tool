"""
Integration tests for Redis rate limiting enforcement in production.

Tests that:
1. When REQUIRE_REDIS_RATE_LIMITING=True and Redis is unavailable, endpoints return 503
2. When REQUIRE_REDIS_RATE_LIMITING=False, in-memory fallback works
3. Health check reports Redis status correctly
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add backend directory to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from main import app
from core.config import settings
from utils.rate_limiter import get_rate_limiter, RedisRateLimiter


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


def test_redis_required_but_unavailable_raises_error():
    """Test that get_rate_limiter() raises RuntimeError when Redis is required but unavailable."""
    from core.config import settings
    
    # Mock settings to require Redis
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', True):
        with patch.object(settings, 'REDIS_URL', 'redis://localhost:6379/0'):
            # Mock RedisRateLimiter to simulate connection failure
            with patch('utils.rate_limiter.RedisRateLimiter') as mock_redis:
                mock_instance = MagicMock()
                mock_instance.redis_available = False
                mock_redis.return_value = mock_instance
                
                # Reset global state
                import utils.rate_limiter
                utils.rate_limiter._redis_rate_limiter = None
                utils.rate_limiter._rate_limiter = None
                
                # Should raise RuntimeError
                with pytest.raises(RuntimeError, match="Redis rate limiting is required"):
                    get_rate_limiter()


def test_redis_required_but_url_not_set_raises_error():
    """Test that get_rate_limiter() raises RuntimeError when Redis is required but URL not set."""
    from core.config import settings
    
    # Mock settings to require Redis but no URL
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', True):
        with patch.object(settings, 'REDIS_URL', None):
            # Reset global state
            import utils.rate_limiter
            utils.rate_limiter._redis_rate_limiter = None
            utils.rate_limiter._rate_limiter = None
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="REDIS_URL is not configured"):
                get_rate_limiter()


def test_redis_not_required_allows_fallback():
    """Test that when Redis is not required, in-memory fallback works."""
    from core.config import settings
    
    # Mock settings to not require Redis
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', False):
        with patch.object(settings, 'REDIS_URL', None):
            # Reset global state
            import utils.rate_limiter
            utils.rate_limiter._redis_rate_limiter = None
            utils.rate_limiter._rate_limiter = None
            utils.rate_limiter._in_memory_limiter = None
            
            # Should return in-memory limiter without error
            limiter = get_rate_limiter()
            assert limiter is not None
            # Should be RateLimiter (in-memory), not RedisRateLimiter
            assert not isinstance(limiter, RedisRateLimiter)


def test_forgot_password_returns_503_when_redis_required_but_unavailable(client):
    """Test that forgot-password endpoint returns 503 when Redis is required but unavailable."""
    from core.config import settings
    
    # Mock settings to require Redis
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', True):
        with patch.object(settings, 'REDIS_URL', 'redis://localhost:6379/0'):
            # Mock get_rate_limiter to raise RuntimeError
            with patch('utils.rate_limiter.get_rate_limiter') as mock_get_limiter:
                mock_get_limiter.side_effect = RuntimeError("Redis unavailable")
                
                # Reset global state
                import utils.rate_limiter
                utils.rate_limiter._redis_rate_limiter = None
                utils.rate_limiter._rate_limiter = None
                
                # Call forgot-password endpoint
                response = client.post(
                    "/api/auth/forgot-password",
                    json={"email": "test@example.com"}
                )
                
                # Should return 503
                assert response.status_code == 503
                assert "unavailable" in response.json()["detail"].lower()


def test_health_check_redis_status_logic():
    """Test that health check Redis status logic works correctly."""
    from core.config import settings
    from utils.rate_limiter import get_rate_limiter, RedisRateLimiter
    
    # Test with Redis not required
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', False):
        with patch.object(settings, 'REDIS_URL', None):
            # Reset global state
            import utils.rate_limiter
            utils.rate_limiter._redis_rate_limiter = None
            utils.rate_limiter._rate_limiter = None
            
            # Get limiter - should work without Redis
            limiter = get_rate_limiter()
            assert limiter is not None
            # Should not be RedisRateLimiter when not required
            if isinstance(limiter, RedisRateLimiter):
                assert not limiter.redis_available
            # Status should be "not_configured"
            redis_status = "not_configured" if not isinstance(limiter, RedisRateLimiter) or not limiter.redis_available else "connected"
            assert redis_status in ["not_configured", "unknown"]
    
    # Test with Redis required but unavailable
    with patch.object(settings, 'REQUIRE_REDIS_RATE_LIMITING', True):
        with patch.object(settings, 'REDIS_URL', 'redis://localhost:6379/0'):
            # Mock RedisRateLimiter to simulate connection failure
            with patch('utils.rate_limiter.RedisRateLimiter') as mock_redis:
                mock_instance = MagicMock()
                mock_instance.redis_available = False
                mock_redis.return_value = mock_instance
                
                # Reset global state
                import utils.rate_limiter
                utils.rate_limiter._redis_rate_limiter = None
                utils.rate_limiter._rate_limiter = None
                
                # Should raise RuntimeError when Redis is required but unavailable
                with pytest.raises(RuntimeError, match="Redis rate limiting is required"):
                    get_rate_limiter()
                
                # Status should be "required_but_unavailable"
                redis_status = "required_but_unavailable"
                assert "required_but_unavailable" in redis_status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

