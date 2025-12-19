"""
Monitoring Service - System event logging and observability

Provides structured logging of system events to the system_events table.
Used by middleware and error handlers to track requests, errors, and performance.
"""
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from core.database import get_connection, table_exists
from core.config import settings

logger = logging.getLogger(__name__)


def log_system_event(
    level: str,
    category: str,
    message: str,
    endpoint: Optional[str] = None,
    method: Optional[str] = None,
    status_code: Optional[int] = None,
    duration_ms: Optional[int] = None,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a system event to the system_events table.
    
    Args:
        level: Event level (INFO, WARN, ERROR)
        category: Event category (http, exception, auth, upload, export, chat, db, redis)
        message: Human-readable message
        endpoint: API endpoint path (e.g., "/api/metrics")
        method: HTTP method (GET, POST, etc.)
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        tenant_id: Tenant ID (for multi-tenant isolation)
        user_id: User ID
        request_id: Unique request ID (for tracing)
        meta: Additional metadata as dict (will be serialized to JSON)
    
    Note: This function does NOT log sensitive data (passwords, JWTs, tokens).
    The meta dict should be pre-sanitized by the caller.
    """
    if not settings.MONITORING_ENABLED:
        return
    
    try:
        if not table_exists('system_events'):
            logger.warning("system_events table does not exist - skipping event log")
            return
        
        conn = get_connection()
        
        # Serialize meta to JSON string if provided
        meta_json = None
        if meta:
            try:
                meta_json = json.dumps(meta)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize meta to JSON: {e}")
                meta_json = json.dumps({"error": "Failed to serialize metadata"})
        
        # Insert event
        insert_sql = """
        INSERT INTO system_events (
            created_at, level, category, message, endpoint, method,
            status_code, duration_ms, tenant_id, user_id, request_id, meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        conn.execute(insert_sql, [
            datetime.now(),
            level,
            category,
            message,
            endpoint,
            method,
            status_code,
            duration_ms,
            tenant_id,
            user_id,
            request_id,
            meta_json,
        ])
        
    except Exception as e:
        # Don't let monitoring failures break the application
        logger.error(f"Failed to log system event: {e}", exc_info=True)


def cleanup_old_events() -> int:
    """
    Delete events older than MONITORING_RETENTION_DAYS.
    
    Returns:
        Number of events deleted
    """
    if not settings.MONITORING_ENABLED:
        return 0
    
    try:
        if not table_exists('system_events'):
            return 0
        
        conn = get_connection()
        cutoff_date = datetime.now() - timedelta(days=settings.MONITORING_RETENTION_DAYS)
        
        # Count before deletion
        count_sql = "SELECT COUNT(*) as count FROM system_events WHERE created_at < ?"
        count_df = conn.execute(count_sql, [cutoff_date]).fetchdf()
        deleted_count = int(count_df.iloc[0]['count']) if not count_df.empty else 0
        
        if deleted_count > 0:
            delete_sql = "DELETE FROM system_events WHERE created_at < ?"
            conn.execute(delete_sql, [cutoff_date])
            logger.info(f"Cleaned up {deleted_count} old monitoring events (older than {settings.MONITORING_RETENTION_DAYS} days)")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Failed to cleanup old events: {e}", exc_info=True)
        return 0


def sanitize_request_data(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize request data by removing sensitive fields.
    
    Removes:
    - Authorization headers
    - Cookies
    - Password fields
    - Token fields
    - JWT tokens
    
    Args:
        request_data: Raw request data dict
    
    Returns:
        Sanitized dict with sensitive data removed
    """
    sensitive_keys = [
        'authorization', 'cookie', 'password', 'token', 'jwt',
        'access_token', 'refresh_token', 'api_key', 'secret',
        'reset_token', 'auth_token', 'bearer',
    ]
    
    sanitized = {}
    for key, value in request_data.items():
        key_lower = str(key).lower()
        # Skip sensitive keys
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_request_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_request_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized

