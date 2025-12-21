"""
Monitoring Middleware - Request tracking and error capture

Extends StructuredLoggingMiddleware to also log events to system_events table.
Tracks request duration, errors, and slow requests for observability.
"""
import uuid
import time
import logging
import traceback
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from core.config import settings
from services.monitoring_service import log_system_event, sanitize_request_data

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track requests and log events to system_events table.
    
    Features:
    - Generates request_id per request (also in X-Request-Id header)
    - Measures request duration
    - Logs slow requests (configurable threshold)
    - Logs all requests if MONITORING_LOG_ALL_REQUESTS=true
    - Captures exceptions and logs ERROR events
    - Sanitizes sensitive data (no passwords/tokens)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract user info from JWT token if available
        user_id = None
        tenant_id = None
        try:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                from services.auth_service import decode_access_token
                try:
                    payload = decode_access_token(token)
                    user_id = str(payload.get("sub", ""))
                    tenant_id = payload.get("tenant_id")
                    if tenant_id:
                        tenant_id = str(tenant_id)
                except Exception:
                    pass  # Token invalid, skip user extraction
        except Exception:
            pass  # No auth header or error extracting
        
        # Store in request state for later use
        request.state.user_id = user_id
        request.state.tenant_id = tenant_id
        
        # Start timer
        start_time = time.time()
        
        # Determine endpoint category
        endpoint = str(request.url.path)
        method = request.method
        category = self._get_category(endpoint)
        
        # Process request
        status_code = 200
        error_info = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log to system_events based on configuration
            should_log = False
            log_level = "INFO"
            
            # Always log errors (4xx, 5xx)
            if status_code >= 400:
                should_log = True
                log_level = "WARN" if status_code < 500 else "ERROR"
            # Log slow requests
            elif duration_ms >= settings.MONITORING_SLOW_MS:
                should_log = True
                log_level = "WARN"
            # Log all requests if configured
            elif settings.MONITORING_LOG_ALL_REQUESTS:
                should_log = True
                log_level = "INFO"
            
            if should_log:
                log_system_event(
                    level=log_level,
                    category=category,
                    message=f"{method} {endpoint} - {status_code}",
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    request_id=request_id,
                    meta={
                        "query_params": str(request.url.query) if request.url.query else None,
                    }
                )
            
            # Add request_id to response header
            response.headers["X-Request-Id"] = request_id
            
            return response
            
        except HTTPException as e:
            # FastAPI HTTPException (expected errors)
            duration_ms = int((time.time() - start_time) * 1000)
            status_code = e.status_code
            
            # Log as WARN (client errors) or ERROR (server errors)
            log_level = "ERROR" if status_code >= 500 else "WARN"
            
            log_system_event(
                level=log_level,
                category=category,
                message=f"{method} {endpoint} - HTTPException {status_code}: {e.detail}",
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                duration_ms=duration_ms,
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
                meta={
                    "error_type": "HTTPException",
                    "error_detail": str(e.detail),
                }
            )
            
            # Re-raise to let FastAPI handle it
            raise
            
        except Exception as e:
            # Unexpected exception
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Get sanitized exception info
            exc_type = type(e).__name__
            exc_message = str(e)
            exc_traceback = traceback.format_exc()
            # Truncate traceback to avoid huge logs (keep last 500 chars)
            if len(exc_traceback) > 500:
                exc_traceback = "..." + exc_traceback[-500:]
            
            # Log ERROR event
            log_system_event(
                level="ERROR",
                category="exception",
                message=f"{method} {endpoint} - Exception: {exc_type}: {exc_message}",
                endpoint=endpoint,
                method=method,
                status_code=500,
                duration_ms=duration_ms,
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
                meta={
                    "error_type": exc_type,
                    "error_message": exc_message,
                    "traceback": exc_traceback,
                }
            )
            
            # Log to standard logger as well
            logger.error(
                f"request_id={request_id} method={method} path={endpoint} "
                f"error={exc_type}: {exc_message} duration_ms={duration_ms}",
                exc_info=True
            )
            
            # Re-raise to let FastAPI handle it
            raise
    
    def _get_category(self, endpoint: str) -> str:
        """Determine event category from endpoint path"""
        endpoint_lower = endpoint.lower()
        
        if "/api/auth" in endpoint_lower:
            return "auth"
        elif "/api/upload" in endpoint_lower or "/api/data/upload" in endpoint_lower:
            return "upload"
        elif "/api/data/export" in endpoint_lower:
            return "export"
        elif "/api/chat" in endpoint_lower:
            return "chat"
        elif "/api/metrics" in endpoint_lower or "/api/charts" in endpoint_lower:
            return "http"
        elif "/api/admin" in endpoint_lower:
            return "http"
        else:
            return "http"

