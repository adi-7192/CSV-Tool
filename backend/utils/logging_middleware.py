"""
Structured Logging Middleware

Adds request_id and tenant_id to all logs for better debugging in multi-tenant environments.
"""
import uuid
import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json

logger = logging.getLogger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add structured logging with request_id and tenant_id.
    
    Logs include:
    - request_id: Unique ID for each request
    - user_id: Authenticated user ID (if available)
    - tenant_id: User's tenant ID (if available)
    - method, path, status_code, duration_ms
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Extract user info from JWT token if available (for logging)
        # Note: We can't access user from request.state here since auth happens in dependencies
        # Instead, we'll extract from the Authorization header if present
        user_id = None
        tenant_id = None
        try:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                from services.auth_service import decode_access_token
                try:
                    payload = decode_access_token(token)
                    user_id = payload.get("sub")
                    tenant_id = payload.get("tenant_id")
                except Exception:
                    pass  # Token invalid, skip user extraction
        except Exception:
            pass  # No auth header or error extracting
        
        # Start timer
        start_time = time.time()
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": str(request.url.path),
            "user_id": user_id,
            "tenant_id": tenant_id,
        }
        
        # Add query params if present
        if request.url.query:
            log_data["query"] = request.url.query
        
        logger.info(f"request_id={request_id} method={request.method} path={request.url.path} user_id={user_id} tenant_id={tenant_id}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"request_id={request_id} method={request.method} path={request.url.path} "
                f"status={response.status_code} duration_ms={duration_ms:.2f} user_id={user_id} tenant_id={tenant_id}"
            )
            
            # Add request_id to response header for debugging
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"request_id={request_id} method={request.method} path={request.url.path} "
                f"error={str(e)} duration_ms={duration_ms:.2f} user_id={user_id} tenant_id={tenant_id}",
                exc_info=True
            )
            raise


def get_request_id(request: Request) -> str:
    """Get request ID from request state"""
    return getattr(request.state, 'request_id', 'unknown')

