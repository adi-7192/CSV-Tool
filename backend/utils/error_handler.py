"""
Error Handler Utility - Centralized error handling for API endpoints

Provides custom error classes, error response formatting, and logging helpers.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM ERROR CLASSES
# ============================================================================

class ValidationError(Exception):
    """Raised when input validation fails"""
    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class DatabaseError(Exception):
    """Raised when database operations fail"""
    def __init__(self, message: str, query: Optional[str] = None):
        self.message = message
        self.query = query
        super().__init__(self.message)


class ColumnNotFoundError(Exception):
    """Raised when required column is not found in database"""
    def __init__(self, message: str, column_name: Optional[str] = None):
        self.message = message
        self.column_name = column_name
        super().__init__(self.message)


class DataProcessingError(Exception):
    """Raised when data processing/transformation fails"""
    def __init__(self, message: str, step: Optional[str] = None):
        self.message = message
        self.step = step
        super().__init__(self.message)


# ============================================================================
# ERROR RESPONSE FORMATTER
# ============================================================================

def format_error_response(
    error: Exception,
    status_code: int = 500,
    user_message: Optional[str] = None,
    include_details: bool = False
) -> Dict[str, Any]:
    """
    Format error response for API endpoints
    
    Args:
        error: Exception that occurred
        status_code: HTTP status code (400, 500, etc.)
        user_message: User-friendly error message (optional)
        include_details: Whether to include technical details (for debugging)
    
    Returns:
        Dictionary with error response structure
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    response = {
        "error": True,
        "status": status_code,
        "message": user_message or error_message,
        "error_type": error_type,
        "timestamp": datetime.now().isoformat()
    }
    
    # Include technical details only if requested (for debugging)
    if include_details:
        response["details"] = {
            "error_message": error_message,
            "traceback": traceback.format_exc()
        }
    
    return response


def handle_service_error(
    error: Exception,
    function_name: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle errors in service functions and return appropriate response
    
    Args:
        error: Exception that occurred
        function_name: Name of the function where error occurred
        context: Additional context (e.g., parameters passed to function)
    
    Returns:
        Dictionary with error response (to be returned by service function)
    """
    # Log error with context
    log_error(error, function_name, context)
    
    # Determine status code based on error type
    if isinstance(error, ValidationError):
        status_code = 400
        user_message = f"Validation error: {error.message}"
    elif isinstance(error, DatabaseError):
        status_code = 500
        user_message = "Database operation failed. Please try again later."
    elif isinstance(error, ColumnNotFoundError):
        status_code = 500
        user_message = f"Required column not found: {error.column_name or 'unknown'}"
    elif isinstance(error, DataProcessingError):
        status_code = 500
        user_message = f"Data processing failed: {error.message}"
    elif isinstance(error, ValueError):
        status_code = 400
        user_message = f"Invalid input: {str(error)}"
    elif isinstance(error, KeyError):
        status_code = 400
        user_message = f"Missing required field: {str(error)}"
    else:
        status_code = 500
        user_message = "An unexpected error occurred. Please try again later."
    
    # Return error response structure
    return {
        "error": True,
        "status": status_code,
        "message": user_message,
        "error_type": type(error).__name__,
        "function": function_name,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# LOGGING HELPER
# ============================================================================

def log_error(
    error: Exception,
    function_name: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log error with context information
    
    Args:
        error: Exception that occurred
        function_name: Name of the function where error occurred
        context: Additional context (e.g., parameters, state)
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Build log message
    log_msg = f"❌ ERROR in {function_name}: {error_type} - {error_message}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        log_msg += f" | Context: {context_str}"
    
    # Log error with traceback
    logger.error(log_msg)
    logger.error(traceback.format_exc())
    
    # Also print to console for immediate visibility
    print(f"\n{log_msg}\n")
    print(traceback.format_exc())


# ============================================================================
# DECORATOR FOR AUTOMATIC ERROR HANDLING
# ============================================================================

def handle_errors(
    return_empty_on_error: bool = False,
    default_return: Optional[Dict[str, Any]] = None
):
    """
    Decorator to automatically handle errors in service functions
    
    Usage:
        @handle_errors(return_empty_on_error=True)
        def my_function():
            # function code
            return result
    
    Args:
        return_empty_on_error: If True, return empty result instead of error dict
        default_return: Default return value if error occurs (if return_empty_on_error=True)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error
                log_error(e, func.__name__, {"args": str(args), "kwargs": str(kwargs)})
                
                # Return error response or empty result
                if return_empty_on_error:
                    if default_return is not None:
                        return default_return
                    # Return empty result based on function return type
                    return {}
                else:
                    return handle_service_error(e, func.__name__)
        return wrapper
    return decorator

