"""
Comprehensive Logging System
Logs to both console and file with different levels and detailed context
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
LOG_FILE = LOGS_DIR / "app.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"
API_LOG_FILE = LOGS_DIR / "api.log"
DB_LOG_FILE = LOGS_DIR / "database.log"


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (optional, defaults to app.log)
        level: File logging level (default: DEBUG)
        console_level: Console logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # File handler
    if log_file is None:
        log_file = LOG_FILE
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Detailed formatter (for file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simpler formatter (for console)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Setup main application logger
app_logger = setup_logger('app', LOG_FILE)
api_logger = setup_logger('api', API_LOG_FILE)
db_logger = setup_logger('database', DB_LOG_FILE)
error_logger = setup_logger('errors', ERROR_LOG_FILE, level=logging.ERROR, console_level=logging.ERROR)


def log_function_entry(logger: logging.Logger, function_name: str, params: Optional[dict] = None):
    """
    Log function entry with parameters
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        params: Function parameters (optional)
    """
    if params:
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.debug(f"Entering {function_name}() with params: {params_str}")
    else:
        logger.debug(f"Entering {function_name}()")


def log_function_exit(logger: logging.Logger, function_name: str, success: bool = True, result: Optional[any] = None):
    """
    Log function exit
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        success: Whether function completed successfully
        result: Function result (optional, for debugging)
    """
    if success:
        if result is not None:
            logger.debug(f"Exiting {function_name}() successfully. Result: {result}")
        else:
            logger.debug(f"Exiting {function_name}() successfully")
    else:
        logger.warning(f"Exiting {function_name}() with errors")


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    function_name: str,
    context: Optional[dict] = None
):
    """
    Log error with full context and traceback
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        function_name: Name of the function where error occurred
        context: Additional context (optional)
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    log_msg = f"Error in {function_name}(): {error_type} - {error_message}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        log_msg += f" | Context: {context_str}"
    
    logger.error(log_msg)
    logger.error(traceback.format_exc())


def log_api_request(logger: logging.Logger, method: str, endpoint: str, params: Optional[dict] = None):
    """
    Log API request
    
    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        params: Request parameters (optional)
    """
    if params:
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        logger.info(f"{method} {endpoint} | Params: {params_str}")
    else:
        logger.info(f"{method} {endpoint}")


def log_api_response(logger: logging.Logger, method: str, endpoint: str, status_code: int, duration_ms: Optional[float] = None):
    """
    Log API response
    
    Args:
        logger: Logger instance
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds (optional)
    """
    if duration_ms:
        logger.info(f"{method} {endpoint} | Status: {status_code} | Duration: {duration_ms:.2f}ms")
    else:
        logger.info(f"{method} {endpoint} | Status: {status_code}")


def log_database_query(logger: logging.Logger, query: str, params: Optional[dict] = None):
    """
    Log database query (for debugging)
    
    Args:
        logger: Logger instance
        query: SQL query
        params: Query parameters (optional)
    """
    if params:
        logger.debug(f"Executing query: {query[:200]}... | Params: {params}")
    else:
        logger.debug(f"Executing query: {query[:200]}...")


def log_data_processing(logger: logging.Logger, operation: str, record_count: Optional[int] = None):
    """
    Log data processing operation
    
    Args:
        logger: Logger instance
        operation: Operation description
        record_count: Number of records processed (optional)
    """
    if record_count is not None:
        logger.info(f"Processing: {operation} | Records: {record_count}")
    else:
        logger.info(f"Processing: {operation}")

