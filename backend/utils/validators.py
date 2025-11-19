"""
Input Validation Module
Validates all user inputs before processing to prevent SQL injection and invalid data
"""
from datetime import datetime, timedelta
from typing import Optional, List
import re
import logging

from core.config import (
    APPROVED_CITIES,
    VALID_TRANSACTION_TYPES,
    VALID_GROUP_BY,
    MAX_DATE_RANGE_DAYS,
    MAX_HISTORICAL_YEARS,
    MAX_SKU_LENGTH,
)

logger = logging.getLogger(__name__)

# Convert config lists to sets for faster lookup
APPROVED_CITIES_SET = set(APPROVED_CITIES)
VALID_TRANSACTION_TYPES_SET = set(VALID_TRANSACTION_TYPES)
VALID_GROUP_BY_SET = set(VALID_GROUP_BY)


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range parameters
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If dates are invalid
    """
    try:
        # Parse dates
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Check start_date <= end_date
        if start > end:
            raise ValueError("Start date must be before or equal to end date")
        
        # Check date range doesn't exceed max days
        if (end - start).days > MAX_DATE_RANGE_DAYS:
            raise ValueError(f"Date range cannot exceed {MAX_DATE_RANGE_DAYS} days (1 year)")
        
        # Check dates are not too far in the past
        max_years_ago = datetime.now() - timedelta(days=MAX_HISTORICAL_YEARS*365)
        if start < max_years_ago:
            raise ValueError(f"Start date cannot be more than {MAX_HISTORICAL_YEARS} years in the past")
        
        # Check dates are not in the future
        if end > datetime.now():
            raise ValueError("End date cannot be in the future")
        
        return True
    
    except ValueError as e:
        # Re-raise if it's our validation error
        if "must be" in str(e) or "cannot" in str(e):
            raise
        # Otherwise, it's a parsing error
        raise ValueError(f"Invalid date format. Expected YYYY-MM-DD, got: {start_date} or {end_date}")


def validate_date(date_str: str, param_name: str = "date") -> bool:
    """
    Validate a single date string
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        param_name: Name of parameter for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If date is invalid
    """
    if not date_str:
        raise ValueError(f"{param_name} cannot be empty")
    
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Check date is not too far in the past
        max_years_ago = datetime.now() - timedelta(days=MAX_HISTORICAL_YEARS*365)
        if date < max_years_ago:
            raise ValueError(f"{param_name} cannot be more than {MAX_HISTORICAL_YEARS} years in the past")
        
        # Check date is not in the future
        if date > datetime.now():
            raise ValueError(f"{param_name} cannot be in the future")
        
        return True
    
    except ValueError as e:
        if "cannot be" in str(e):
            raise
        raise ValueError(f"Invalid {param_name} format. Expected YYYY-MM-DD, got: {date_str}")


def validate_sku(sku: str) -> bool:
    """
    Validate SKU format
    
    Args:
        sku: SKU string
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If SKU is invalid
    """
    if not sku:
        raise ValueError("SKU cannot be empty")
    
    if len(sku) > MAX_SKU_LENGTH:
        raise ValueError(f"SKU cannot exceed {MAX_SKU_LENGTH} characters")
    
    # Allow alphanumeric, underscore, and hyphen
    if not re.match(r'^[A-Za-z0-9_-]+$', sku):
        raise ValueError("SKU must contain only alphanumeric characters, underscores, and hyphens")
    
    return True


def validate_city_name(city: str) -> bool:
    """
    Validate city name against approved list
    
    Args:
        city: City name string
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If city is invalid
    """
    if not city:
        raise ValueError("City name cannot be empty")
    
    # Normalize city name (lowercase, strip whitespace)
    normalized = city.lower().strip()
    
    if normalized not in APPROVED_CITIES_SET:
        raise ValueError(f"Invalid city name: {city}. Must be one of the approved cities.")
    
    return True


def validate_transaction_type(transaction_type: str) -> bool:
    """
    Validate transaction type
    
    Args:
        transaction_type: Transaction type string
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If transaction type is invalid
    """
    if not transaction_type:
        raise ValueError("Transaction type cannot be empty")
    
    if transaction_type not in VALID_TRANSACTION_TYPES_SET:
        raise ValueError(
            f"Invalid transaction type: {transaction_type}. "
            f"Must be one of: {', '.join(sorted(VALID_TRANSACTION_TYPES_SET))}"
        )
    
    return True


def validate_group_by(group_by: str) -> bool:
    """
    Validate group_by parameter
    
    Args:
        group_by: Group by value (day, week, month)
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If group_by is invalid
    """
    if not group_by:
        raise ValueError("group_by cannot be empty")
    
    if group_by not in VALID_GROUP_BY_SET:
        raise ValueError(f"Invalid group_by: {group_by}. Must be one of: {', '.join(VALID_GROUP_BY_SET)}")
    
    return True


def validate_positive_number(value: float, param_name: str = "number", min_value: float = 0, max_value: Optional[float] = None) -> bool:
    """
    Validate a positive number within range
    
    Args:
        value: Number to validate
        param_name: Name of parameter for error messages
        min_value: Minimum allowed value (default: 0)
        max_value: Maximum allowed value (None = no limit)
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If number is invalid
    """
    if value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}")
    
    return True


def validate_integer(value: int, param_name: str = "integer", min_value: int = 1, max_value: Optional[int] = None) -> bool:
    """
    Validate an integer within range
    
    Args:
        value: Integer to validate
        param_name: Name of parameter for error messages
        min_value: Minimum allowed value (default: 1)
        max_value: Maximum allowed value (None = no limit)
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If integer is invalid
    """
    if not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer")
    
    if value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}")
    
    return True


def validate_limit(limit: int) -> bool:
    """
    Validate limit parameter for pagination
    
    Args:
        limit: Limit value
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If limit is invalid
    """
    return validate_integer(limit, "limit", min_value=1, max_value=100)


def validate_page(page: int) -> bool:
    """
    Validate page parameter for pagination
    
    Args:
        page: Page number
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If page is invalid
    """
    return validate_integer(page, "page", min_value=1, max_value=None)

