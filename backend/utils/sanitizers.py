"""
Input Sanitization Module
Sanitizes user inputs to prevent injection attacks and handle edge cases
"""
from typing import Optional, Any
import re
import logging

logger = logging.getLogger(__name__)


def sanitize_string(value: Optional[str], max_length: Optional[int] = None, allow_empty: bool = False) -> Optional[str]:
    """
    Sanitize a string input
    
    Args:
        value: String value to sanitize
        max_length: Maximum allowed length (None = no limit)
        allow_empty: Whether empty strings are allowed
    
    Returns:
        Sanitized string or None if empty and not allowed
    """
    if value is None:
        return None
    
    # Convert to string if not already
    if not isinstance(value, str):
        value = str(value)
    
    # Trim whitespace
    value = value.strip()
    
    # Check if empty
    if not value and not allow_empty:
        return None
    
    # Truncate if exceeds max length
    if max_length is not None and len(value) > max_length:
        logger.warning(f"String truncated from {len(value)} to {max_length} characters")
        value = value[:max_length]
    
    return value


def sanitize_sku(sku: Optional[str]) -> Optional[str]:
    """
    Sanitize SKU string
    
    Args:
        sku: SKU string
    
    Returns:
        Sanitized SKU or None
    """
    if not sku:
        return None
    
    # Trim and convert to uppercase
    sku = sku.strip().upper()
    
    # Remove any special characters except underscore and hyphen
    sku = re.sub(r'[^A-Z0-9_-]', '', sku)
    
    return sku if sku else None


def sanitize_city_name(city: Optional[str]) -> Optional[str]:
    """
    Sanitize city name
    
    Args:
        city: City name string
    
    Returns:
        Sanitized city name or None
    """
    if not city:
        return None
    
    # Trim whitespace and convert to title case
    city = city.strip().title()
    
    return city if city else None


def sanitize_transaction_type(transaction_type: Optional[str]) -> Optional[str]:
    """
    Sanitize transaction type
    
    Args:
        transaction_type: Transaction type string
    
    Returns:
        Sanitized transaction type or None
    """
    if not transaction_type:
        return None
    
    # Trim and capitalize first letter
    transaction_type = transaction_type.strip()
    
    # Normalize common variations
    type_mapping = {
        'all': 'All Transactions',
        'all transactions': 'All Transactions',
        'shipment': 'Shipment',
        'refund': 'Refund',
        'cancellation': 'Cancellation',
        'cancel': 'Cancellation',
        'free replacement': 'Free Replacement',
        'freereplacement': 'Free Replacement',
    }
    
    normalized = transaction_type.lower()
    if normalized in type_mapping:
        return type_mapping[normalized]
    
    return transaction_type


def sanitize_date_string(date_str: Optional[str]) -> Optional[str]:
    """
    Sanitize date string
    
    Args:
        date_str: Date string (YYYY-MM-DD)
    
    Returns:
        Sanitized date string or None
    """
    if not date_str:
        return None
    
    # Trim whitespace
    date_str = date_str.strip()
    
    # Remove any time component if present
    if ' ' in date_str:
        date_str = date_str.split(' ')[0]
    
    # Remove any timezone info
    if '+' in date_str or 'T' in date_str:
        date_str = date_str.split('T')[0].split('+')[0]
    
    return date_str if date_str else None


def sanitize_integer(value: Any, default: Optional[int] = None) -> Optional[int]:
    """
    Safely convert value to integer
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value or default
    """
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to integer, using default: {default}")
        return default


def sanitize_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely convert value to float
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Failed to convert {value} to float, using default: {default}")
        return default


def sanitize_sql_string(value: Optional[str]) -> str:
    """
    Sanitize string for SQL queries (basic protection)
    Note: This is a basic sanitizer. Always use parameterized queries!
    
    Args:
        value: String value
    
    Returns:
        Sanitized string
    """
    if not value:
        return ""
    
    # Remove SQL injection patterns (basic protection)
    # Note: This is NOT a replacement for parameterized queries!
    dangerous_patterns = [
        r"';?\s*--",
        r"';?\s*/\*",
        r"';?\s*DROP\s+TABLE",
        r"';?\s*DELETE\s+FROM",
        r"';?\s*UPDATE\s+.*\s+SET",
        r"';?\s*INSERT\s+INTO",
    ]
    
    sanitized = str(value)
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    return sanitized

