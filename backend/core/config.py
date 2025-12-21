"""
Configuration management using Pydantic Settings

Loads from environment variables with sensible defaults
"""
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # API Settings
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    ENV: str = "development"  # "development" or "production"

    # Database (relative to project root)
    DATABASE_PATH: str = "../data/analytics.duckdb"

    # Ollama/LLM
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"
    OLLAMA_TIMEOUT: int = 30

    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 100
    UPLOAD_DIR: str = "../data/raw"

    # Frontend
    FRONTEND_URL: str = "http://localhost:5173"  # Default to Vite dev server port

    # API Key Encryption
    API_KEY_ENCRYPTION_KEY: Optional[str] = None

    # JWT Authentication
    JWT_SECRET: str = "your-secret-key-change-in-production"  # Should be set via env var
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRES_IN: int = 86400  # 24 hours in seconds

    # Password Reset
    PASSWORD_RESET_TOKEN_EXPIRES_MINUTES: int = 15  # Token validity in minutes
    PASSWORD_MIN_LENGTH: int = 8
    
    # Email/SMTP Configuration
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""  # Primary field name
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""  # Primary field name
    SMTP_FROM_NAME: str = "Datadost Analytics"
    SMTP_USE_TLS: bool = True
    SMTP_USE_SSL: bool = False
    
    # Backward compatibility: support both old and new env var names
    SMTP_USER: str = ""  # Legacy name (will be set from env if present)
    SMTP_FROM_EMAIL: str = ""  # Legacy name (will be set from env if present)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If legacy env vars are set but new ones aren't, use legacy values
        if not self.SMTP_USERNAME and self.SMTP_USER:
            self.SMTP_USERNAME = self.SMTP_USER
        if not self.SMTP_FROM and self.SMTP_FROM_EMAIL:
            self.SMTP_FROM = self.SMTP_FROM_EMAIL
    
    # Rate Limiting (requests per window)
    RATE_LIMIT_FORGOT_PASSWORD_PER_IP: int = 5  # per 15 minutes
    RATE_LIMIT_FORGOT_PASSWORD_PER_EMAIL: int = 3  # per 15 minutes
    RATE_LIMIT_WINDOW_MINUTES: int = 15
    
    # Redis (optional - for distributed rate limiting)
    REDIS_URL: Optional[str] = None  # e.g., "redis://localhost:6379/0"
    REQUIRE_REDIS_RATE_LIMITING: bool = False  # Require Redis for rate limiting (auto-set based on ENV)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # If legacy env vars are set but new ones aren't, use legacy values
        if not self.SMTP_USERNAME and self.SMTP_USER:
            self.SMTP_USERNAME = self.SMTP_USER
        if not self.SMTP_FROM and self.SMTP_FROM_EMAIL:
            self.SMTP_FROM = self.SMTP_FROM_EMAIL
        
        # Auto-set REQUIRE_REDIS_RATE_LIMITING based on ENV if not explicitly set
        if "REQUIRE_REDIS_RATE_LIMITING" not in kwargs:
            if self.ENV.lower() == "production":
                self.REQUIRE_REDIS_RATE_LIMITING = True
            else:
                self.REQUIRE_REDIS_RATE_LIMITING = False

    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Monitoring/Observability
    MONITORING_ENABLED: bool = True
    MONITORING_LOG_ALL_REQUESTS: bool = False  # If False, only log slow requests and errors
    MONITORING_SLOW_MS: int = 1000  # Threshold in milliseconds for slow requests
    MONITORING_RETENTION_DAYS: int = 7  # Days to keep events before cleanup

    class Config:
        env_file = "../.env"  # Look for .env in project root
        case_sensitive = True
        # Allow extra environment variables (needed for API_KEY_ENCRYPTION_KEY)
        extra = "allow"


# Global settings instance
settings = Settings()


# ============================================================================
# APPLICATION CONSTANTS
# ============================================================================

# Pagination
ITEMS_PER_PAGE: int = 10
MAX_PAGE_SIZE: int = 100
DEFAULT_PAGE_SIZE: int = 50

# Date Range
MAX_DATE_RANGE_DAYS: int = 365
DEFAULT_DATE_RANGE_DAYS: int = 30
MAX_HISTORICAL_YEARS: int = 5

# Product Limits
TOP_PRODUCTS_LIMIT: int = 10
MAX_PRODUCTS_LIMIT: int = 100

# SKU Validation
MAX_SKU_LENGTH: int = 50
MIN_SKU_LENGTH: int = 1

# Approved Cities (normalized lowercase for matching)
APPROVED_CITIES: List[str] = [
    'bangalore', 'mumbai', 'delhi', 'hyderabad', 'chennai', 'kolkata',
    'pune', 'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur',
    'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna',
    'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad',
    'meerut', 'rajkot', 'varanasi', 'srinagar', 'amritsar', 'navi mumbai',
    'gurugram', 'noida', 'bengaluru', 'new delhi'
]

# City Name Mapping (for normalization)
CITY_NAME_MAPPING: dict = {
    'bangalore': 'Bangalore',
    'bengaluru': 'Bangalore',
    'mumbai': 'Mumbai',
    'delhi': 'Delhi',
    'new delhi': 'Delhi',
    'hyderabad': 'Hyderabad',
    'chennai': 'Chennai',
    'kolkata': 'Kolkata',
    'pune': 'Pune',
    'ahmedabad': 'Ahmedabad',
    'jaipur': 'Jaipur',
    'surat': 'Surat',
    'lucknow': 'Lucknow',
    'kanpur': 'Kanpur',
    'nagpur': 'Nagpur',
    'indore': 'Indore',
    'thane': 'Thane',
    'bhopal': 'Bhopal',
    'visakhapatnam': 'Visakhapatnam',
    'patna': 'Patna',
    'vadodara': 'Vadodara',
    'ghaziabad': 'Ghaziabad',
    'ludhiana': 'Ludhiana',
    'agra': 'Agra',
    'nashik': 'Nashik',
    'faridabad': 'Faridabad',
    'meerut': 'Meerut',
    'rajkot': 'Rajkot',
    'varanasi': 'Varanasi',
    'srinagar': 'Srinagar',
    'amritsar': 'Amritsar',
    'navi mumbai': 'Navi Mumbai',
    'gurugram': 'Gurugram',
    'noida': 'Noida',
}

# Valid Transaction Types
VALID_TRANSACTION_TYPES: List[str] = [
    'Shipment',
    'Refund',
    'Cancellation',
    'Cancel',
    'Free Replacement',
    'FreeReplacement',
    'All Transactions',
    'all'
]

# Valid Group By Values
VALID_GROUP_BY: List[str] = ['day', 'week', 'month']

# Quality Issues Thresholds
REFUND_PERCENTAGE_THRESHOLDS: dict = {
    'high': 15.0,
    'medium': 10.0,
    'low': 5.0
}

CANCEL_PERCENTAGE_THRESHOLDS: dict = {
    'high': 20.0,
    'medium': 10.0,
    'low': 5.0
}

# Movers & Decliners Thresholds
MOVER_GROWTH_THRESHOLD: float = 30.0
DECLINER_GROWTH_THRESHOLD: float = -30.0


