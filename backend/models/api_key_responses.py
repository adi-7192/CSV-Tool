"""
Pydantic models for API key responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class APIKeyInfo(BaseModel):
    """API key information (masked for frontend)"""
    
    id: str = Field(..., description="Unique key identifier")
    user_id: str = Field(..., description="User identifier")
    provider: str = Field(..., description="API provider (openai, anthropic, or gemini)")
    masked_key: str = Field(..., description="Masked API key (first 6, last 4 chars)")
    enabled: bool = Field(True, description="Whether this API key is enabled for use")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")


class APIKeyResponse(BaseModel):
    """Response for API key operations"""
    
    success: bool = Field(..., description="Operation success status")
    data: Optional[APIKeyInfo] = Field(None, description="API key information")
    message: Optional[str] = Field(None, description="Response message")


class APIKeyListResponse(BaseModel):
    """Response for listing all API keys"""
    
    success: bool = Field(..., description="Operation success status")
    data: List[APIKeyInfo] = Field(..., description="List of API keys")
    count: int = Field(..., description="Number of keys")


class EncryptionStatusResponse(BaseModel):
    """Response for encryption verification"""
    
    encryption_working: bool = Field(..., description="Whether encryption is working")
    message: str = Field(..., description="Status message")

