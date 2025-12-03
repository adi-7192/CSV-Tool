"""
Pydantic models for API key requests
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating/updating an API key"""
    
    user_id: str = Field(..., description="User identifier")
    provider: Literal['openai', 'anthropic'] = Field(..., description="API provider")
    api_key: str = Field(..., min_length=1, description="The API key to store (will be encrypted)")
    
    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty"""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()


class GetAPIKeyRequest(BaseModel):
    """Request model for getting an API key"""
    
    user_id: str = Field(..., description="User identifier")
    provider: Literal['openai', 'anthropic'] = Field(..., description="API provider")


class DeleteAPIKeyRequest(BaseModel):
    """Request model for deleting an API key"""
    
    user_id: str = Field(..., description="User identifier")
    provider: Literal['openai', 'anthropic'] = Field(..., description="API provider")

