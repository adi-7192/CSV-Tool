"""
Pydantic response models
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


class ChatResponse(BaseModel):
    """AI chat response"""
    answer: str
    sql: Optional[str] = None
    conversation_id: Optional[str] = None

