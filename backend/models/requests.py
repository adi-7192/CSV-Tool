"""
Pydantic request models
"""
from pydantic import BaseModel
from typing import Optional, List
from datetime import date


class ChatRequest(BaseModel):
    """AI chat query request"""
    question: str
    conversation_id: Optional[str] = None


class DateRangeRequest(BaseModel):
    """Date range filter request"""
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    transaction_type: Optional[str] = None

