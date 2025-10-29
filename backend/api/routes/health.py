"""
Health check endpoints - verify system status
"""
from fastapi import APIRouter, HTTPException
from core.database import get_connection
from core.ai_service import check_ollama_connection
from core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """
    Basic health check - returns OK if API is running
    """
    return {"status": "healthy", "version": "2.0.0"}


@router.get("/detailed")
async def detailed_health():
    """
    Detailed health check - verifies database and Ollama connections
    """
    health_status = {
        "api": "healthy",
        "database": "unknown",
        "ollama": "unknown",
    }

    # Check database
    try:
        conn = get_connection()
        conn.execute("SELECT 1").fetchone()
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"

    # Check Ollama
    try:
        ollama_status = check_ollama_connection()
        health_status["ollama"] = "connected" if ollama_status else "disconnected"
    except Exception as e:
        health_status["ollama"] = f"error: {str(e)}"

    # Overall status
    all_healthy = (
        health_status["database"] == "connected" and
        health_status["ollama"] == "connected"
    )

    if not all_healthy:
        raise HTTPException(status_code=503, detail=health_status)

    return health_status

