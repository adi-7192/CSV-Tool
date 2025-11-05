"""
FastAPI Backend for Analytics Dashboard

Inspired by Metabase architecture - Clean, modular, API-first design
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, upload, metrics, charts, chat, verification, data_status
from core.config import settings
from core.database import init_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Analytics API...")
    init_database()  # Initialize database connections, create tables if needed
    print(f"✅ Database initialized: {settings.DATABASE_PATH}")
    print(f"✅ Ollama URL: {settings.OLLAMA_URL}")
    yield
    # Shutdown
    print("👋 Shutting down Analytics API...")


# Initialize FastAPI app
app = FastAPI(
    title="Analytics Dashboard API",
    description="FastAPI backend for business analytics with AI-powered insights",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# CORS middleware (allow React frontend to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        settings.FRONTEND_URL,    # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(data_status.router, prefix="/api/data", tags=["Data Status"])


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Analytics Dashboard API",
        "version": "2.0.0",
        "docs": "/api/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
    )

