"""
FastAPI Backend for Analytics Dashboard

Inspired by Metabase architecture - Clean, modular, API-first design
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, upload, metrics, charts, chat, verification, data_status, data_routes, file_routes, user_api_keys
from core.config import settings
from core.database import init_database
from utils.logger import setup_logger, app_logger

# Initialize logging
logger = setup_logger('main')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Analytics API...")
    logger.info(f"Database path: {settings.DATABASE_PATH}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    init_database()  # Initialize database connections, create tables if needed
    logger.info("✅ Database initialized")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Analytics API...")


# Initialize FastAPI app
app = FastAPI(
    title="Analytics Dashboard API",
    description="FastAPI backend for business analytics with AI-powered insights",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
    redirect_slashes=False,  # Don't redirect /path to /path/ - prevents 307 issues
)

# CORS middleware (allow React frontend to access API)
# In development, allow all localhost ports; in production, use specific origins
cors_origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:5173",  # Vite dev server (default)
    "http://localhost:5174",  # Vite dev server (alternate port)
    "http://127.0.0.1:5173",  # Vite dev server (127.0.0.1)
    "http://127.0.0.1:5174",  # Vite dev server (127.0.0.1 alternate)
]

# Add production frontend URL if set
if settings.FRONTEND_URL:
    cors_origins.append(settings.FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # Allow any localhost port in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(file_routes.router, prefix="/api/files", tags=["Files"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(data_status.router, prefix="/api/data", tags=["Data Status"])
app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(user_api_keys.router, tags=["User API Keys"])


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

