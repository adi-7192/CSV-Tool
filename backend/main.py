"""
FastAPI Backend for Analytics Dashboard

Inspired by Metabase architecture - Clean, modular, API-first design
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import health, upload, metrics, charts, chat, verification, data_status, data_routes, file_routes, user_api_keys, auth, users, admin_users, admin_monitoring, admin_stats
from core.config import settings
from core.database import init_database
from utils.logger import setup_logger, app_logger
from utils.monitoring_middleware import MonitoringMiddleware
from services.monitoring_service import cleanup_old_events
import logging

# Configure logging level from settings
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)

# Initialize logging
logger = setup_logger('main')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Analytics API...")
    logger.info(f"Environment: {settings.ENV}")
    logger.info(f"Database path: {settings.DATABASE_PATH}")
    logger.info(f"Ollama URL: {settings.OLLAMA_URL}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Validate email configuration in production
    if settings.ENV.lower() == "production":
        from services.email_service import validate_email_config
        is_valid, error_msg = validate_email_config()
        if not is_valid:
            logger.error(f"❌ Email configuration validation failed: {error_msg}")
            logger.error("Application will start but email sending will fail.")
            logger.error("Set ENV=development to allow dry-run mode, or configure SMTP settings.")
        else:
            logger.info("✅ Email configuration validated")
    else:
        logger.info("ℹ️  Development mode: Email will use dry-run (log to console)")
    
    # Validate Redis rate limiting in production
    if settings.REQUIRE_REDIS_RATE_LIMITING:
        try:
            from utils.rate_limiter import get_rate_limiter
            limiter = get_rate_limiter()
            if hasattr(limiter, 'redis_available') and limiter.redis_available:
                logger.info("✅ Redis rate limiting validated and connected")
            else:
                logger.error("❌ Redis rate limiting is required but Redis is unavailable")
                logger.error("Application will fail rate limit checks. Please configure Redis.")
        except RuntimeError as e:
            logger.error(f"❌ Redis rate limiting validation failed: {e}")
            logger.error("Application will start but rate-limited endpoints will return 503.")
            logger.error("Set REQUIRE_REDIS_RATE_LIMITING=false to allow in-memory fallback.")
    else:
        # Check if Redis is available but not required
        if settings.REDIS_URL:
            try:
                from utils.rate_limiter import get_rate_limiter
                limiter = get_rate_limiter()
                if hasattr(limiter, 'redis_available') and limiter.redis_available:
                    logger.info("ℹ️  Redis available (optional mode): Using Redis for rate limiting")
                else:
                    logger.info("ℹ️  Redis URL configured but unavailable: Using in-memory rate limiting")
            except Exception:
                logger.info("ℹ️  Redis URL configured but connection failed: Using in-memory rate limiting")
        else:
            logger.info("ℹ️  Development mode: Using in-memory rate limiting (Redis optional)")
    
    init_database()  # Initialize database connections, create tables if needed
    logger.info("✅ Database initialized")
    
    # Cleanup old monitoring events on startup
    if settings.MONITORING_ENABLED:
        try:
            deleted_count = cleanup_old_events()
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} old monitoring events")
        except Exception as e:
            logger.warning(f"Failed to cleanup old monitoring events: {e}")
    
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

# Monitoring middleware (add first to capture all requests and log to system_events)
if settings.MONITORING_ENABLED:
    app.add_middleware(MonitoringMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",  # Allow any localhost port in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
# Public routes (no authentication required)
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])

# Protected routes (authentication required - will be added via Depends in future)
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(file_routes.router, prefix="/api/files", tags=["Files"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"])
app.include_router(chat.router, prefix="/api/chat", tags=["AI Chat"])
app.include_router(verification.router, prefix="/api/verification", tags=["Verification"])
app.include_router(data_status.router, prefix="/api/data", tags=["Data Status"])
app.include_router(data_routes.router, prefix="/api/data", tags=["Data"])
app.include_router(user_api_keys.router, prefix="/api/user/api-key", tags=["User API Keys"])

# Admin routes (require admin role)
app.include_router(admin_users.router, prefix="/api/admin", tags=["Admin"])
app.include_router(admin_monitoring.router, prefix="/api/admin", tags=["Admin"])
app.include_router(admin_stats.router, prefix="/api/admin", tags=["Admin"])


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

