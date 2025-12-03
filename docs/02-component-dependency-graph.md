# Component Dependency Graph

## Overview

This document maps the dependencies between components in the Analytics Dashboard system.

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│                    (Application Entry Point)                     │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────────────┐
             │                                                      │
             ▼                                                      ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│   api/routes/*.py        │                    │   core/config.py         │
│   (Route Handlers)      │                    │   (Settings)              │
└────────────┬─────────────┘                    └──────────────┬───────────┘
             │                                                  │
             │                                                  │
             ▼                                                  │
┌──────────────────────────┐                    ┌──────────────┴───────────┐
│   services/*.py          │                    │   core/database.py        │
│   (Business Logic)       │                    │   (DB Connection)         │
└────────────┬─────────────┘                    └──────────────┬───────────┘
             │                                                  │
             │                                                  │
             ▼                                                  ▼
┌──────────────────────────┐                    ┌──────────────────────────┐
│   utils/*.py             │                    │   DuckDB                  │
│   (Utilities)            │                    │   (Database)              │
└──────────────────────────┘                    └───────────────────────────┘
```

## Detailed Component Dependencies

### 1. Application Entry Point

**File:** `backend/main.py`

**Dependencies:**
- `api.routes.*` - All route modules
- `core.config` - Application settings
- `core.database` - Database initialization
- `utils.logger` - Logging setup

**Dependents:**
- None (top-level entry point)

---

### 2. API Routes Layer

#### 2.1 Health Routes
**File:** `backend/api/routes/health.py`

**Dependencies:**
- `core.database` - Database connection checks
- `core.ai_service` - Ollama connection checks
- `core.config` - Configuration access

**Dependents:**
- `main.py` (router registration)

---

#### 2.2 Metrics Routes
**File:** `backend/api/routes/metrics.py`

**Dependencies:**
- `services.metrics_service` - Business logic
- `utils.validators` - Input validation
- `utils.sanitizers` - Input sanitization
- `utils.logger` - API logging
- `utils.error_handler` - Error formatting

**Dependents:**
- `main.py` (router registration)

---

#### 2.3 Upload Routes
**File:** `backend/api/routes/upload.py`

**Dependencies:**
- `services.upload_service` - CSV processing
- `services.database_reset` - Database reset
- `models.responses` - Response models
- `core.database` - Database operations

**Dependents:**
- `main.py` (router registration)

---

#### 2.4 Charts Routes
**File:** `backend/api/routes/charts.py`

**Dependencies:**
- `core.database` - Query execution
- `services.chart_service` - Chart data processing

**Dependents:**
- `main.py` (router registration)

---

#### 2.5 Chat Routes (AI)
**File:** `backend/api/routes/chat.py`

**Dependencies:**
- `core.ai_service` - AI/LLM integration
- `core.database` - Query execution
- `models.requests` - Request models
- `models.responses` - Response models

**Dependents:**
- `main.py` (router registration)

---

#### 2.6 Data Routes
**File:** `backend/api/routes/data_routes.py`

**Dependencies:**
- `services.data_service` - Transaction data
- `services.upload_service` - CSV upload
- `utils.validators` - Input validation
- `utils.sanitizers` - Input sanitization
- `utils.error_handler` - Error handling

**Dependents:**
- `main.py` (router registration)

---

#### 2.7 Data Status Routes
**File:** `backend/api/routes/data_status.py`

**Dependencies:**
- `core.database` - Database info
- `core.config` - Configuration

**Dependents:**
- `main.py` (router registration)

---

#### 2.8 Verification Routes
**File:** `backend/api/routes/verification.py`

**Dependencies:**
- `services.validation_service` - Data validation
- `core.database` - Database queries

**Dependents:**
- `main.py` (router registration)

---

### 3. Service Layer

#### 3.1 Metrics Service
**File:** `backend/services/metrics_service.py`

**Dependencies:**
- `core.database` - Query execution
- `core.config` - Configuration constants
- `utils.logger` - Logging

**Dependents:**
- `api/routes/metrics.py`

---

#### 3.2 Upload Service
**File:** `backend/services/upload_service.py`

**Dependencies:**
- `core.database` - Database operations
- `services.validation_service` - Data validation
- `services.data_service` - Column detection
- `core.config` - Configuration
- `utils.logger` - Logging

**Dependents:**
- `api/routes/upload.py`
- `api/routes/data_routes.py`

---

#### 3.3 Data Service
**File:** `backend/services/data_service.py`

**Dependencies:**
- `core.database` - Query execution
- `core.config` - Configuration

**Dependents:**
- `api/routes/data_routes.py`
- `services/upload_service.py`

---

#### 3.4 Chart Service
**File:** `backend/services/chart_service.py`

**Dependencies:**
- `core.database` - Query execution
- `services.metrics_service` - Metrics calculations

**Dependents:**
- `api/routes/charts.py`

---

#### 3.5 Validation Service
**File:** `backend/services/validation_service.py`

**Dependencies:**
- `core.database` - Query execution
- `core.config` - Validation rules

**Dependents:**
- `services/upload_service.py`
- `api/routes/verification.py`

---

#### 3.6 Database Reset Service
**File:** `backend/services/database_reset.py`

**Dependencies:**
- `core.database` - Database operations

**Dependents:**
- `api/routes/upload.py`

---

### 4. Core Layer

#### 4.1 Configuration
**File:** `backend/core/config.py`

**Dependencies:**
- `pydantic_settings` - Settings management
- `pathlib` - Path handling

**Dependents:**
- `main.py`
- `core/database.py`
- `core/ai_service.py`
- `services/*.py`
- `utils/validators.py`

---

#### 4.2 Database
**File:** `backend/core/database.py`

**Dependencies:**
- `duckdb` - Database library
- `pandas` - Data manipulation
- `core.config` - Database path
- `logging` - Logging

**Dependents:**
- `main.py` (initialization)
- All services
- All routes (indirectly)

---

#### 4.3 AI Service
**File:** `backend/core/ai_service.py`

**Dependencies:**
- `core.database` - Schema retrieval
- `core.config` - Ollama settings
- `langchain` - LLM integration
- `httpx` - HTTP client

**Dependents:**
- `api/routes/chat.py`
- `api/routes/health.py`

---

### 5. Utility Layer

#### 5.1 Validators
**File:** `backend/utils/validators.py`

**Dependencies:**
- `core.config` - Validation constants
- `datetime` - Date validation
- `re` - Pattern matching
- `logging` - Logging

**Dependents:**
- `api/routes/metrics.py`
- `api/routes/data_routes.py`

---

#### 5.2 Sanitizers
**File:** `backend/utils/sanitizers.py`

**Dependencies:**
- `re` - Pattern matching
- `logging` - Logging

**Dependents:**
- `api/routes/metrics.py`
- `api/routes/data_routes.py`

---

#### 5.3 Logger
**File:** `backend/utils/logger.py`

**Dependencies:**
- `logging` - Python logging
- `pathlib` - Log file paths

**Dependents:**
- `main.py`
- All services
- All routes

---

#### 5.4 Error Handler
**File:** `backend/utils/error_handler.py`

**Dependencies:**
- `utils.logger` - Error logging
- `traceback` - Stack traces

**Dependents:**
- `api/routes/metrics.py`
- `api/routes/data_routes.py`

---

### 6. Models Layer

#### 6.1 Request Models
**File:** `backend/models/requests.py`

**Dependencies:**
- `pydantic` - Data validation

**Dependents:**
- `api/routes/chat.py`

---

#### 6.2 Response Models
**File:** `backend/models/responses.py`

**Dependencies:**
- `pydantic` - Data validation

**Dependents:**
- `api/routes/upload.py`
- `api/routes/chat.py`

---

## Dependency Matrix

| Component | Depends On | Used By |
|-----------|-----------|---------|
| `main.py` | routes, config, database, logger | - |
| `api/routes/*` | services, utils, models | `main.py` |
| `services/*` | core, utils | `api/routes/*` |
| `core/config` | pydantic_settings | All |
| `core/database` | duckdb, config | All services, routes |
| `core/ai_service` | database, config, langchain | `api/routes/chat.py` |
| `utils/validators` | config | `api/routes/*` |
| `utils/sanitizers` | - | `api/routes/*` |
| `utils/logger` | logging | All |
| `utils/error_handler` | logger | `api/routes/*` |
| `models/*` | pydantic | `api/routes/*` |

## Circular Dependencies

**Status:** ✅ No circular dependencies detected

All dependencies flow in one direction:
1. Entry point (`main.py`)
2. Routes layer
3. Services layer
4. Core/Utils layer
5. External libraries

## External Dependencies

### Python Packages
- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **pydantic** - Data validation
- **duckdb** - Database
- **pandas** - Data manipulation
- **langchain** - AI/LLM integration
- **httpx** - HTTP client
- **requests** - HTTP client

### External Services
- **Ollama** - LLM service (optional, for AI features)

## Dependency Injection Points

### 1. Database Connection
- **Singleton pattern** in `core/database.py`
- Accessed via `get_connection()`
- Initialized in `main.py` lifespan

### 2. Configuration
- **Global settings instance** in `core/config.py`
- Accessed via `settings` object
- Loaded from environment variables

### 3. Logging
- **Module-level loggers** in `utils/logger.py`
- Accessed via `app_logger`, `api_logger`, etc.
- Initialized on import

## Testing Dependencies

### Test Files
- `backend/tests/test_api.py` - API endpoint tests
- `backend/tests/test_metrics_service.py` - Service tests
- `backend/tests/test_cleanup_verification.py` - Integration tests

**Dependencies:**
- `pytest` - Testing framework
- `httpx` - Test client
- Application modules (for testing)

## Build Dependencies

### Backend
- Python 3.11+
- `requirements.txt` - Python packages
- Virtual environment (recommended)

### Frontend
- Node.js 18+
- `package.json` - npm packages
- Vite build tool

### Docker
- Docker Engine
- Docker Compose
- Base images: `python:3.11-slim`, `ollama/ollama:latest`







