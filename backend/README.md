# FastAPI Backend - Analytics Dashboard

Production-ready FastAPI backend for the Analytics Dashboard.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload

# API available at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

## Project Structure

```
backend/
├── main.py              # FastAPI app entry point
├── api/                 # API routes
│   └── routes/
│       ├── health.py    # Health check endpoints
│       ├── upload.py   # CSV upload endpoints
│       ├── metrics.py   # KPI/metrics endpoints
│       ├── charts.py    # Chart data endpoints
│       └── chat.py      # AI chat endpoints
├── core/                # Core modules
│   ├── config.py        # Configuration (Pydantic Settings)
│   ├── database.py      # Database connection
│   └── ai_service.py    # AI/LLM service
├── models/              # Pydantic models
│   ├── requests.py     # Request models
│   └── responses.py    # Response models
├── services/            # Business logic layer
│   ├── data_service.py
│   ├── metrics_service.py
│   ├── chart_service.py
│   └── validation_service.py
└── tests/               # API tests
    └── test_api.py
```

## Environment Variables

Create `.env` file in project root:

```env
DATABASE_PATH=data/analytics.duckdb
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
FRONTEND_URL=http://localhost:3000
LOG_LEVEL=INFO
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Basic health check
- `GET /api/health/detailed` - Detailed health (DB + ChromaDB + Ollama)
- `GET /api/docs` - Interactive API documentation (Swagger)
- `GET /api/redoc` - Alternative API documentation (ReDoc)
- `POST /api/admin/backups/create` - Create backup (admin only)

## Development

```bash
# Run with auto-reload
uvicorn main:app --reload

# Run tests
pytest tests/

# Run from project root
cd backend
uvicorn main:app --reload
```

## Backup & Restore

See [Backup and Monitoring Guide](docs/BACKUP_AND_MONITORING.md) for complete documentation.

**Quick backup:**
```bash
python scripts/backup.py
```

**Restore (⚠️ stop server first):**
```bash
python scripts/restore.py backups/backup_YYYYMMDD_HHMMSS --confirm
```

## Monitoring & Logging

**Structured logging:**
- All requests include `request_id`, `user_id`, `tenant_id`
- Logs stored in `backend/logs/`
- Configurable via `LOG_LEVEL` env var

**Health checks:**
- Basic: `GET /api/health`
- Detailed: `GET /api/health/detailed` (includes DB, ChromaDB, Ollama status)

See [Backup and Monitoring Guide](docs/BACKUP_AND_MONITORING.md) for details.

## Docker

See main project `docker-compose.yml` for backend service configuration.

