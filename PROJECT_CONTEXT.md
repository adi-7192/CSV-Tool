# Project Context - Sales Analytics Dashboard

## Overview
Production-ready business analytics platform with AI-powered insights. Multi-tenant SaaS application for sales data analysis with natural language querying.

**Tech Stack:**
- **Backend**: FastAPI (Python), DuckDB (analytics DB), ChromaDB (vector store)
- **Frontend**: React 18 + TypeScript + Vite + Ant Design
- **AI**: Google Gemini API with RAG (Retrieval Augmented Generation)
- **Auth**: JWT-based authentication with password reset
- **State**: Zustand for frontend state management

## Architecture

### Backend Structure (`backend/`)
```
api/routes/          # API endpoints (auth, metrics, chat, upload, admin)
core/                # Config, database, AI service
services/            # Business logic (metrics, RAG, validation, upload)
models/              # Pydantic request/response models
utils/               # Utilities (encryption, validation, LLM clients)
```

### Frontend Structure (`frontend/src/`)
```
pages/               # Main views (Dashboard, Workspace, AIAnalyst, Admin)
components/          # Reusable UI components
services/            # API client services
store/               # Zustand state stores (data, auth, chat)
styles/              # Design tokens and themes
```

### Database
- **DuckDB** (`data/analytics.duckdb`): Main analytics database
  - `sales` table: Core transaction data
  - `users` table: User accounts with tenant isolation
  - `ingestion_log` table: File upload tracking
  - `system_events` table: Monitoring/observability logs
  - `api_keys` table: Encrypted API keys
- **ChromaDB** (`backend/data/chromadb/`): Vector embeddings for RAG

## Key Features

### 1. Multi-File CSV Upload
- Batch upload with progress tracking
- Automatic column mapping
- Data validation and deduplication
- Tenant isolation (users only see their data)

### 2. AI-Powered Analytics
- Natural language questions → SQL queries
- Intent classification (METRIC, ADVISORY, EXPLORATION, CONVERSATIONAL)
- RAG-enhanced context retrieval
- Safe SQL validation before execution

### 3. Business Metrics
- Revenue (gross, net, margin)
- Refund/cancellation rates
- Top products, cities, SKUs
- Period comparisons (MoM, YoY)
- Movers & decliners analysis

### 4. Authentication & Authorization
- JWT-based auth with refresh tokens
- Password reset via email
- Role-based access (user, admin)
- Tenant isolation at database level
- Session invalidation on password change

### 5. Admin Console
- User management (activate/deactivate)
- Tenant & data management
- System monitoring (errors, slow requests, health)
- Usage analytics
- Admin-only API endpoints

### 6. Monitoring & Observability
- Structured logging with request IDs
- System events table (errors, slow requests)
- Health checks (DB, Redis, app version)
- Admin monitoring dashboard

## Database Schema

### `sales` Table
```sql
order_date, sku, product_name, city, state,
gross_revenue, refund_amount, cancellation_amount,
transaction_type, source_file, ingestion_id,
tenant_id, loaded_at, updated_at
```

### `users` Table
```sql
id, email, password_hash, role, plan, onboarded,
tenant_id, is_active, last_login_at,
password_changed_at, token_version, created_at
```

### `system_events` Table
```sql
id (BIGINT, manual increment), created_at, level, category,
message, endpoint, method, status_code, duration_ms,
tenant_id, user_id, request_id, meta (JSON)
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - Login (returns JWT)
- `POST /api/auth/refresh` - Refresh access token
- `POST /api/auth/forgot-password` - Request password reset
- `POST /api/auth/reset-password` - Reset password with token
- `POST /api/users/change-password` - Change password (requires current password)

### Data Management
- `POST /api/data/upload` - Single CSV upload
- `POST /api/data/upload/multiple` - Multiple CSV upload (batch)
- `GET /api/data/export?format=csv|parquet` - Export data
- `GET /api/data/summary` - Data availability summary
- `GET /api/data/statistics` - Data statistics
- `GET /api/data/transactions` - Paginated transactions
- `DELETE /api/data/ingestion/{ingestion_id}` - Delete uploaded file

### Analytics
- `GET /api/metrics` - Business KPIs
- `GET /api/charts/revenue-trend` - Revenue trends
- `GET /api/charts/product-performance` - Product metrics
- `POST /api/chat/ask` - AI chat question

### Admin (Admin-only)
- `GET /api/admin/monitoring/events` - System events
- `GET /api/admin/monitoring/summary` - Monitoring summary
- `GET /api/admin/monitoring/health` - Health status
- `GET /api/admin/stats/summary` - System statistics
- `GET /api/admin/tenants/usage` - Tenant usage data
- `POST /api/admin/users/{user_id}/activate` - Activate user
- `POST /api/admin/users/{user_id}/deactivate` - Deactivate user
- `DELETE /api/admin/tenants/{tenant_id}` - Delete tenant

## Key Services

### Backend Services
- **`metrics_service.py`**: KPI calculations (revenue, margins, rates)
- **`rag_service.py`**: RAG context retrieval and prompt enhancement
- **`upload_service.py`**: CSV processing, validation, ingestion
- **`auth_service.py`**: Authentication, password hashing, JWT generation
- **`monitoring_service.py`**: System event logging
- **`email_service.py`**: SMTP email sending (password reset)
- **`api_key_service.py`**: Encrypted API key storage
- **`intent_classifier.py`**: Classifies user questions
- **`sql_validator.py`**: Validates generated SQL queries

### Frontend Services
- **`api.ts`**: Axios client with interceptors, API service definitions
- **`dataService.ts`**: Data fetching (transactions, metrics, upload)
- **`authService.ts`**: Authentication API calls
- **`adminService.ts`**: Admin API calls

### State Stores (Zustand)
- **`dataStore.ts`**: Global data state (metrics, date range, loading)
- **`authStore.ts`**: Authentication state (user, token, login/logout)
- **`chatStore.ts`**: Chat conversations (persisted to localStorage)

## Important Implementation Details

### Tenant Isolation
- Every query filters by `tenant_id` (from JWT)
- Users can only access their own data
- Admin can view all tenants
- Database-level isolation enforced

### File Upload Flow
1. User uploads CSV(s) → `upload_multiple_csv_endpoint`
2. Column mapping (if needed) → `ColumnMappingModal`
3. Validation & cleaning → `upload_service.py`
4. Insert to DuckDB → `sales` table with `tenant_id`
5. Success modal → `UploadSuccessModal` with animation
6. Auto-detect date range → Update global date range

### AI Chat Flow
1. User question → `POST /api/chat/ask`
2. Intent classification → `intent_classifier.py`
3. If METRIC → Check `metrics_registry.py` for function match
4. If no match → RAG context retrieval → SQL generation
5. SQL validation → Execute → Format response
6. Return natural language answer

### Monitoring Middleware
- Generates `request_id` (UUID) per request
- Measures request duration
- Logs to `system_events` table
- Captures exceptions with stack traces
- Sanitizes sensitive data (no passwords, tokens)

### Password Security
- Bcrypt hashing (cost factor 12)
- Password reset tokens (15min expiry)
- Rate limiting (5 per IP, 3 per email per 15min)
- Session invalidation on password change (`token_version`)

### Date Range Management
- Global date range in `dataStore`
- Auto-detected from uploaded data
- Future dates disabled in date pickers
- Placeholder text when no dates selected

## Configuration

### Environment Variables (`.env`)
```env
# Database
DATABASE_PATH=data/analytics.duckdb

# JWT
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=86400

# Email (SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=your-email@gmail.com

# Monitoring
MONITORING_ENABLED=True
MONITORING_SLOW_MS=1000
MONITORING_RETENTION_DAYS=7

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REQUIRE_REDIS_RATE_LIMITING=False

# Environment
ENV=development  # or "production"
```

## Key Files Reference

### Backend
- `backend/main.py` - FastAPI app entry, middleware setup
- `backend/core/database.py` - DuckDB connection, table initialization
- `backend/core/config.py` - Pydantic settings
- `backend/utils/monitoring_middleware.py` - Request logging middleware
- `backend/api/routes/data_routes.py` - Data upload/export endpoints
- `backend/api/routes/admin_monitoring.py` - Admin monitoring endpoints

### Frontend
- `frontend/src/App.tsx` - Main app, routing
- `frontend/src/pages/Workspace.tsx` - File upload page
- `frontend/src/pages/Dashboard.tsx` - Analytics dashboard
- `frontend/src/pages/AIAnalyst.tsx` - AI chat interface
- `frontend/src/components/UploadSuccessModal.tsx` - Upload success animation
- `frontend/src/components/BatchUploadProgress.tsx` - Batch upload progress

## Development

### Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```

### Run Tests
```bash
pytest backend/tests/ -v
```

## Security Notes
- API keys encrypted with AES-256
- Passwords hashed with bcrypt
- JWT tokens for authentication
- CORS configured for frontend
- SQL injection prevention (parameterized queries)
- Input validation on all endpoints
- Rate limiting on sensitive endpoints

## Data Flow Examples

### Upload Multiple Files
1. User selects files → `Workspace.tsx`
2. `handleBatchUpload()` → Process sequentially
3. Each file → `processBatchFile()` → `uploadCSV()`
4. Backend → `upload_multiple_csv_endpoint` → `process_csv_upload()`
5. Column mapping if needed → Pause batch → Resume after mapping
6. Success → Update `batchFiles` state → Show `UploadSuccessModal`
7. Auto-detect date range → Update `dataStore`

### AI Question
1. User types question → `AIAnalyst.tsx`
2. `POST /api/chat/ask` → `chat.py`
3. `intent_classifier.classify()` → Intent type
4. If METRIC → `metrics_registry.get_function()` → Execute function
5. If no match → `rag_service.get_context()` → `ai_service.generate_sql()`
6. `sql_validator.validate()` → Execute → `response_generator.format()`
7. Return answer → Display in chat

---

**Last Updated**: December 2024  
**Status**: Production-ready with multi-tenant support, AI analytics, and comprehensive monitoring

