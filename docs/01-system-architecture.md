# System Architecture Diagram

## Overview

The Analytics Dashboard is a full-stack application with a React frontend, FastAPI backend, DuckDB database, and AI-powered insights via Ollama.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  React Frontend  │         │  API Consumers   │             │
│  │  (Vite + TS)     │         │  (External)      │             │
│  │  Port: 3000/5173 │         │                  │             │
│  └────────┬─────────┘         └────────┬─────────┘             │
│           │                             │                        │
│           └─────────────┬───────────────┘                        │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          │ HTTP/REST API
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    API GATEWAY LAYER                          │
├─────────────────────────┼────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           FastAPI Application (main.py)              │   │
│  │  - CORS Middleware                                   │   │
│  │  - Request Validation                                │   │
│  │  - Error Handling                                    │   │
│  │  - Logging                                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    ROUTING LAYER                             │
├─────────────────────────┼────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Health       │  │ Metrics      │  │ Upload       │      │
│  │ /api/health  │  │ /api/metrics │  │ /api/upload  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Charts       │  │ Chat (AI)    │  │ Data         │      │
│  │ /api/charts  │  │ /api/chat    │  │ /api/data    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    SERVICE LAYER                              │
├─────────────────────────┼────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Metrics Service  │  │ Upload Service   │                 │
│  │ - KPI Calc       │  │ - CSV Processing │                 │
│  │ - Trends         │  │ - Validation     │                 │
│  │ - Analytics      │  │ - Deduplication  │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Chart Service    │  │ Data Service     │                 │
│  │ - Visualizations │  │ - Transactions   │                 │
│  │ - Aggregations   │  │ - Filtering      │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Validation      │  │ AI Service       │                 │
│  │ Service         │  │ - SQL Generation │                 │
│  │ - Data Quality  │  │ - Query Exec     │                 │
│  └──────────────────┘  └──────────────────┘                 │
│                                                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    UTILITY LAYER                              │
├─────────────────────────┼────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Validators   │  │ Sanitizers   │  │ Logger       │      │
│  │ - Date Range │  │ - Input Clean │  │ - Multi-level│      │
│  │ - SKU Format │  │ - Type Conv   │  │ - File/Console│    │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐                                           │
│  │ Error Handler │                                           │
│  │ - Format      │                                           │
│  │ - Logging     │                                           │
│  └──────────────┘                                           │
│                                                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          │
┌─────────────────────────┼────────────────────────────────────┐
│                    DATA LAYER                                 │
├─────────────────────────┼────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Database Module (core/database.py)      │   │
│  │  - Connection Management (Singleton)                │   │
│  │  - Query Execution                                   │   │
│  │  - Schema Management                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DuckDB Database                         │   │
│  │  Location: data/analytics.duckdb                     │   │
│  │                                                       │   │
│  │  Tables:                                             │   │
│  │  - sales (main transaction data)                      │   │
│  │  - ingestion_log (upload history)                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│                    EXTERNAL SERVICES                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Ollama LLM Service                       │   │
│  │  URL: http://localhost:11434                        │   │
│  │  Model: llama3.1:8b                                 │   │
│  │  Purpose: Natural language to SQL conversion        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Component Layers

### 1. Client Layer
- **React Frontend** (TypeScript + Vite)
  - Dashboard UI
  - Data visualization (Recharts)
  - State management (Zustand)
  - API communication (Axios)

### 2. API Gateway Layer
- **FastAPI Application**
  - CORS middleware for frontend access
  - Request/response validation
  - Centralized error handling
  - Request logging

### 3. Routing Layer
- **API Routes** (8 route modules)
  - Health checks
  - Metrics endpoints
  - Upload endpoints
  - Chart data endpoints
  - AI chat endpoints
  - Data query endpoints

### 4. Service Layer
- **Business Logic Services**
  - Metrics calculation
  - Data processing
  - Validation
  - AI integration

### 5. Utility Layer
- **Cross-cutting Concerns**
  - Input validation
  - Input sanitization
  - Logging (multi-level)
  - Error formatting

### 6. Data Layer
- **Database Access**
  - DuckDB connection (singleton)
  - Query execution
  - Schema management

### 7. External Services
- **Ollama LLM**
  - Natural language processing
  - SQL query generation

## Data Flow

### Upload Flow
```
CSV File → Upload Endpoint → Upload Service
    → Validation → Transformation → Database
    → Ingestion Log → Response
```

### Query Flow
```
API Request → Route → Service Layer
    → Database Query → DuckDB
    → Result Processing → Response
```

### AI Query Flow
```
Natural Language → Chat Endpoint → AI Service
    → Ollama (SQL Generation) → Validation
    → Database Query → Format Response
```

## Technology Stack

### Frontend
- **React 18.2** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Ant Design** - UI components
- **Recharts** - Data visualization
- **Zustand** - State management
- **Axios** - HTTP client

### Backend
- **FastAPI 0.115.0** - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **DuckDB 0.10.3** - Analytical database
- **Pandas 2.1.4** - Data manipulation
- **LangChain** - AI/LLM integration

### Infrastructure
- **Docker** - Containerization
- **Docker Compose** - Orchestration
- **Ollama** - LLM runtime

## Deployment Architecture

### Development
```
Frontend (Vite Dev Server) → Backend (Uvicorn) → DuckDB (Local File)
```

### Production (Docker)
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Frontend      │    │  Backend        │    │  Ollama         │
│  Container     │───▶│  Container      │───▶│  Container      │
│  (Nginx)       │    │  (FastAPI)      │    │  (LLM Service)  │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                                ▼
                         ┌─────────────────┐
                         │  DuckDB         │
                         │  (Volume Mount) │
                         └─────────────────┘
```

## Security Architecture

### Input Validation
- **Validators** - Type and format validation
- **Sanitizers** - Input cleaning and normalization
- **SQL Injection Prevention** - Parameterized queries

### Error Handling
- **Centralized Error Handler** - Consistent error responses
- **Error Logging** - Detailed error context
- **User-Friendly Messages** - No sensitive data exposure

### Logging
- **Multi-level Logging** - DEBUG, INFO, WARNING, ERROR
- **Separate Log Files** - app.log, api.log, errors.log, database.log
- **Structured Logging** - Context and metadata

## Scalability Considerations

### Current Limitations
- **DuckDB** - Single connection (singleton pattern)
- **No Caching** - All queries hit database
- **Synchronous Processing** - No async database operations

### Future Enhancements
- Connection pooling for DuckDB
- Redis caching layer
- Async database operations
- Background job processing for large uploads





