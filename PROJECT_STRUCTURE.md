# 📊 Project Structure & Implementation Guide

**Amazon Sales Analytics Platform** - AI-powered business intelligence dashboard with real-time insights.

---

## 🏗️ Project Overview

- **Backend**: FastAPI (Python) with DuckDB analytics database
- **Frontend**: React + TypeScript + Vite + Ant Design
- **AI**: Gemini API integration with RAG (Retrieval Augmented Generation)
- **Database**: DuckDB (in-process SQL OLAP) + ChromaDB (vector store for RAG)

---

## 📁 Folder Structure

```
Nisarg Project/
│
├── backend/                    # FastAPI Backend
│   ├── api/routes/            # API Endpoints
│   ├── core/                  # Core Services
│   ├── services/              # Business Logic
│   ├── models/                # Pydantic Models
│   ├── utils/                 # Utilities
│   ├── data/                  # Database & ChromaDB
│   └── main.py                # FastAPI App Entry
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── pages/             # Page Components
│   │   ├── components/        # Reusable Components
│   │   ├── services/          # API Services
│   │   ├── store/             # Zustand State Management
│   │   └── styles/            # Styling & Themes
│   └── package.json
│
├── data/                       # DuckDB Database Files
├── docs/                       # Documentation
└── tests/                      # Test Files
```

---

## 🔧 Backend Structure

### **API Routes** (`backend/api/routes/`)

| File | Purpose |
|------|---------|
| `chat.py` | AI Analyst chat endpoint - handles natural language questions |
| `metrics.py` | Business metrics (revenue, refunds, orders, etc.) |
| `charts.py` | Chart data endpoints (trends, comparisons) |
| `upload.py` | CSV/Excel file upload and processing |
| `user_api_keys.py` | Gemini API key management |
| `data_routes.py` | Data management endpoints |
| `health.py` | Health check endpoint |

### **Core Services** (`backend/core/`)

- **`database.py`**: DuckDB connection and query execution
- **`ai_service.py`**: SQL generation from natural language using LLMs
- **`config.py`**: Application configuration

### **Business Logic** (`backend/services/`)

| Service | Functionality |
|---------|--------------|
| `metrics_service.py` | KPI calculations (revenue, refund rate, margins, etc.) |
| `metrics_registry.py` | Maps natural language questions to backend functions |
| `intent_classifier.py` | Classifies user questions (METRIC, ADVISORY, EXPLORATION, CONVERSATIONAL) |
| `metric_queries.py` | Safe SQL templates for common metric queries |
| `response_generator.py` | Generates natural language responses from query results |
| `rag_service.py` | Retrieval Augmented Generation - enhances prompts with context |
| `embedding_service.py` | ChromaDB initialization and document embedding |
| `upload_service.py` | CSV/Excel processing and data ingestion |
| `api_key_service.py` | Encrypted API key storage and retrieval |
| `sql_validator.py` | Validates generated SQL against schema |

### **Utilities** (`backend/utils/`)

- **`external_llm.py`**: Gemini API integration for SQL generation
- **`api_key_validator.py`**: Validates API keys against provider APIs
- **`encryption.py`**: Encrypts/decrypts API keys in database

### **Data Storage**

- **`data/analytics.duckdb`**: Main analytics database (DuckDB)
- **`data/chromadb/`**: Vector database for RAG embeddings

---

## 🎨 Frontend Structure

### **Pages** (`frontend/src/pages/`)

| Page | Description |
|------|-------------|
| `Dashboard.tsx` | Main analytics dashboard with metrics, charts, and insights |
| `AIAnalyst.tsx` | AI chat interface for natural language queries |
| `DataManagement.tsx` | File upload and data management |
| `Settings.tsx` | API key configuration |
| `Workspace.tsx` | Data workspace view |

### **Components** (`frontend/src/components/`)

- **`common/`**: Reusable components (MetricCard, ChartCard, EmptyState, etc.)
- **`TopBar.tsx`**: Navigation bar with date range picker
- **`TrendChart.tsx`**: Revenue trend visualization
- **`PerformanceTable.tsx`**: Product performance table
- **`UploadWizard.tsx`**: Multi-step file upload wizard

### **State Management** (`frontend/src/store/`)

- **`dataStore.ts`**: Global data state (metrics, charts, date range)
- **`chatStore.ts`**: Chat conversations and messages (persisted to localStorage)
- **`uiStore.ts`**: UI state management

### **Services** (`frontend/src/services/`)

- **`api.ts`**: Axios client and API service definitions
- **`apiKeyService.ts`**: API key CRUD operations
- **`dataService.ts`**: Data fetching utilities

---

## 🔄 Key Features & Data Flow

### **1. Data Ingestion**
```
CSV/Excel Upload → upload_service.py → 
Column Mapping → Data Transformation → 
DuckDB Storage (sales table)
```

### **2. AI Chat Flow**
```
User Question → Intent Classification → 
Metrics Registry Check → 
  ├─ Match Found → Call Backend Function → Format Response
  └─ No Match → RAG Context Retrieval → 
      SQL Generation → Validation → 
      Query Execution → Response Generation
```

### **3. RAG System**
```
Schema + Business Rules + Sample Data → 
ChromaDB Embedding → 
Semantic Search on User Question → 
Enhanced Prompt → LLM SQL Generation
```

### **4. Metrics Calculation**
```
Date Range Selection → 
Backend Metrics Service → 
DuckDB Queries → 
Aggregated KPIs → 
Frontend Display
```

---

## 🗄️ Database Schema

### **Main Tables**

- **`sales`**: Core sales data with columns:
  - `order_date`, `sku`, `product_name`, `city`, `state`
  - `gross_revenue`, `refund_amount`, `cancellation_amount`
  - `transaction_type`, `source_file`, `ingestion_id`
  - `loaded_at`, `updated_at`

### **Metadata Tables**

- **`api_keys`**: Encrypted API keys (Gemini, OpenAI, Anthropic)
- **`column_registry`**: Column mapping metadata

---

## 🔐 Security Features

- **API Key Encryption**: AES-256 encryption for stored API keys
- **Input Validation**: Question validation to prevent invalid queries
- **SQL Safety**: Validates SQL queries before execution
- **CORS**: Configured for local development

---

## 📊 Key Metrics & Calculations

### **Available Metrics**

- Gross Revenue, Net Revenue, Net Margin
- Refund Rate, Cancellation Rate
- Order Count, Units Sold
- Average Order Value
- Success Rate
- Top Products, Top Cities
- Movers & Decliners (growth trends)
- Period Comparisons

### **Backend Functions** (`metrics_service.py`)

- `calculate_metrics()`: Overall KPIs
- `get_top_products()`: Top N products by revenue
- `get_revenue_by_city()`: Regional performance
- `get_movers_decliners()`: Growth/decline trends
- `compare_periods()`: Period-over-period comparison

---

## 🤖 AI Features

### **Intent Classification**
- **METRIC**: Quantitative questions → Safe SQL templates
- **ADVISORY**: Business advice → Comprehensive data + LLM analysis
- **EXPLORATION**: Data exploration → RAG-enhanced SQL generation
- **CONVERSATIONAL**: General chat → Predefined responses

### **Question Validation**
- Detects invalid/nonsensical inputs
- Validates repeated characters, keyboard patterns
- Requires business keywords or question words

### **RAG Context**
- Schema documentation
- Business rules
- Sample data patterns
- Historical insights

---

## 🚀 API Endpoints

### **Core Endpoints**

- `POST /api/chat/ask` - AI chat question
- `GET /api/metrics` - Business metrics
- `GET /api/charts/*` - Chart data
- `POST /api/upload` - File upload
- `GET /api/user-api-keys` - API key management
- `GET /api/health` - Health check

---

## 📦 Dependencies

### **Backend**
- FastAPI, Pydantic, DuckDB
- ChromaDB, sentence-transformers (RAG)
- httpx (Gemini API), cryptography (encryption)

### **Frontend**
- React, TypeScript, Vite
- Ant Design (UI components)
- Recharts (charts)
- Zustand (state management)
- Axios (HTTP client)

---

## 🔄 State Management

### **Frontend Stores**

- **`dataStore`**: Metrics, charts, date range, loading states
- **`chatStore`**: Conversations, messages (persisted)
- **`uiStore`**: UI preferences

### **Persistence**

- Chat history: `localStorage` (via zustand-persist)
- Date range: Global store (session-based)

---

## 📝 Key Implementation Details

### **Message Limiting**
- Chat displays last 100 messages to prevent performance issues
- Scroll tracking prevents auto-scroll when user scrolls up

### **Error Handling**
- Individual message rendering wrapped in try-catch
- Graceful fallbacks for API failures
- User-friendly error messages

### **Date Range Context**
- Global date range from `dataStore`
- Passed to all API calls
- Used for filtering queries

---

## 🎯 Development Workflow

1. **Backend**: `cd backend && uvicorn main:app --reload`
2. **Frontend**: `cd frontend && npm run dev`
3. **Access**: `http://localhost:5174` (frontend) → `http://localhost:8000` (backend)

---

## 📚 Documentation Files

- `README.md` - Project overview
- `TECHNICAL_ARCHITECTURE_DOCUMENTATION.md` - Architecture details
- `docs/` - Component documentation
- `CLEANUP_REPORT.md` - Cleanup sprint report

---

**Last Updated**: December 2025  
**Status**: Production-ready with AI-powered analytics

