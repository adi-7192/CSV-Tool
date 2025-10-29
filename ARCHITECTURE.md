# 📊 CSV Analytics Dashboard - Architecture & Components Documentation

## 📁 Project Folder Structure

```
Nisarg Project/
│
├── 📄 Core Application Files
│   ├── app.py                      # Main Streamlit application (4,140 lines)
│   ├── ai_assistant.py             # AI Assistant module (3,168 lines)
│   ├── db_manager.py               # Database management module
│   ├── diagnose.py                 # Diagnostic tool for troubleshooting
│   │
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                 # Git ignore rules
│   │
│   └── 📚 Documentation
│       ├── README.md              # Quick start guide
│       ├── FEATURE_DOCUMENTATION.md  # Detailed feature docs
│       └── ARCHITECTURE.md         # This file
│
├── 💾 Data Storage
│   └── data/
│       ├── analytics.duckdb       # DuckDB database (main storage)
│       ├── mapping.json           # Persistent column mappings
│       │
│       ├── raw/                   # Original uploaded CSV files
│       │   ├── JulyMonthly_raw.csv
│       │   ├── Augmonthly_raw_*.csv
│       │   └── SeptMonthly_raw_*.csv
│       │
│       └── cleaned/               # Processed CSV files
│           ├── JulyMonthly_cleaned_*.csv
│           ├── Augmonthly_cleaned_*.csv
│           └── SeptMonthly_cleaned_*.csv
│
├── 🗂️ Generated Files (gitignored)
│   ├── __pycache__/               # Python bytecode cache
│   ├── ai_assistant_errors.log    # Error logs
│   └── ai_query_log.csv           # AI query performance logs
│
└── 🐍 Virtual Environment
    └── venv/                      # Python virtual environment
```

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                      (Streamlit Web App)                         │
│                          (app.py)                               │
└────────────┬───────────────────────────────────────────────────┘
             │
             │ 1. File Upload & Processing
             │ 2. User Queries
             │ 3. Display Results
             │
     ┌───────┴───────────┬──────────────────────────┬──────────┐
     │                   │                          │          │
     ▼                   ▼                          ▼          ▼
┌─────────┐     ┌──────────────┐          ┌──────────────┐ ┌──────┐
│   Data   │     │    AI        │          │   Database   │ │Files │
│ Cleaning │     │  Assistant   │          │   Manager    │ │(CSV) │
│  Logic   │     │              │          │              │ │      │
│          │     │ (ai_         │          │ (db_manager) │ │      │
│          │     │ assistant.py)│          │              │ │      │
└─────┬────┘     └──────┬───────┘          └──────┬───────┘ └──┬───┘
      │                 │                          │           │
      │                 │                          │           │
      └─────────────────┴──────────────────────────┴───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   DuckDB Database    │
                    │ (analytics.duckdb)   │
                    │                      │
                    │  - sales table       │
                    │  - Indexed columns   │
                    │  - Persistent storage│
                    └──────────────────────┘
```

---

## 🧩 Core Components & Their Connections

### 1. **app.py** - Main Application (4,140 lines)

**Purpose:** Main Streamlit web application serving as the user interface layer.

**Key Responsibilities:**
- 🖥️ **UI Rendering**: Streamlit interface with sidebar and tabs
- 📤 **File Upload Management**: Handles CSV file uploads and validation
- 🔄 **Column Mapping**: Automatic and manual column mapping
- 📊 **Dashboard Display**: KPIs, charts, and data visualization
- 💬 **Chat Interface**: Integration with AI Assistant
- 🧹 **Data Cleaning**: Transaction-aware data processing
- 📥 **Data Export**: CSV download functionality

**Connections:**
```
app.py
  │
  ├─→ db_manager.py
  │    ├─→ store_data()        # Store cleaned data
  │    ├─→ query_data()         # Query database
  │    ├─→ get_row_count()      # Get statistics
  │    ├─→ table_exists()       # Check table existence
  │    └─→ clear_database()     # Reset database
  │
  ├─→ ai_assistant.py
  │    ├─→ AIAssistant class    # Main AI class
  │    ├─→ log_ai_query()        # Log queries
  │    ├─→ get_query_history()   # Get query logs
  │    └─→ calculate_performance_stats()  # Performance metrics
  │
  ├─→ data/mapping.json         # Persistent column mappings
  ├─→ data/raw/                 # Store original CSV files
  └─→ data/cleaned/             # Store processed CSV files
```

**Main Functions:**
- `normalize_header()` - Column header normalization
- `normalize_key_fields()` - Normalize order_id, sku, order_date
- `clean_dataframe()` - Transaction-aware data cleaning
- `get_suggested_questions()` - AI chat suggestions
- `main()` - Streamlit app entry point

---

### 2. **ai_assistant.py** - AI Query Engine (3,168 lines)

**Purpose:** Natural language query processing with intelligent caching and SQL generation.

**Key Responsibilities:**
- 🤖 **LLM Integration**: Ollama (llama3.1:8b) for SQL generation
- 💾 **Intelligent Caching**: LRU cache for questions, SQL, and analysis
- 📝 **Query Logging**: Performance tracking and analytics
- 🔍 **SQL Generation**: Natural language to SQL conversion
- ✅ **SQL Safety**: Validation and sanitization
- 📊 **Response Formatting**: Business-friendly responses
- 🔮 **Predictive Analytics**: Trend analysis and forecasting

**Connections:**
```
ai_assistant.py
  │
  ├─→ db_manager.py
  │    ├─→ query_data()         # Execute SQL queries
  │    ├─→ get_row_count()       # Get row counts
  │    └─→ table_exists()        # Check table existence
  │
  ├─→ Ollama API (http://localhost:11434)
  │    ├─→ /api/generate        # SQL generation
  │    └─→ /api/tags            # Model verification
  │
  ├─→ ai_query_log.csv          # Performance logging
  └─→ ai_assistant_errors.log   # Error logging
```

**Main Classes:**
- **`AIAssistant`** - Main AI assistant class
  - `ask_question()` - Process natural language questions
  - `_generate_intelligent_sql()` - Generate SQL from questions
  - `_handle_intelligent_sql_question_with_timing()` - Process with timing
  - `clear_all_caches()` - Clear all caches
  - `get_cache_stats()` - Get performance statistics

- **`PredictiveAnalytics`** - Predictive analysis features
  - `analyze_trends()` - Trend analysis
  - `forecast_revenue()` - Revenue forecasting
  - `detect_anomalies()` - Anomaly detection
  - `get_seasonal_patterns()` - Seasonal pattern analysis

**Caching System:**
- `question_cache` - LRU cache for complete Q&A pairs
- `sql_cache` - LRU cache for generated SQL queries
- `analysis_cache` - LRU cache for LLM analysis results
- `schema_cache` - Database schema cache

---

### 3. **db_manager.py** - Database Layer

**Purpose:** DuckDB database operations and connection management.

**Key Responsibilities:**
- 🗄️ **Connection Management**: Cached DuckDB connections
- 💾 **Data Storage**: Store and append DataFrame to database
- 📊 **Data Querying**: Execute SQL queries and return results
- 🔍 **Table Management**: Create tables, check existence
- 📈 **Indexing**: Automatic index creation for performance

**Connections:**
```
db_manager.py
  │
  ├─→ data/analytics.duckdb     # DuckDB database file
  │
  └─→ Streamlit Cache
       └─→ get_connection()      # Cached connection via @st.cache_resource
```

**Main Functions:**
- `get_connection()` - Get cached DuckDB connection
- `store_data()` - Store DataFrame in database
- `query_data()` - Execute SQL query and return DataFrame
- `clear_database()` - Clear all tables
- `get_row_count()` - Get row count for table
- `table_exists()` - Check if table exists

---

### 4. **diagnose.py** - Diagnostic Tool

**Purpose:** Troubleshooting and diagnostic utilities.

**Key Responsibilities:**
- 🔍 **Database Checks**: Verify database integrity
- 📊 **Data Validation**: Check data quality
- 🐛 **Error Detection**: Identify common issues

---

## 🔄 Data Flow Architecture

### **Upload & Processing Flow**

```
User Uploads CSV
      │
      ▼
┌─────────────────┐
│  app.py         │
│  File Handler   │
└─────┬───────────┘
      │
      ├─→ Save to data/raw/
      │
      ▼
┌─────────────────┐
│  Column Mapping │
│  (Auto/Manual)   │
└─────┬───────────┘
      │
      ├─→ Save mapping to data/mapping.json
      │
      ▼
┌─────────────────┐
│  Data Cleaning  │
│  clean_dataframe│
└─────┬───────────┘
      │
      ├─→ Transaction type handling
      ├─→ Duplicate detection
      ├─→ Revenue calculation
      │
      ├─→ Save to data/cleaned/
      │
      ▼
┌─────────────────┐
│ db_manager.py   │
│ store_data()    │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  DuckDB         │
│  analytics.duckdb│
│  sales table    │
└─────────────────┘
```

### **AI Query Flow**

```
User Asks Question
      │
      ▼
┌─────────────────┐
│  app.py         │
│  Chat UI        │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  ai_assistant.py│
│  AIAssistant    │
│  ask_question() │
└─────┬───────────┘
      │
      ├─→ Check question_cache (LRU)
      │   └─→ If found: Return cached result
      │
      ├─→ Check similar_cached_question()
      │   └─→ If similarity > 0.85: Return similar result
      │
      ▼
┌─────────────────┐
│  Generate SQL   │
│  _generate_     │
│  intelligent_sql│
└─────┬───────────┘
      │
      ├─→ Check sql_cache (LRU)
      │   └─→ If found: Use cached SQL
      │
      ├─→ Create prompt with schema
      │
      ├─→ Call Ollama API
      │   └─→ llama3.1:8b model
      │
      ├─→ Extract SQL from response
      │
      ├─→ Cache generated SQL
      │
      ▼
┌─────────────────┐
│  SQL Safety     │
│  Validation    │
│  _validate_sql │
│  _safety()     │
└─────┬───────────┘
      │
      ├─→ Check for dangerous operations
      ├─→ Validate column names
      ├─→ Ensure safe queries only
      │
      ▼
┌─────────────────┐
│  Query          │
│  Optimization   │
│  _optimize_     │
│  query()        │
└─────┬───────────┘
      │
      ├─→ Add LIMIT clauses
      ├─→ Optimize ORDER BY
      │
      ▼
┌─────────────────┐
│ db_manager.py   │
│ query_data()    │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  DuckDB         │
│  Execute SQL    │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Format         │
│  Response       │
│  _format_       │
│  response_      │
│  professionally │
└─────┬───────────┘
      │
      ├─→ Check analysis_cache (LRU)
      │   └─→ If found: Use cached analysis
      │
      ├─→ Format with Indian currency
      ├─→ Add business insights
      │
      ├─→ Cache complete result
      │
      ├─→ Log query to ai_query_log.csv
      │
      ▼
┌─────────────────┐
│  Return to User │
│  Display Result │
└─────────────────┘
```

---

## 🔌 Component Interactions

### **1. File Upload → Data Storage**

```
app.py (upload handler)
  │
  ├─→ Save original → data/raw/
  │
  ├─→ Column mapping → data/mapping.json (persist)
  │
  ├─→ Clean data → clean_dataframe()
  │   ├─→ Normalize columns
  │   ├─→ Handle transactions
  │   ├─→ Detect duplicates
  │   └─→ Calculate metrics
  │
  ├─→ Save cleaned → data/cleaned/
  │
  └─→ db_manager.store_data()
      └─→ DuckDB storage
          └─→ Index creation
```

### **2. Query Processing → Response**

```
User Query
  │
  ├─→ app.py (chat UI)
  │
  ├─→ ai_assistant.py (AIAssistant)
  │   ├─→ Cache lookup
  │   ├─→ SQL generation (Ollama)
  │   ├─→ SQL validation
  │   ├─→ SQL optimization
  │   │
  │   └─→ db_manager.query_data()
  │       └─→ DuckDB query
  │           └─→ Return DataFrame
  │
  ├─→ Format response
  │   ├─→ Currency formatting
  │   ├─→ Business insights
  │   └─→ Cache result
  │
  └─→ Display to user
```

### **3. Dashboard → Analytics**

```
app.py (dashboard tab)
  │
  ├─→ Date filter selection
  │
  ├─→ db_manager.query_data()
  │   ├─→ Filter by date range
  │   ├─→ Calculate KPIs
  │   └─→ Aggregate data
  │
  ├─→ Generate visualizations
  │   ├─→ Plotly charts
  │   └─→ Interactive graphs
  │
  └─→ Display dashboard
```

---

## 📊 Feature Breakdown

### **Core Features**

#### **1. Multi-File Upload System**
- **Location:** `app.py` - File upload handlers
- **Dependencies:** 
  - `data/raw/` - Storage
  - `data/mapping.json` - Column mappings
- **Features:**
  - Drag & drop interface
  - Multiple file support
  - File validation
  - Automatic cleanup

#### **2. Intelligent Column Mapping**
- **Location:** `app.py` - `SYNONYMS` dictionary + mapping logic
- **Dependencies:**
  - `data/mapping.json` - Persistent mappings
- **Features:**
  - Auto-detection using synonyms
  - Manual override capability
  - Type inference
  - Persistent storage

#### **3. Transaction-Aware Data Cleaning**
- **Location:** `app.py` - `clean_dataframe()` function
- **Dependencies:**
  - `normalize_key_fields()` - Normalization
  - `data/cleaned/` - Cleaned file storage
- **Features:**
  - Duplicate detection
  - Transaction logic (Shipment/Refund/FreeReplacement)
  - Revenue calculation
  - Data validation

#### **4. Comprehensive Dashboard**
- **Location:** `app.py` - Dashboard tab
- **Dependencies:**
  - `db_manager.query_data()` - Data retrieval
  - Plotly - Visualizations
- **Features:**
  - KPI metrics
  - Date filtering
  - Interactive charts
  - Trend analysis
  - Month breakdown

#### **5. AI-Powered Chat**
- **Location:** `app.py` (UI) + `ai_assistant.py` (Logic)
- **Dependencies:**
  - `AIAssistant` class
  - Ollama API
  - `db_manager.query_data()`
- **Features:**
  - Natural language queries
  - SQL generation
  - Intelligent caching
  - Query logging
  - Performance tracking

#### **6. Database Management**
- **Location:** `db_manager.py`
- **Dependencies:**
  - DuckDB
  - `data/analytics.duckdb`
- **Features:**
  - Connection pooling
  - Data storage
  - Query execution
  - Index management

#### **7. Intelligent Caching System**
- **Location:** `ai_assistant.py` - LRU cache implementation
- **Dependencies:**
  - `OrderedDict` from collections
- **Features:**
  - Question caching
  - SQL caching
  - Analysis caching
  - Similarity matching
  - Cache statistics

#### **8. Query Logging & Analytics**
- **Location:** `ai_assistant.py` - Logging functions
- **Dependencies:**
  - `ai_query_log.csv`
- **Features:**
  - Query tracking
  - Performance metrics
  - Success rate monitoring
  - Response time analysis

#### **9. Predictive Analytics**
- **Location:** `ai_assistant.py` - `PredictiveAnalytics` class
- **Dependencies:**
  - `db_manager.query_data()` - Data retrieval
- **Features:**
  - Trend analysis
  - Revenue forecasting
  - Anomaly detection
  - Seasonal patterns

---

## 🔐 Data Persistence

### **Persistent Files**

1. **`data/analytics.duckdb`**
   - Main database file
   - Contains `sales` table
   - Indexed columns for performance
   - Managed by `db_manager.py`

2. **`data/mapping.json`**
   - Column mapping configurations
   - Used for auto-mapping similar files
   - Updated when mappings change

3. **`data/raw/` & `data/cleaned/`**
   - Original and processed CSV files
   - Organized by upload timestamp
   - Used for backup and reference

4. **`ai_query_log.csv`**
   - Query performance logs
   - Tracks all AI queries
   - Used for analytics

---

## 🔧 Configuration & Dependencies

### **Python Dependencies** (`requirements.txt`)

```
streamlit          # Web framework
duckdb             # Database
pandas             # Data manipulation
plotly             # Visualizations
requests           # HTTP client (Ollama API)
langchain          # AI framework
langchain-community # LangChain extensions
chromadb           # Vector database (if needed)
```

### **External Services**

1. **Ollama** (Local)
   - URL: `http://localhost:11434`
   - Model: `llama3.1:8b`
   - Used for: SQL generation from natural language
   - Optional: App works in offline mode if unavailable

---

## 🔄 State Management

### **Streamlit Session State**

Maintained in `app.py`:
- `uploaded_df` - Current uploaded DataFrame
- `processed_files` - List of processed files
- `mappings` - Column mappings
- `ai_chat_history` - Chat conversation history
- `ai_assistant` - AIAssistant instance
- `has_existing_data` - Database status flag
- `total_rows_loaded` - Row count tracking

---

## 📈 Performance Optimizations

### **Caching Mechanisms**

1. **Streamlit Cache** (`@st.cache_resource`)
   - DuckDB connection caching
   - Reduces connection overhead

2. **LRU Caches** (in `ai_assistant.py`)
   - Question cache (50 entries)
   - SQL cache (50 entries)
   - Analysis cache (50 entries)
   - Schema cache (unlimited)

3. **Query Optimization**
   - Automatic LIMIT clauses
   - Index usage
   - Efficient SQL generation

### **Database Optimizations**

1. **Indexes** (auto-created by `db_manager.py`)
   - Order ID
   - SKU
   - Invoice Date
   - Transaction Type

2. **Connection Pooling**
   - Single cached connection
   - Reused across queries

---

## 🚨 Error Handling

### **Error Handling Strategy**

1. **Database Errors**
   - Location: `db_manager.py`
   - Handles: Connection failures, query errors
   - Recovery: Graceful fallback, error logging

2. **AI Assistant Errors**
   - Location: `ai_assistant.py`
   - Handles: Ollama connection, SQL generation, query execution
   - Recovery: Offline mode, cached responses, error messages

3. **Data Processing Errors**
   - Location: `app.py`
   - Handles: File upload, column mapping, data cleaning
   - Recovery: Validation messages, manual override options

---

## 🔄 Cache Invalidation

### **When Caches Are Cleared**

1. **Question Cache & SQL Cache**
   - When new data is uploaded (`clear_all_caches()`)
   - When column mapping changes
   - When SQL cache explicitly cleared

2. **Schema Cache**
   - When database structure changes
   - Cleared automatically on data updates

3. **Analysis Cache**
   - Cleared with question cache
   - Dependent on data changes

---

## 📝 Key Data Structures

### **1. Column Mappings**
```python
{
    "order_date": "Invoice Date",
    "order_id": "Invoice Number",
    "sku": "Sku",
    # ... etc
}
```

### **2. Cleaning Report**
```python
{
    'total_rows_read': int,
    'rows_kept': int,
    'rows_dropped': int,
    'duplicates_found': int,
    # ... etc
}
```

### **3. AI Query Result**
```python
{
    'success': bool,
    'response': str,
    'sql': str,
    'result_rows': int,
    'response_time': float,
    'cached': bool,
    # ... etc
}
```

---

## 🎯 Integration Points

### **1. Streamlit ↔ Database**
- **Connection:** `db_manager.get_connection()`
- **Operations:** Store, query, count, clear
- **Caching:** Yes (via `@st.cache_resource`)

### **2. Streamlit ↔ AI Assistant**
- **Connection:** `AIAssistant` instance in session state
- **Operations:** `ask_question()`, cache management
- **Caching:** Yes (internal LRU caches)

### **3. AI Assistant ↔ Database**
- **Connection:** Uses `db_manager.query_data()`
- **Operations:** Execute generated SQL queries
- **Caching:** Yes (SQL query results cached)

### **4. AI Assistant ↔ Ollama**
- **Connection:** HTTP requests to `http://localhost:11434`
- **Operations:** SQL generation, model checks
- **Fallback:** Offline mode if unavailable

---

## 🔍 Monitoring & Logging

### **Log Files**

1. **`ai_assistant_errors.log`**
   - AI Assistant errors
   - SQL generation failures
   - Query execution errors

2. **`ai_query_log.csv`**
   - All AI queries
   - Performance metrics
   - Success/failure tracking

### **Logging Levels**

- **INFO:** Normal operations, cache hits/misses
- **WARNING:** Non-critical issues, fallback modes
- **ERROR:** Failures, exceptions

---

## 🚀 Deployment Considerations

### **File Permissions**
- Database file: Read/write required
- Data folders: Write access needed
- Log files: Write access needed

### **Resource Requirements**
- **Memory:** Moderate (depends on data size)
- **Disk:** Moderate (database + CSV files)
- **Network:** None (Ollama is local)

### **Scalability**
- DuckDB handles large datasets efficiently
- Caching reduces repeated computations
- Indexes optimize query performance

---

## 📚 Summary

This CSV Analytics Dashboard is a **three-tier architecture**:

1. **Presentation Layer** (`app.py`) - Streamlit UI
2. **Business Logic Layer** (`ai_assistant.py`, data cleaning) - Processing
3. **Data Layer** (`db_manager.py`, DuckDB) - Storage

All components are **loosely coupled** and communicate through well-defined interfaces, making the system **maintainable** and **extensible**.

---

*Last Updated: Based on latest codebase review*  
*Version: 1.0*  
*Components: 4 core modules + 9 major features*

