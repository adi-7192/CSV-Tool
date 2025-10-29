# 📊 CSV Analytics Dashboard

**Production-ready business analytics platform** with AI-powered insights.

**Status**: ✅ Phase 0 Complete | 🚀 Phase 1: FastAPI Backend (In Progress)

---

## 🎯 Quick Start

### **Option 1: New FastAPI Backend (Recommended)**

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload

# API available at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

### **Option 2: Legacy Streamlit App (Reference Only)**

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
cd legacy
streamlit run app.py
```

---

## 📁 Project Structure

```
Nisarg Project/
│
├── backend/              ← 🆕 FastAPI backend (NEW)
│   ├── main.py          # FastAPI app entry point
│   ├── api/             # API routes
│   ├── core/            # Configuration, database, AI service
│   ├── models/          # Pydantic request/response models
│   ├── services/        # Business logic layer
│   └── tests/           # API tests
│
├── legacy/               ← Original Streamlit app (reference)
│   ├── app.py
│   ├── ai_assistant.py
│   └── db_manager.py
│
├── data/                 ← Shared database (both use this)
│   ├── analytics.duckdb
│   ├── raw/
│   └── cleaned/
│
└── tests/                ← Test suite (works for both)
```

---

## 🚀 Features

✅ **Multi-File CSV Upload** - Automatic column mapping  
✅ **Business Intelligence** - Comprehensive KPIs and analytics  
✅ **AI-Powered Insights** - Natural language queries (Ollama)  
✅ **Data Trust** - Validation, deduplication, lineage tracking  
✅ **Production Ready** - Docker, tests, reconciliation  

---

## 📚 Documentation

- **AFTER_PHASE_0_IMPLEMENTATION.md** - Complete Phase 0 summary
- **ARCHITECTURE.md** - System architecture details
- **DOCKER_SETUP.md** - Docker deployment guide
- **backend/** - FastAPI backend (NEW)

---

## 🔧 Development

### Backend API

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run backend API tests
pytest backend/tests/ -v
```

### Docker

```bash
docker-compose up --build
```

---

## 📝 Status

**Phase 0**: ✅ Complete
- Streamlit app functional
- All features implemented
- Tests passing (38/38)

**Phase 1**: 🚀 In Progress
- FastAPI backend structure created
- Health endpoints working
- Migration of business logic ongoing

---

## 📄 License

Internal project - See project documentation
