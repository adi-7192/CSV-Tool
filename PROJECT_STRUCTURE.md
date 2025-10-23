# Project Structure

## 📁 Clean Project Structure

```
Nisarg Project/
├── app.py                    # Main Streamlit application (single file)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Data storage directory
│   ├── mapping.json         # Saved column mappings
│   └── data.db              # SQLite database (auto-created)
└── venv/                    # Python virtual environment
    ├── bin/                 # Python executables
    ├── lib/                 # Installed packages
    └── ...
```

## 🗂️ File Descriptions

### Core Files
- **`app.py`** - Complete Streamlit application (584 lines)
  - CSV upload and auto-mapping
  - KPI dashboard with product names
  - Natural language chat system
  - All functionality in single file

- **`requirements.txt`** - Python dependencies
  - streamlit
  - pandas

- **`README.md`** - Project documentation
  - Features, quick start, usage guide

### Data Files
- **`data/mapping.json`** - Saved column mappings for reuse
- **`data.db`** - SQLite database with sales data
- **`venv/`** - Python virtual environment

## 🧹 Cleanup Completed

### Removed Files
- ❌ All sample CSV files (7 files)
- ❌ All documentation guides (15+ files)
- ❌ Unnecessary cache directories
- ❌ Duplicate virtual environment

### Kept Files
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Dependencies
- ✅ `README.md` - Documentation
- ✅ `data/` - Data storage
- ✅ `venv/` - Python environment

## 🚀 Ready for Use

The project is now clean and minimal:
1. **Single file app** - Everything in `app.py`
2. **Minimal dependencies** - Just streamlit + pandas
3. **Clean structure** - Only essential files
4. **Ready to run** - `streamlit run app.py`

## 📊 Features

- **CSV Upload** → Auto-mapping → KPIs → Chat
- **Product names** displayed instead of SKU codes
- **INR currency** formatting throughout
- **Natural language** chat about data
- **Persistent mappings** for future uploads
