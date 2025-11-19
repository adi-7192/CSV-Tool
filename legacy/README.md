# Legacy Streamlit App

This folder contains the original Streamlit application kept as reference.

## Status

**⚠️ This version is deprecated.** Use the new React + FastAPI version in `../backend/`.

## Running the Legacy App

```bash
cd legacy
streamlit run app.py
```

**Note**: This app uses the same `../data/` folder as the new backend, so both can access the same database.

## Files

- `app.py` - Main Streamlit application
- `ai_assistant.py` - AI Assistant module
- `db_manager.py` - Database manager
- `diagnose.py` - Diagnostic tool
- `reconcile.py` - Reconciliation script

## Testing

Legacy test suite is located in `../tests/` and should still work:

```bash
cd ..
pytest tests/ -v
```

