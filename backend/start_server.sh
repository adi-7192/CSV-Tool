#!/bin/bash
# Backend Server Startup Script
# This script ensures the correct environment and directory for starting the FastAPI server

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Navigate to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found at $PROJECT_ROOT/venv"
    echo "Please create a virtual environment first: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Navigate to backend directory
cd "$SCRIPT_DIR"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "❌ Error: uvicorn not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Starting FastAPI server..."
echo "📍 Directory: $SCRIPT_DIR"
echo "🐍 Python: $(which python)"
echo "🔧 Uvicorn: $(which uvicorn)"
echo ""
echo "🚀 Server will be available at: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/api/docs"
echo ""

# Start uvicorn from the backend directory
uvicorn main:app --reload --host 0.0.0.0 --port 8000








