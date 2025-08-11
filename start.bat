@echo off
REM PropertyGuru Production Startup Script for Windows

REM Set default values
if not defined FLASK_ENV set FLASK_ENV=production
if not defined FLASK_DEBUG set FLASK_DEBUG=false
if not defined HOST set HOST=0.0.0.0
if not defined PORT set PORT=8000
if not defined WORKERS set WORKERS=1

REM Create necessary directories
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "chromadb_data" mkdir "chromadb_data"
if not exist "logs" mkdir "logs"

REM Check for required environment variables
if "%COHERE_API_KEY%"=="" if "%ZAMEEN_COHERE_API_KEY%"=="" (
    echo ❌ Error: COHERE_API_KEY or ZAMEEN_COHERE_API_KEY environment variable is required
    exit /b 1
)

REM Check for vector database
if not exist "chromadb_data\*" (
    echo ⚠️  Warning: Vector database not found. You may need to run embedding process first:
    echo    python scripts\embed_and_store.py --input data\processed\zameen_phase7_chunks.jsonl
)

REM Check for processed data
if not exist "data\processed\zameen_phase7_processed.json" (
    echo ⚠️  Warning: Processed data not found. You may need to process raw data first:
    echo    python scripts\clean_and_enrich.py
)

echo 🚀 Starting PropertyGuru in production mode...
echo 📍 Host: %HOST%
echo 🔌 Port: %PORT%
echo 👥 Workers: %WORKERS%

REM Check if Gunicorn is available
gunicorn --version >nul 2>&1
if %errorlevel%==0 (
    echo 🔄 Using Gunicorn production server...
    gunicorn ^
        --bind %HOST%:%PORT% ^
        --workers %WORKERS% ^
        --timeout 120 ^
        --keep-alive 60 ^
        --max-requests 1000 ^
        --max-requests-jitter 100 ^
        --access-logfile logs\access.log ^
        --error-logfile logs\error.log ^
        --log-level info ^
        --preload ^
        web_ui.app:app
) else (
    echo 🔄 Using Flask development server...
    echo ⚠️  Consider installing Gunicorn for production: pip install gunicorn
    python web_ui\app.py
)
