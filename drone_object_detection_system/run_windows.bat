@echo off
TITLE Drone Object Detection System Using AI - Launcher
COLOR 0B

echo =====================================================================
echo       DRONE OBJECT DETECTION SYSTEM USING AI - WINDOWS LAUNCHER
echo =====================================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not detected in your system PATH!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)

echo [1/4] Python detected successfully.
echo.

:: 2. Check and create virtual environment
if not exist "venv" (
    echo [2/4] Creating virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
) else (
    echo [2/4] Virtual environment (venv) already exists.
)

:: 3. Activate virtual environment and install requirements
echo [3/4] Activating venv and verifying dependencies...
call venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies had installation issues. Trying core packages...
    pip install streamlit folium streamlit-folium ultralytics opencv-python-headless pandas numpy shapely
)

echo.
echo [4/4] Starting Streamlit Application...
echo The dashboard will automatically open in your default web browser.
echo Address: http://localhost:8501
echo.
echo Press CTRL+C in this command window to stop the server.
echo =====================================================================
echo.

streamlit run app.py

pause