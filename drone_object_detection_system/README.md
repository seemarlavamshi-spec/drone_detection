# DRONE OBJECT DETECTION SYSTEM USING AI 🚁

A complete, production-grade aerial reconnaissance and surveillance system built with Python, YOLOv8/YOLO11, OpenCV, PyTorch, ByteTrack, Streamlit, Folium, and SQLite.

## 🚀 Beginner Quick-Start Guide for Windows

### Step 1: Install Python
1. Download Python 3.10 or 3.11 from https://www.python.org/downloads/
2. CRITICAL: When running the installer, check: "Add Python to PATH"
3. Complete installation.

### Step 2: One-Click Launch
1. Extract the downloaded ZIP to a folder (e.g. C:\DroneAI).
2. Double-click "run_windows.bat".
3. The script automatically creates the virtual environment, installs packages from requirements.txt, and launches Streamlit at http://localhost:8501.

### Step 3: Test Surveillance & Geofencing
1. In Tab 1, upload any recorded video (MP4, AVI, MOV).
2. Click "Start Drone Video Processing".
3. Watch real-time YOLO bounding boxes, ByteTrack persistent IDs, and non-duplicate unique counts.
4. Check Tab 2 for the interactive Folium GPS map and restricted zones.
5. Check Tab 4 for intrusion alerts and Tab 5 for CSV export.