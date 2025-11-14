@echo off
echo ============================================================
echo ISL EMERGENCY GESTURE DETECTION - LAUNCHER
echo ============================================================
echo.
echo Select an option:
echo [1] Demo - Show detection on training video
echo [2] Webcam - Live detection from your camera
echo [3] Test - Run system test
echo [4] Batch - Process all videos
echo [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Running demo on help gesture video...
    python yolov5\detect.py --weights data/processed/artifacts/models/yolov5/best.pt --source Dataset/Raw_Data/help_Raw/help_001_01.AVI --view-img --conf 0.25
)

if "%choice%"=="2" (
    echo.
    echo Starting webcam detection...
    echo Press 'q' in the window to quit
    python yolov5\detect.py --weights data/processed/artifacts/models/yolov5/best.pt --source 0 --view-img --conf 0.2
)

if "%choice%"=="3" (
    echo.
    echo Running system test...
    python test_all_features.py
    pause
)

if "%choice%"=="4" (
    echo.
    echo Processing all videos in dataset...
    python yolov5\detect.py --weights data/processed/artifacts/models/yolov5/best.pt --source Dataset/Raw_Data --save-txt --save-conf
)

if "%choice%"=="5" (
    exit
)

pause
