@echo off
echo ============================================
echo   Facial Emotion Recognition - Image Mode
echo ============================================
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
if "%1"=="" (
    echo Usage: run_image_test.bat [image_path]
    echo Example: run_image_test.bat img1.jpeg
    echo.
    echo No image specified - using default img2.jpeg
    python testdata.py
) else (
    python testdata.py %1
)
pause
