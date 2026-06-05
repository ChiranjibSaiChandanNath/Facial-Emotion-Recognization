@echo off
echo ============================================
echo   Facial Emotion Recognition - Webcam Mode
echo ============================================
set TF_CPP_MIN_LOG_LEVEL=3
set TF_ENABLE_ONEDNN_OPTS=0
python test.py
pause
