@echo off
echo Starting Object Detection Web App...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting server...
echo Open your browser and go to: http://localhost:8000
echo.
python app.py
pause
