@echo off
title AxIn AI Agent - Demo Launcher
color 0A

echo ===================================================
echo      Starting AxIn AI Agent Demo Environment
echo ===================================================
echo.

echo [1/3] Waking up the local Ollama server...
:: This starts Ollama in the background silently so it doesn't clutter the screen
start /B ollama serve >nul 2>&1
:: Give it 3 seconds to fully initialize
timeout /t 3 >nul 

echo [2/3] Activating secure Python environment...
:: We use 'call' so the script doesn't stop after activation
call ".\AxIn_Help\Scripts\activate.bat"

echo [3/3] Launching the Streamlit interface...
echo.
echo ===================================================
echo   The AxIn AI UI is opening in your browser!
echo   (Do not close this window during the demo)
echo ===================================================
echo.

streamlit run app.py

:: If the app ever crashes, this keeps the window open so you can read the error
pause