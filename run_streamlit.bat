@echo off
REM Script para ejecutar TCA Software Solutions en Windows

echo.
echo ===============================================
echo   TCA Software Solutions - Streamlit App
echo ===============================================
echo.
echo URL local: http://localhost:8501
echo Presiona Ctrl+C para detener la aplicacion
echo.

setlocal enabledelayedexpansion
set PYTHONPATH=%PYTHONPATH%;src
python -m streamlit run src/assist_v10/streamlit_app.py
pause
