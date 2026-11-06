@echo off
setlocal
cd /d "%~dp0"
set "STREAMLIT_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "STREAMLIT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined MODEL_MODE set "MODEL_MODE=real"
echo =======================================================================
echo   STARTING BRAIN TUMOR CADx MEDICAL AI STREAMLIT PLATFORM
echo =======================================================================
"%STREAMLIT_PYTHON%" -m streamlit run app.py --server.port 8501 --theme.base "light"
pause
