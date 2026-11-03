@echo off
setlocal
cd /d "%~dp0"
set "HYBRID_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "HYBRID_PYTHON=%~dp0.venv\Scripts\python.exe"
"%HYBRID_PYTHON%" -u train.py --model swin_hybrid %*
exit /b %ERRORLEVEL%
