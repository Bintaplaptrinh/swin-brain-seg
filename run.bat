@echo off
setlocal
cd /d "%~dp0"
set "TRAIN_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "TRAIN_PYTHON=%~dp0.venv\Scripts\python.exe"
"%TRAIN_PYTHON%" -c "import torch, dynamic_network_architectures, PIL, numpy, scipy, surface_distance, tqdm" >nul 2>&1
if errorlevel 1 (
    echo Training dependencies are missing. Install them with:
    echo "%TRAIN_PYTHON%" -m pip install -r requirements.txt
    exit /b 1
)
"%TRAIN_PYTHON%" -u train.py %*
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" echo Training exited with code %TRAIN_EXIT%.
exit /b %TRAIN_EXIT%
