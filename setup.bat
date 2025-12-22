@echo off
REM Setup script for learning_parameters repository (Windows)

echo =========================================
echo Setting up learning_parameters repository
echo =========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch (CPU version by default)
echo.
echo Installing PyTorch (CPU version)...
echo Note: For CUDA support, install PyTorch separately:
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio

REM Install other requirements
echo.
echo Installing other requirements...
pip install -r requirements.txt

REM Verify installation
echo.
echo Verifying installation...
python -c "import torch; import numpy; import h5py; import qdarts; print('All core packages imported successfully!')"
if errorlevel 1 (
    echo Installation verification failed!
    exit /b 1
)

echo.
echo =========================================
echo Setup complete!
echo =========================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
pause

