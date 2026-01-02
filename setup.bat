@echo off
REM Setup script for learning_parameters repository (Windows)
REM Don't exit on error - we want to track failures and report them at the end
setlocal enabledelayedexpansion

echo =========================================
echo Setting up learning_parameters repository
echo =========================================
echo.

REM Initialize failed packages tracking
set FAILED_PACKAGES=
set FAILED_COUNT=0

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Get Python version for checking
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Found Python %PYTHON_VERSION%

REM Check for Python 3.14+ compatibility issues
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)
if "!PYTHON_MAJOR!"=="3" (
    if !PYTHON_MINOR! geq 14 (
        echo.
        echo WARNING: Python 3.14+ may have compatibility issues with some packages.
        echo    Some packages (cvxpy, osqp) may not have pre-built wheels and will require
        echo    compilation. Consider using Python 3.11 or 3.12 for better compatibility.
        echo.
    )
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
if errorlevel 1 (
    echo Warning: Could not activate venv, continuing anyway...
)

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo Warning: Failed to upgrade pip, continuing anyway...
)

REM Install PyTorch (CPU version by default)
echo.
echo Installing PyTorch (CPU version)...
echo Note: For CUDA support, install PyTorch separately:
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio >nul 2>&1
if errorlevel 1 (
    echo Failed to install PyTorch packages
    set FAILED_PACKAGES=torch torchvision torchaudio
    set /a FAILED_COUNT+=3
)

REM Install other requirements
echo.
echo Installing other requirements...
echo Note: Some packages (cvxpy, osqp) may require C++ build tools.
echo If installation fails on Windows, install Microsoft C++ Build Tools from:
echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo.

REM Try to install requirements
echo Attempting to install all packages from requirements.txt...
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Standard installation failed. Attempting to install packages individually...
    echo.
    
    REM Read requirements.txt and try installing each package individually
    for /f "usebackq tokens=* delims=" %%a in ("requirements.txt") do (
        set "line=%%a"
        REM Skip empty lines
        if not "!line!"=="" (
            REM Skip comment lines
            echo !line! | findstr /r "^[ ]*#" >nul
            if errorlevel 1 (
                REM Extract package name - everything before first version specifier or space
                set "package_line=!line!"
                REM Remove leading/trailing spaces
                for /f "tokens=* delims= " %%s in ("!package_line!") do set "package_line=%%s"
                
                REM Extract package name by replacing version specifiers with spaces and taking first token
                set "temp_line=!package_line!"
                set "temp_line=!temp_line:>= =!"
                set "temp_line=!temp_line:<= =!"
                set "temp_line=!temp_line:== =!"
                set "temp_line=!temp_line:!= =!"
                set "temp_line=!temp_line:> =!"
                set "temp_line=!temp_line:< =!"
                
                for /f "tokens=1" %%p in ("!temp_line!") do set "package_name=%%p"
                
                if not "!package_name!"=="" (
                    echo Installing !package_name!...
                    pip install "!line!" >nul 2>&1
                    set INSTALL_RESULT=!errorlevel!
                    if !INSTALL_RESULT! neq 0 (
                        echo Failed to install: !package_name!
                        if "!FAILED_PACKAGES!"=="" (
                            set "FAILED_PACKAGES=!package_name!"
                        ) else (
                            set "FAILED_PACKAGES=!FAILED_PACKAGES! !package_name!"
                        )
                        set /a FAILED_COUNT+=1
                    ) else (
                        echo Successfully installed: !package_name!
                    )
                )
            )
        )
    )
) else (
    echo All packages installed successfully from requirements.txt
)

REM Verify installation
echo.
echo Verifying installation...
set VERIFICATION_FAILED=0
set VERIFICATION_ERRORS=

REM Check core packages
for %%p in (torch numpy h5py) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        set VERIFICATION_FAILED=1
        if "!VERIFICATION_ERRORS!"=="" (
            set "VERIFICATION_ERRORS=%%p"
        ) else (
            set "VERIFICATION_ERRORS=!VERIFICATION_ERRORS! %%p"
        )
    )
)

REM Check qdarts (may fail if cvxpy is missing, but that's okay)
python -c "import qdarts" >nul 2>&1
if errorlevel 1 (
    echo Warning: qdarts import failed (may be due to missing cvxpy/osqp)
    if "!VERIFICATION_ERRORS!"=="" (
        set "VERIFICATION_ERRORS=qdarts (may require cvxpy)"
    ) else (
        set "VERIFICATION_ERRORS=!VERIFICATION_ERRORS! qdarts (may require cvxpy)"
    )
)

REM Check optional packages that might have failed
for %%p in (cvxpy osqp) do (
    python -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        REM Check if package is already in failed list
        echo !FAILED_PACKAGES! | findstr /c:"%%p" >nul
        if errorlevel 1 (
            REM Package was supposed to be installed but import failed
            if "!FAILED_PACKAGES!"=="" (
                set "FAILED_PACKAGES=%%p"
            ) else (
                set "FAILED_PACKAGES=!FAILED_PACKAGES! %%p"
            )
            set /a FAILED_COUNT+=1
        )
    )
)

if !VERIFICATION_FAILED!==0 (
    if "!VERIFICATION_ERRORS!"=="" (
        echo Core packages verified successfully!
    ) else (
        echo Verification warnings:
        for %%e in (!VERIFICATION_ERRORS!) do echo    - %%e
    )
) else (
    echo Verification warnings:
    for %%e in (!VERIFICATION_ERRORS!) do echo    - %%e
)

REM Final summary
echo.
echo =========================================
if !FAILED_COUNT!==0 (
    echo Setup complete!
) else (
    echo Setup completed with warnings
)
echo =========================================
echo.

REM Report failed packages
if not "!FAILED_PACKAGES!"=="" (
    echo The following packages failed to install:
    echo.
    for %%p in (!FAILED_PACKAGES!) do echo    - %%p
    echo.
    echo To manually install failed packages, activate the venv and run:
    echo   venv\Scripts\activate.bat
    echo   pip install !FAILED_PACKAGES!
    echo.
    echo =========================================
    echo.
)

echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
pause
