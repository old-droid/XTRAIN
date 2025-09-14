@echo off
REM CPUWARP-ML Setup Script for Windows
REM Installs dependencies and compiles optimized extensions

echo ============================================================
echo CPUWARP-ML Framework Setup for Windows
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; print(f'Python {sys.version}')"
echo.

REM Install required Python packages
echo Installing Python dependencies...
pip install --upgrade pip
pip install numpy scipy psutil py-cpuinfo

REM Check if we have a C compiler (Visual Studio Build Tools)
echo.
echo Checking for C compiler...
where cl >nul 2>&1
if errorlevel 1 (
    echo Warning: No C compiler found (cl.exe)
    echo C extensions will not be compiled
    echo Framework will work with NumPy fallbacks
    echo.
    echo To enable C extensions, install:
    echo - Visual Studio 2019/2022 with C++ Build Tools
    echo - OR Windows 10/11 SDK
    echo - OR MinGW-w64
    goto :skip_compile
)

echo C compiler found. Attempting to compile extensions...
python setup.py build_ext --inplace
if errorlevel 1 (
    echo Warning: Failed to compile C extensions
    echo Framework will work with NumPy fallbacks
) else (
    echo C extensions compiled successfully!
)

:skip_compile

REM Test the installation
echo.
echo Testing CPUWARP-ML installation...
python -c "import cpuwarp_ml; print('CPUWARP-ML imported successfully')"
if errorlevel 1 (
    echo Error: Failed to import CPUWARP-ML
    pause
    exit /b 1
)

REM Run basic performance test
echo.
echo Running basic performance test...
python cpuwarp_ml.py

echo.
echo ============================================================
echo CPUWARP-ML Setup Complete!
echo ============================================================
echo.
echo Available scripts:
echo - train_cnn.py  : Train CNN models
echo - train_llm.py  : Train LLM models
echo.
echo Example usage:
echo   python train_cnn.py --batch-size 8 --input-size 32
echo   python train_llm.py --d-model 256 --batch-size 4
echo.
echo For help: python [script].py --help
echo.
pause