@echo off
REM ============================================================================
REM SOMA AI Workstation - Windows Build Script
REM 
REM This script builds the SOMA application using PyInstaller and creates
REM a distributable ZIP package.
REM
REM Prerequisites:
REM   - Python 3.12+ installed and in PATH
REM   - uv package manager installed (pip install uv)
REM   - All dependencies installed (uv sync)
REM
REM Usage:
REM   build_windows.bat              # Full build
REM   build_windows.bat --quick      # Quick build (skip dependency install)
REM   build_windows.bat --clean      # Clean build
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo   SOMA AI Workstation - Windows Build
echo ====================================================================
echo.

REM Get project root directory
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

REM Parse arguments
set "QUICK_BUILD=0"
set "CLEAN_BUILD=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--quick" set "QUICK_BUILD=1"
if /i "%~1"=="--clean" set "CLEAN_BUILD=1"
shift
goto :parse_args
:args_done

REM Check Python version
echo [1/6] Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.12+ and add to PATH.
    exit /b 1
)

REM Check uv
echo [2/6] Checking uv package manager...
where uv >nul 2>nul
if errorlevel 1 (
    echo WARNING: uv not found. Using pip instead.
    echo Installing PyInstaller with pip...
    pip install pyinstaller
) else (
    if "%QUICK_BUILD%"=="0" (
        echo Installing dependencies with uv...
        uv sync --extra dev
    )
)

REM Install PyInstaller if needed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean if requested
if "%CLEAN_BUILD%"=="1" (
    echo [3/6] Cleaning previous build...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    if exist soma.spec.bak del soma.spec.bak
) else (
    echo [3/6] Skipping clean (use --clean to force clean build)
)

REM Build with PyInstaller
echo [4/6] Building with PyInstaller...
echo        This may take several minutes on first build...
echo.

pyinstaller soma.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed!
    echo Check the output above for errors.
    exit /b 1
)

echo.
echo [5/6] Build successful!

REM Create ZIP package
echo [6/6] Creating ZIP package...

REM Get version from pyproject.toml
for /f "tokens=*" %%i in ('python -c "import tomllib; f=open('pyproject.toml','rb'); print(tomllib.load(f)['project']['version'])"') do set "VERSION=%%i"

set "ZIP_NAME=SOMA-v%VERSION%-win64"
set "ZIP_PATH=dist\%ZIP_NAME%.zip"

REM Create temporary directory for packaging
set "PKG_DIR=dist\_package"
if exist "%PKG_DIR%" rmdir /s /q "%PKG_DIR%"
mkdir "%PKG_DIR%"
mkdir "%PKG_DIR%\SOMA"

REM Copy build output
xcopy /e /q /y "dist\SOMA\*" "%PKG_DIR%\SOMA\"

REM Copy additional files
copy /y README.md "%PKG_DIR%\SOMA\" 2>nul
copy /y LICENSE "%PKG_DIR%\SOMA\" 2>nul

REM Create user directories
mkdir "%PKG_DIR%\SOMA\models" 2>nul
mkdir "%PKG_DIR%\SOMA\output" 2>nul

REM Create a simple README for the package
(
echo SOMA AI Workstation v%VERSION%
echo ====================================
echo.
echo Quick Start:
echo   1. Double-click SOMA.exe to launch
echo   2. Models are stored in: %%USERPROFILE%%\.soma\models
echo   3. Output files are saved to: %%USERPROFILE%%\.soma\output
echo.
echo System Requirements:
echo   - Windows 10/11 64-bit
echo   - 8GB RAM minimum (16GB recommended)
echo   - For GPU acceleration: NVIDIA GPU with 6GB+ VRAM
echo.
echo Support:
echo   - GitHub: https://github.com/Zeng254/soma-audio-ai
echo.
) > "%PKG_DIR%\SOMA\README.txt"

REM Create ZIP
cd dist\_package
python -c "import shutil; shutil.make_archive('../%ZIP_NAME%', 'zip', '.', 'SOMA')"
cd ..\..

REM Cleanup temp directory
rmdir /s /q "%PKG_DIR%"

echo.
echo ====================================================================
echo   Build Complete!
echo ====================================================================
echo.
echo   Executable:  dist\SOMA\SOMA.exe
echo   ZIP Package: %ZIP_PATH%
echo.
echo   To run: Double-click dist\SOMA\SOMA.exe
echo   To distribute: Share %ZIP_PATH%
echo.
echo ====================================================================

endlocal
