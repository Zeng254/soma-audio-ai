#!/bin/bash
# ============================================================================
# SOMA AI Workstation - Linux/macOS Build Script
#
# This script builds the SOMA application using PyInstaller and creates
# a distributable ZIP package.
#
# Prerequisites:
#   - Python 3.12+ installed
#   - uv package manager installed (pip install uv)
#   - All dependencies installed (uv sync)
#
# Usage:
#   ./build.sh              # Full build
#   ./build.sh --quick      # Quick build (skip dependency install)
#   ./build.sh --clean      # Clean build
# ============================================================================

set -eo pipefail

echo ""
echo "===================================================================="
echo "  SOMA AI Workstation - Build"
echo "===================================================================="
echo ""

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
QUICK_BUILD=0
CLEAN_BUILD=0

for arg in "$@"; do
    case $arg in
        --quick) QUICK_BUILD=1 ;;
        --clean) CLEAN_BUILD=1 ;;
    esac
done

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    ARCH=$(uname -m)
else
    PLATFORM="linux"
    ARCH=$(uname -m)
fi

echo "[1/6] Platform: $PLATFORM ($ARCH)"

# Check Python version
echo "[2/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.12+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "       Python version: $PYTHON_VERSION"

# Check/install dependencies
echo "[3/6] Checking dependencies..."
if command -v uv &> /dev/null && [ "$QUICK_BUILD" -eq 0 ]; then
    echo "       Installing dependencies with uv..."
    uv sync --extra dev
elif [ "$QUICK_BUILD" -eq 0 ]; then
    echo "       Installing PyInstaller with pip..."
    pip3 install pyinstaller
fi

# Ensure PyInstaller is available
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "       Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Clean if requested
if [ "$CLEAN_BUILD" -eq 1 ]; then
    echo "[4/6] Cleaning previous build..."
    rm -rf build/ dist/
else
    echo "[4/6] Skipping clean (use --clean to force clean build)"
fi

# Build with PyInstaller
echo "[5/6] Building with PyInstaller..."
echo "       This may take several minutes on first build..."
echo ""

pyinstaller soma.spec --noconfirm --clean

echo ""
echo "[6/6] Creating ZIP package..."

# Get version
VERSION=$(python3 -c "import tomllib; f=open('pyproject.toml','rb'); print(tomllib.load(f)['project']['version'])" 2>/dev/null || echo "0.1.0")

ZIP_NAME="SOMA-v${VERSION}-${PLATFORM}-${ARCH}"
ZIP_PATH="dist/${ZIP_NAME}.zip"

# Create temporary directory for packaging
PKG_DIR="dist/_package"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/SOMA"

# Copy build output
cp -r dist/SOMA/* "$PKG_DIR/SOMA/"

# Copy additional files
cp README.md "$PKG_DIR/SOMA/" 2>/dev/null || true
cp LICENSE "$PKG_DIR/SOMA/" 2>/dev/null || true

# Create user directories
mkdir -p "$PKG_DIR/SOMA/models"
mkdir -p "$PKG_DIR/SOMA/output"

# Create README for the package
cat > "$PKG_DIR/SOMA/README.txt" << EOF
SOMA AI Workstation v${VERSION}
====================================

Quick Start:
  1. Run ./SOMA to launch the application
  2. Models are stored in: ~/.soma/models
  3. Output files are saved to: ~/.soma/output

System Requirements:
  - ${PLATFORM^} ${ARCH}
  - 8GB RAM minimum (16GB recommended)
  - For GPU acceleration: NVIDIA GPU with 6GB+ VRAM

Support:
  - GitHub: https://github.com/Zeng254/soma-audio-ai
EOF

# Make executable
chmod +x "$PKG_DIR/SOMA/SOMA"

# Create ZIP
cd dist/_package
zip -r "../${ZIP_NAME}.zip" SOMA
cd ../..

# Cleanup temp directory
rm -rf "$PKG_DIR"

echo ""
echo "===================================================================="
echo "  Build Complete!"
echo "===================================================================="
echo ""
echo "  Executable:  dist/SOMA/SOMA"
echo "  ZIP Package: ${ZIP_PATH}"
echo ""
echo "  To run: ./dist/SOMA/SOMA"
echo "  To distribute: Share ${ZIP_PATH}"
echo ""
echo "===================================================================="
