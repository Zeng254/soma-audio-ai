# -*- mode: python ; coding: utf-8 -*-
"""
SOMA AI Workstation - PyInstaller Build Configuration

Build commands:
    pyinstaller soma.spec                  # Build (incremental)
    pyinstaller soma.spec --clean          # Clean build
    pyinstaller soma.spec --noconfirm      # Overwrite without asking

Output:
    dist/SOMA/          # Directory mode (default)
    dist/SOMA.exe       # Single file mode (set ONE_FILE=True below)

Final package:
    dist/SOMA-v0.1.0-win64.zip
"""

import sys
import os
from pathlib import Path

block_cipher = None

# ============================================================================
# Project Configuration
# ============================================================================

project_root = Path(__file__).parent
src_dir = project_root / "src"

# Application info
APP_NAME = "SOMA"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "SOMA AI Audio Workstation"
APP_AUTHOR = "SOMA Team"

# Build mode: False = directory (recommended), True = single file
ONE_FILE = False

# ============================================================================
# Hidden Imports
# ============================================================================

hiddenimports = [
    # --- GUI (tkinter is built-in, but we need to ensure it's included) ---
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.colorchooser',
    'tkinter.font',

    # --- Audio Processing ---
    'numpy',
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'scipy.io',
    'soundfile',
    'librosa',
    'librosa.core',
    'librosa.feature',
    'librosa.effects',
    'pydub',

    # --- PyTorch (CPU-only for smaller package) ---
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchaudio',

    # --- SOMA Core Modules ---
    'src',
    'src.exceptions',
    'src.config',
    'src.config.config',
    'src.config.defaults',

    # --- Security ---
    'src.security',
    'src.security.path_validator',
    'src.security.audio_validator',
    'src.security.model_loader',

    # --- Audio Processing ---
    'src.utils',
    'src.utils.audio',
    'src.utils.audio.validation',
    'src.utils.audio_io',
    'src.utils.file',
    'src.utils.file.file',
    'src.utils.file.file_utils',
    'src.utils.logger',
    'src.utils.validator',

    # --- Separators ---
    'src.separators',
    'src.separators.base',
    'src.separators.audio_separator',
    'src.separators.demucs_separator',
    'src.separators.msst_separator',

    # --- Effects ---
    'src.effects',
    'src.effects.base',
    'src.effects.eq',
    'src.effects.reverb',
    'src.effects.pitch',

    # --- Voice Converters ---
    'src.voice_converters',
    'src.voice_converters.base',
    'src.voice_converters.factory',
    'src.voice_converters.rvc_converter',
    'src.voice_converters.rvc_models',
    'src.voice_converters.sovits_converter',
    'src.voice_converters.sovits_models',

    # --- Converters ---
    'src.converters',
    'src.converters.converter',

    # --- Pipeline ---
    'src.pipeline',
    'src.pipeline.pipeline',

    # --- Training ---
    'src.training',
    'src.training.config',
    'src.training.dataset',
    'src.training.feature_extractor',
    'src.training.inference',
    'src.training.preprocess',
    'src.training.trainer',
    'src.training.cover_pipeline',
    'src.training.cli',

    # --- Storage ---
    'src.storage',
    'src.storage.database',
    'src.storage.database.db',
    'src.storage.database.shared',
    'src.storage.database.shared.model',
    'src.storage.memory',
    'src.storage.memory.memory_saver',

    # --- GUI ---
    'gui',
    'gui.app',
    'gui.main',
    'gui.styles',
    'gui.pages',
    'gui.pages.base',
    'gui.pages.dashboard',
    'gui.pages.training',
    'gui.pages.models',
    'gui.pages.settings',
    'gui.widgets',
    'gui.widgets.navigation',
    'gui.utils',
    'gui.utils.constants',
    'gui.utils.common',
    'gui.utils.settings_manager',

    # --- GUI Pages (Package structure) ---
    'gui.pages.separation',
    'gui.pages.separation.page',
    'gui.pages.separation.ui_mixin',
    'gui.pages.separation.worker_mixin',
    'gui.pages.inference',
    'gui.pages.inference.page',
    'gui.pages.inference.ui_mixin',
    'gui.pages.inference.worker_mixin',
    'gui.pages.comparison',
    'gui.pages.comparison.page',
    'gui.pages.comparison.ui_mixin',
    'gui.pages.comparison.worker_mixin',
    'gui.pages.comparison.playback_mixin',

    # --- Third-party ---
    'yaml',
    'pydantic',
    'PIL',
    'PIL.Image',
]

# ============================================================================
# Data Files
# ============================================================================

datas = []

# Include any config files if they exist
config_dir = project_root / 'config'
if config_dir.exists():
    datas.append((str(config_dir), 'config'))

# Include assets if they exist
assets_dir = project_root / 'assets'
if assets_dir.exists():
    datas.append((str(assets_dir), 'assets'))

# ============================================================================
# Excludes (reduce package size)
# ============================================================================

excludes = [
    # Testing
    'pytest',
    'pytest_asyncio',
    'pytest_mock',
    'unittest',
    'test',
    'tests',

    # Development
    'black',
    'ruff',
    'pylint',
    'mypy',

    # Jupyter/IPython
    'IPython',
    'notebook',
    'jupyter',
    'ipykernel',

    # Web frameworks (not needed for desktop)
    'django',
    'flask',

    # Database drivers (not needed for desktop)
    'psycopg2',
    'psycopg',
    'boto3',
    'botocore',

    # Document processing (not needed)
    'pypdf',
    'docx2python',
    'openpyxl',
    'python_pptx',

    # LLM/AI frameworks (not needed for desktop)
    'langchain',
    'langchain_openai',
    'langgraph',
    'langsmith',
    'cozeloop',
    'coze_coding_utils',
    'coze_workload_identity',
    'coze_coding_dev_sdk',

    # Other unused
    'matplotlib',
    'cv2',
    'pandas',
    'alembic',
    'SQLAlchemy',
    'fastapi',
    'uvicorn',
    'Jinja2',
    'rich',
    'chardet',
    'ffmpeg_python',
    'psutil',
    'python_dotenv',
    'pyyaml',

    # Virtual environment
    'venv',
    '.venv',
    '__pycache__',
]

# ============================================================================
# Analysis
# ============================================================================

a = Analysis(
    ['launcher.py'],
    pathex=[
        str(project_root),
        str(src_dir),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ============================================================================
# PYZ (Python Archive)
# ============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ============================================================================
# EXE
# ============================================================================

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add icon when available: icon='assets/icon.ico'
    version='version_info.txt',  # Windows version info
)

# ============================================================================
# COLLECT (Directory mode)
# ============================================================================

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
