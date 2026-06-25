# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件 for SOMA

使用方法:
    pyinstaller soma.spec              # 打包
    pyinstaller soma.spec --clean       # 清理后重新打包
    pyinstaller soma.spec --onedir      # 打包为目录
    pyinstaller soma.spec --onefile     # 打包为单文件

Windows 用户:
    使用 pyinstaller soma.spec
    打包完成后在 dist/ 目录下会生成 soma.exe

Linux/macOS 用户:
    使用 pyinstaller soma.spec
    打包完成后在 dist/ 目录下会生成 soma 可执行文件
"""

import sys
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# ============================================================================
# 项目配置
# ============================================================================

project_root = Path(__file__).parent
src_dir = project_root / "src"

# 应用信息
APP_NAME = "SOMA"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "AI驱动的音频处理工作站"
APP_AUTHOR = "SOMA Team"

# 打包类型: onedir (目录) 或 onefile (单文件)
ONE_FILE = False  # 设为 True 打包为单文件

# ============================================================================
# 隐藏导入 (Hidden Imports)
# 所有在运行时动态导入的模块都需要在这里声明
# ============================================================================

hiddenimports = [
    # PyQt6 核心
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',

    # 音频处理
    'torch',
    'torch.cuda',
    'torch.nn',
    'torch.nn.functional',
    'librosa',
    'librosa.core',
    'librosa.output',
    'soundfile',
    'scipy',
    'scipy.signal',
    'numpy',

    # 模型相关
    'fairseq',
    'pyworld',
    
    # 语音转换器（voice_converters）
    'src.voice_converters.base',
    'src.voice_converters.factory',
    'src.voice_converters.rvc_converter',
    'src.voice_converters.sovits_converter',
    'src.voice_converters.diffusion_converter',

    # 配置和工具
    'yaml',
    'json',
    'logging',
    'pathlib',
    'threading',
    'queue',

    # GUI 组件
    'gui.main_window',
    'gui.components.audio_input_panel',
    'gui.components.model_config_panel',
    'gui.components.output_panel',
    'gui.components.status_bar',
    'gui.workers.conversion_worker',
    'gui.styles.dark_theme',

    # 核心模块
    'src.exceptions',
    'src.config.config',
    'src.config.defaults',
    'src.security.path_validator',
    'src.security.audio_validator',
    'src.security.model_loader',
    'src.utils.audio_io',
    'src.utils.validator',
    'src.utils.logger',
    'src.separators.base',
    'src.separators.demucs_separator',
    'src.separators.msst_separator',
    'src.effects.base',
    'src.effects.eq',
    'src.effects.reverb',
    'src.effects.pitch',
    'src.converters.converter',
    'src.voice_converters.base',
    'src.voice_converters.factory',
    'src.voice_converters.rvc_converter',
    'src.voice_converters.sovits_converter',
    'src.pipeline.pipeline',
]

# ============================================================================
# 数据文件 (Data Files)
# ============================================================================

datas = [
    # 配置文件
    (str(project_root / 'config'), 'config'),
]

# ============================================================================
# 收集第三方库的数据文件
# ============================================================================

# PyInstaller 4.x 使用 collect_data_files
try:
    # PyTorch
    torch_datas, torch_binaries, torch_redirected_binaries_neg_whitelist = collect_all('torch')
    datas += torch_datas
    hiddenimports += ['torch.' + m for m in collect_submodules('torch')]

    # NumPy
    numpy_datas, numpy_binaries, numpy_redirected_binaries_neg_whitelist = collect_all('numpy')
    datas += numpy_datas
    hiddenimports += ['numpy.' + m for m in collect_submodules('numpy')]

except Exception as e:
    print(f"Warning: Could not collect all data files: {e}")

# ============================================================================
# 排除模块 (Excludes)
# ============================================================================

excludes = [
    'tkinter',
    'matplotlib',
    'IPython',
    'notebook',
    'jupyter',
    'test',
    'tests',
    'pytest',
    'venv',
    '.venv',
    '__pycache__',
]

# ============================================================================
# PyInstaller 选项
# ============================================================================

a = Analysis(
    ['gui/main_window.py'],  # 入口文件
    pathex=[str(project_root)],
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if ONE_FILE:
    # 单文件打包
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
        console=False,  # GUI 程序不显示控制台
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,  # 可以添加图标: icon='assets/icon.ico'
        version='version_info.txt',  # Windows 版本信息
    )
    coll = COLLECT(
        exe,
        a.binaries + a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
else:
    # 目录打包 (推荐)
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
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=None,
    )

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
