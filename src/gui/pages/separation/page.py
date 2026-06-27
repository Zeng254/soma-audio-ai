"""
SeparationPage - 音频分离主页面类。

组合 SeparationUIMixin 和 SeparationWorkerMixin 提供完整的分离页面功能。

MRO（方法解析顺序）:
    SeparationPage -> SeparationUIMixin -> SeparationWorkerMixin -> BasePage -> object
    - SeparationUIMixin 提供: _create_widgets, _create_*_section, _browse_*, 文件信息
    - SeparationWorkerMixin 提供: _start_separation, _stop_separation, _separation_worker,
      _separation_complete, _separation_error, 计时器, 日志
    - BasePage 提供: safe_after, _widget_alive, cleanup, on_show, on_hide
    - Mixin 之间无方法名冲突（已验证）。

Bug 修复:
    - Bug 1: 所有变量初始化在 super().__init__() 之前（因为 BasePage.__init__ 调用 _create_widgets）
    - Bug 2: Mixin 类放在 BasePage 前面继承，确保 ABC 抽象方法正确识别
"""

import tkinter as tk
import threading
import os
from typing import Optional, List

from gui.pages.base import BasePage
from gui.utils import SettingsManager

from .ui_mixin import SeparationUIMixin
from .worker_mixin import SeparationWorkerMixin


# Bug 2 修复：Mixin 类放在 BasePage 前面，确保 ABC 抽象方法已实现
# MRO: SeparationPage -> SeparationUIMixin -> SeparationWorkerMixin -> BasePage -> object
class SeparationPage(SeparationUIMixin, SeparationWorkerMixin, BasePage):
    """
    音频分离页面。

    功能:
    - 源音频选择，带文件信息显示和拖放区域
    - 分离模式: 2-stem, 4-stem, HPSS, 去混响
    - 后端选择: librosa (默认), demucs, HPSS
    - 去混响开关
    - 输出格式: wav, mp3, flac
    - 输出目录选择，带目录记忆
    - 进度条，带状态文字和耗时显示
    - 完成对话框，带打开文件夹按钮
    - 友好的错误处理
    """

    PAGE_NAME = "声源分离"
    PAGE_ICON = "\U0001f3bc"
    PAGE_DESCRIPTION = "分离音频轨道"

    # Separation modes
    MODES = {
        "2-stem (Vocals + Accompaniment)": "2stems",
        "4-stem (Vocals + Drums + Bass + Other)": "4stems",
        "HPSS (Harmonic + Percussive)": "hpss",
        "Dereverb Only": "dereverb",
    }

    # Backends
    BACKENDS = {
        "librosa": "Lightweight, offline-first (default)",
        "demucs": "High quality deep learning model",
        "HPSS": "Spectral harmonic-percussive separation",
    }

    # Output formats
    OUTPUT_FORMATS = {
        "WAV": ".wav",
        "FLAC": ".flac",
        "MP3": ".mp3",
    }

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """初始化分离页面。

        Bug 1 修复: 所有变量初始化必须在 super().__init__() 之前，
        因为 BasePage.__init__() 会调用 _create_widgets()
        """
        # ============================================================
        # Bug 1 修复：所有变量初始化必须在 super().__init__() 之前
        # ============================================================

        # ---- 设置管理器（单例，线程安全）----
        self._settings = SettingsManager()

        # ---- 取消事件（threading.Event，统一取消机制）----
        self._cancel_event = threading.Event()

        # ---- 处理状态 ----
        self._processing_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # ---- Tkinter 变量（UI <-> Worker 通信）----
        # 源/输出路径
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()

        # 分离参数
        self.separation_mode = tk.StringVar(value="2-stem (人声 + 伴奏)")
        self.backend = tk.StringVar(value="librosa")
        self.dereverb_enabled = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="WAV")

        # 进度显示
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="就绪")
        self.elapsed_var = tk.StringVar(value="")

        # 文件信息变量
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="未选择文件")

        # ---- 目录记忆 ----
        self._last_directory = self._settings.get(
            "separation_last_dir", os.path.expanduser("~")
        )

        # 输出文件
        self._output_files: List[str] = []

        # 输出文件路径（完成后设置）
        self._output_file_path: Optional[str] = None

        # 文件信息加载状态
        self._file_info_loading = False

        # 后端描述标签（延迟引用）
        self._backend_desc_label = None

        # 现在调用 super().__init__()，它会调用 _create_widgets()
        super().__init__(parent, app)
