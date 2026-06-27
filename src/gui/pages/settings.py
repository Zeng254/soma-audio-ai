"""
设置页面 - SOMA GUI

提供应用程序配置界面。
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class SettingsPage(BasePage):
    """
    设置页面 - 应用程序配置。
    
    功能：
    - 默认输出目录
    - 设备选择（CPU/GPU）
    - 缓存目录
    - 关于信息
    """
    
    PAGE_NAME = "设置"
    PAGE_ICON = "⚙️"
    PAGE_DESCRIPTION = "应用程序设置"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """初始化设置页面。"""
        # 设置变量（必须在 super().__init__ 之前初始化）
        self.output_dir = tk.StringVar(value="./output")
        self.cache_dir = tk.StringVar(value="./cache")
        self.device = tk.StringVar(value="auto")
        self.auto_save = tk.BooleanVar(value=True)
        
        super().__init__(parent, app)
    
    def _create_widgets(self):
        """创建设置页面组件。"""
        # 标题区域
        self.create_title_section(
            self.content_frame,
            "设置",
            "配置应用程序首选项"
        )
        
        # 常规设置
        general_card = self.create_card(self.content_frame, "常规设置")
        
        # 输出目录
        row1 = ttk.Frame(general_card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="输出目录：", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.output_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(row1, text="浏览...", style="Secondary.TButton",
                  command=self._browse_output_dir).pack(side=tk.RIGHT)
        
        # 缓存目录
        row2 = ttk.Frame(general_card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="缓存目录：", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.cache_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(row2, text="浏览...", style="Secondary.TButton",
                  command=self._browse_cache_dir).pack(side=tk.RIGHT)
        
        # 自动保存复选框
        row3 = ttk.Frame(general_card, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row3, text="自动保存输出文件",
                       variable=self.auto_save).pack(anchor=tk.W)
        
        # 设备设置
        device_card = self.create_card(self.content_frame, "计算设备")
        
        # 设备选择
        row = ttk.Frame(device_card, style="Card.TFrame")
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="计算设备：", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        
        device_combo = ttk.Combobox(row, textvariable=self.device,
                                   values=["auto", "cpu", "cuda"],
                                   state="readonly")
        device_combo.pack(side=tk.LEFT)
        
        # 设备信息
        info_label = ttk.Label(device_card,
                              text=self._get_device_info(),
                              style="Muted.TLabel")
        info_label.pack(anchor=tk.W, pady=(10, 0))
        
        # 保存按钮
        button_frame = ttk.Frame(self.content_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="💾 保存设置",
                  style="Primary.TButton",
                  command=self._save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="恢复默认",
                  style="Secondary.TButton",
                  command=self._reset_defaults).pack(side=tk.LEFT)
        
        # 关于区域
        about_card = self.create_card(self.content_frame, "关于")
        
        about_text = """SOMA AI - 翻唱工作站
版本：1.0.0

一款 AI 驱动的音频处理工作站，支持声音克隆、
歌曲翻唱生成和音频声源分离。

基于 Python + tkinter 构建，支持离线运行。

© 2024 SOMA AI 团队"""
        
        about_label = ttk.Label(about_card, text=about_text,
                               style="Card.TLabel",
                               justify=tk.LEFT)
        about_label.pack(anchor=tk.W)
    
    def _browse_output_dir(self):
        """浏览输出目录。"""
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            self.output_dir.set(folder)
    
    def _browse_cache_dir(self):
        """浏览缓存目录。"""
        folder = filedialog.askdirectory(title="选择缓存目录")
        if folder:
            self.cache_dir.set(folder)
    
    def _get_device_info(self) -> str:
        """获取当前设备信息。"""
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA 可用：{torch.cuda.get_device_name(0)}"
            return "CUDA 不可用 - 使用 CPU"
        except ImportError:
            return "PyTorch 未安装"
    
    def _save_settings(self):
        """保存设置到配置文件。"""
        from tkinter import messagebox
        # 实际会保存到配置文件
        messagebox.showinfo("提示", "设置已保存！（即将推出）")
    
    def _reset_defaults(self):
        """恢复默认设置。"""
        self.output_dir.set("./output")
        self.cache_dir.set("./cache")
        self.device.set("auto")
        self.auto_save.set(True)
