"""
仪表盘页面 - SOMA GUI

提供主屏幕，包含快速操作和最近活动。
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class DashboardPage(BasePage):
    """
    仪表盘页面 - 显示概览和快速操作。
    
    功能：
    - 项目信息和版本
    - 快速操作按钮
    - 最近活动列表
    - 系统状态
    """
    
    PAGE_NAME = "仪表盘"
    PAGE_ICON = "🏠"
    PAGE_DESCRIPTION = "主页和快速操作"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """初始化仪表盘页面。"""
        super().__init__(parent, app)
    
    def _create_widgets(self):
        """创建仪表盘组件。"""
        # 标题区域
        self.create_title_section(
            self.content_frame,
            "欢迎使用 SOMA AI",
            "您的 AI 驱动翻唱工作站"
        )
        
        # 快速操作卡片
        self._create_quick_actions()
        
        # 系统状态卡片
        self._create_system_status()
        
        # 最近活动卡片
        self._create_recent_activity()
    
    def _create_quick_actions(self):
        """创建快速操作按钮区域。"""
        card = self.create_card(self.content_frame, "快速操作")
        
        # 创建按钮网格
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X)
        
        # 快速操作按钮
        actions = [
            ("🎤", "开始训练", "训练新的声音模型", self._on_start_training),
            ("🎵", "制作翻唱", "生成 AI 翻唱", self._on_create_cover),
            ("🎼", "分离音频", "分离人声和伴奏", self._on_separate_audio),
            ("📁", "管理模型", "查看和管理模型", self._on_manage_models),
        ]
        
        for i, (icon, title, desc, cmd) in enumerate(actions):
            btn_frame = ttk.Frame(button_frame, style="Card.TFrame")
            btn_frame.pack(side=tk.LEFT, padx=(0, 15), fill=tk.X, expand=True)
            
            # 操作按钮
            btn = tk.Button(
                btn_frame,
                text=f"{icon}\n{title}",
                font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"),
                bg=Colors.BG_TERTIARY,
                fg=Colors.TEXT_PRIMARY,
                activebackground=Colors.ACCENT_PRIMARY,
                activeforeground=Colors.BG_PRIMARY,
                bd=0,
                padx=20,
                pady=15,
                cursor="hand2",
                command=cmd
            )
            btn.pack(fill=tk.X)
            
            # 描述
            desc_label = ttk.Label(btn_frame, text=desc, style="Muted.TLabel",
                                  wraplength=150)
            desc_label.pack(anchor=tk.W, pady=(5, 0))
    
    def _create_system_status(self):
        """创建系统状态区域。"""
        card = self.create_card(self.content_frame, "系统状态")
        
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        
        # 状态项目
        statuses = [
            ("计算设备", self._get_device_info(), Colors.ACCENT_SUCCESS),
            ("模型数量", f"{self._get_model_count()} 个已训练", Colors.ACCENT_INFO),
            ("存储空间", self._get_storage_info(), Colors.ACCENT_WARNING),
        ]
        
        for label, value, color in statuses:
            row = ttk.Frame(status_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=5)
            
            ttk.Label(row, text=label, style="Card.TLabel", width=15).pack(side=tk.LEFT)
            
            value_label = ttk.Label(row, text=value, style="Card.TLabel")
            value_label.configure(foreground=color)
            value_label.pack(side=tk.LEFT)
    
    def _create_recent_activity(self):
        """创建最近活动区域。"""
        card = self.create_card(self.content_frame, "最近活动")
        
        # 活动列表
        activity_frame = ttk.Frame(card, style="Card.TFrame")
        activity_frame.pack(fill=tk.BOTH, expand=True)
        
        # 示例最近项目（实际会从数据填充）
        recent_items = [
            ("🎤", "模型 'Aria_v2' 训练完成", "2 小时前"),
            ("🎵", "翻唱 'Song_X' 已生成", "昨天"),
            ("🎼", "音频分离完成", "3 天前"),
        ]
        
        if not recent_items:
            ttk.Label(activity_frame, text="暂无最近活动",
                     style="Muted.TLabel").pack(anchor=tk.W, pady=10)
        else:
            for icon, text, time in recent_items:
                row = ttk.Frame(activity_frame, style="Card.TFrame")
                row.pack(fill=tk.X, pady=3)
                
                ttk.Label(row, text=icon, style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 10))
                ttk.Label(row, text=text, style="Card.TLabel").pack(side=tk.LEFT)
                ttk.Label(row, text=time, style="Muted.TLabel").pack(side=tk.RIGHT)
    
    def _get_device_info(self) -> str:
        """获取当前设备信息。"""
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA ({torch.cuda.get_device_name(0)})"
            return "CPU"
        except ImportError:
            return "CPU（torch 未安装）"
    
    def _get_model_count(self) -> int:
        """获取已训练模型数量。"""
        # 实际会查询模型存储
        return 0
    
    def _get_storage_info(self) -> str:
        """获取存储使用信息。"""
        # 实际会计算存储使用量
        return "已使用 0 MB"
    
    def _on_start_training(self):
        """导航到训练页面。"""
        if self.app:
            self.app.navigate_to("training")
    
    def _on_create_cover(self):
        """导航到推理页面。"""
        if self.app:
            self.app.navigate_to("inference")
    
    def _on_separate_audio(self):
        """导航到分离页面。"""
        if self.app:
            self.app.navigate_to("separation")
    
    def _on_manage_models(self):
        """导航到模型管理页面。"""
        if self.app:
            self.app.navigate_to("models")
    
    def _on_visible(self):
        """页面变为可见时调用 - 刷新数据。"""
        # 刷新状态信息
        pass
