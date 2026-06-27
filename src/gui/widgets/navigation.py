"""
导航侧边栏组件 - SOMA GUI

提供带图标和标签的垂直导航菜单。
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Callable, Optional
from gui.styles import Colors, Fonts


class NavItem:
    """导航项。"""
    
    def __init__(self, name: str, icon: str, page_key: str, description: str = ""):
        self.name = name
        self.icon = icon
        self.page_key = page_key
        self.description = description


class NavigationSidebar(ttk.Frame):
    """
    垂直导航侧边栏。
    
    点击导航项时发出导航事件。
    """
    
    def __init__(self, parent: tk.Widget, on_navigate: Callable[[str], None]):
        """
        初始化导航侧边栏。
        
        Args:
            parent: 父组件
            on_navigate: 导航项被选中时的回调函数
        """
        super().__init__(parent, style="Nav.TFrame", width=220)
        self.pack_propagate(False)  # 保持固定宽度
        
        self.on_navigate = on_navigate
        self._active_item: Optional[str] = None
        self._buttons: dict = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """创建导航侧边栏组件。"""
        # 标题区域
        header = ttk.Frame(self, style="Nav.TFrame")
        header.pack(fill=tk.X, padx=15, pady=(20, 30))
        
        logo_label = ttk.Label(header, text="🎵", font=("Segoe UI Emoji", 28))
        logo_label.pack(anchor=tk.W)
        
        title_label = ttk.Label(header, text="SOMA AI", 
                               font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
                               foreground=Colors.TEXT_PRIMARY)
        title_label.pack(anchor=tk.W, pady=(5, 0))
        
        version_label = ttk.Label(header, text="翻唱工作站 v1.0",
                                 font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                                 foreground=Colors.TEXT_MUTED)
        version_label.pack(anchor=tk.W)
        
        # 分隔线
        sep = tk.Frame(self, height=1, bg=Colors.BORDER)
        sep.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # 导航项容器
        self.nav_container = ttk.Frame(self, style="Nav.TFrame")
        self.nav_container.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # 底部状态
        footer = ttk.Frame(self, style="Nav.TFrame")
        footer.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        status_label = ttk.Label(footer, text="● 就绪",
                                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                                foreground=Colors.ACCENT_SUCCESS)
        status_label.pack(anchor=tk.W)
    
    def add_item(self, item: NavItem, is_active: bool = False):
        """
        添加导航项。
        
        Args:
            item: 要添加的导航项
            is_active: 是否初始激活此项
        """
        btn_frame = ttk.Frame(self.nav_container, style="Nav.TFrame")
        btn_frame.pack(fill=tk.X, pady=2)
        
        # 创建带图标和文字的按钮
        btn = tk.Button(
            btn_frame,
            text=f"  {item.icon}  {item.name}",
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.NAV_BG,
            fg=Colors.ACCENT_PRIMARY if is_active else Colors.TEXT_SECONDARY,
            activebackground=Colors.NAV_HOVER,
            activeforeground=Colors.ACCENT_PRIMARY,
            bd=0,
            padx=15,
            pady=10,
            anchor=tk.W,
            cursor="hand2",
            command=lambda: self._on_item_click(item.page_key)
        )
        btn.pack(fill=tk.X)
        
        # 悬停效果
        btn.bind("<Enter>", lambda e: self._on_hover(btn, True))
        btn.bind("<Leave>", lambda e: self._on_hover(btn, False))
        
        self._buttons[item.page_key] = btn
        
        if is_active:
            self._set_active(item.page_key)
    
    def _on_item_click(self, page_key: str):
        """处理导航项点击。"""
        if page_key != self._active_item:
            self._set_active(page_key)
            self.on_navigate(page_key)
    
    def _set_active(self, page_key: str):
        """设置激活的导航项。"""
        # 重置之前的激活项
        if self._active_item and self._active_item in self._buttons:
            prev_btn = self._buttons[self._active_item]
            prev_btn.configure(
                bg=Colors.NAV_BG,
                fg=Colors.TEXT_SECONDARY
            )
        
        # 设置新的激活项
        if page_key in self._buttons:
            btn = self._buttons[page_key]
            btn.configure(
                bg=Colors.NAV_HOVER,
                fg=Colors.ACCENT_PRIMARY
            )
            self._active_item = page_key
    
    def _on_hover(self, btn: tk.Button, entering: bool):
        """处理按钮悬停效果。"""
        page_key = None
        for key, b in self._buttons.items():
            if b == btn:
                page_key = key
                break
        
        if page_key == self._active_item:
            return  # 悬停时不改变激活项外观
        
        if entering:
            btn.configure(bg=Colors.NAV_HOVER)
        else:
            btn.configure(bg=Colors.NAV_BG)
    
    def get_active_item(self) -> Optional[str]:
        """获取当前激活的页面键。"""
        return self._active_item
    
    def set_status(self, status: str, color: str = None):
        """
        更新底部状态指示器。
        
        Args:
            status: 状态文字
            color: 可选颜色（默认为成功绿色）
        """
        # 查找并更新状态标签
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        if "就绪" in child.cget("text") or "●" in child.cget("text"):
                            child.configure(
                                text=f"● {status}",
                                foreground=color or Colors.ACCENT_SUCCESS
                            )
                            return


def create_default_nav_items() -> List[NavItem]:
    """创建应用程序的默认导航项。"""
    return [
        NavItem("仪表盘", "🏠", "dashboard", "主页和快速操作"),
        NavItem("声音克隆", "🎤", "training", "训练声音模型"),
        NavItem("歌曲翻唱", "🎵", "inference", "生成 AI 翻唱"),
        NavItem("声源分离", "🎼", "separation", "分离音频轨道"),
        NavItem("效果对比", "🔀", "comparison", "对比转换结果"),
        NavItem("模型管理", "📁", "models", "管理已训练模型"),
        NavItem("设置", "⚙️", "settings", "应用程序设置"),
    ]
