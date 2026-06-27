"""
SOMA GUI 主应用窗口。

提供带导航和页面管理的主窗口。
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Type
from gui.styles import Theme, Colors, Fonts
from gui.widgets.navigation import NavigationSidebar, NavItem, create_default_nav_items
from gui.pages.base import BasePage
from gui.pages.dashboard import DashboardPage
from gui.pages.training import TrainingPage
from gui.pages.inference import InferencePage
from gui.pages.separation import SeparationPage
from gui.pages.comparison import ComparisonPage
from gui.pages.models import ModelsPage
from gui.pages.settings import SettingsPage


class SOMAApp:
    """
    SOMA AI 主应用窗口。
    
    管理导航、页面切换和应用程序状态。
    """
    
    # 页面注册表
    PAGES: Dict[str, Type[BasePage]] = {
        "dashboard": DashboardPage,
        "training": TrainingPage,
        "inference": InferencePage,
        "separation": SeparationPage,
        "comparison": ComparisonPage,
        "models": ModelsPage,
        "settings": SettingsPage,
    }
    
    def __init__(self):
        """初始化应用程序。"""
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("SOMA AI - 翻唱工作站")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # 应用主题
        self.theme = Theme(self.root)
        
        # 页面实例
        self._pages: Dict[str, BasePage] = {}
        self._current_page: Optional[str] = None
        
        # 构建界面
        self._create_layout()
        self._create_pages()
        
        # 显示初始页面
        self.navigate_to("dashboard")
    
    def _create_layout(self):
        """创建主布局结构。"""
        # 主容器
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # 左侧边栏 - 导航
        self.nav_sidebar = NavigationSidebar(
            self.main_container,
            on_navigate=self.navigate_to
        )
        self.nav_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # 添加导航项
        for item in create_default_nav_items():
            self.nav_sidebar.add_item(item, is_active=(item.page_key == "dashboard"))
        
        # 分隔线
        sep = tk.Frame(self.main_container, width=1, bg=Colors.BORDER)
        sep.pack(side=tk.LEFT, fill=tk.Y)
        
        # 右侧内容区域
        self.content_area = ttk.Frame(self.main_container, style="TFrame")
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def _create_pages(self):
        """创建所有页面实例。"""
        for page_key, page_class in self.PAGES.items():
            page = page_class(self.content_area, app=self)
            self._pages[page_key] = page
    
    def navigate_to(self, page_key: str):
        """
        导航到指定页面。
        
        Args:
            page_key: 要导航到的页面键
        """
        if page_key not in self._pages:
            print(f"未知页面: {page_key}")
            return
        
        # 隐藏当前页面
        if self._current_page and self._current_page in self._pages:
            current = self._pages[self._current_page]
            current.on_hide()
            current.pack_forget()
        
        # 显示新页面
        self._current_page = page_key
        new_page = self._pages[page_key]
        new_page.pack(fill=tk.BOTH, expand=True)
        new_page.on_show()
        
        # 更新导航
        self.nav_sidebar._set_active(page_key)
    
    def run(self):
        """启动应用程序主循环。"""
        self.root.mainloop()

    def quit(self):
        """退出应用程序，清理所有页面资源。"""
        # 清理所有页面（关闭线程池等）
        for page_key, page in self._pages.items():
            try:
                page.cleanup()
            except Exception:
                pass
        self.root.quit()


def main():
    """GUI 应用程序入口。"""
    app = SOMAApp()
    app.run()


if __name__ == "__main__":
    main()
