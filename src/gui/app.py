"""
Main application window for SOMA GUI.

Provides the primary window with navigation and page management.
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
    Main SOMA AI application window.
    
    Manages navigation, page switching, and application state.
    """
    
    # Page registry
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
        """Initialize the application."""
        # Create root window
        self.root = tk.Tk()
        self.root.title("SOMA AI - Cover Workstation")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)
        
        # Apply theme
        self.theme = Theme(self.root)
        
        # Page instances
        self._pages: Dict[str, BasePage] = {}
        self._current_page: Optional[str] = None
        
        # Build UI
        self._create_layout()
        self._create_pages()
        
        # Show initial page
        self.navigate_to("dashboard")
    
    def _create_layout(self):
        """Create the main layout structure."""
        # Main container
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left sidebar - Navigation
        self.nav_sidebar = NavigationSidebar(
            self.main_container,
            on_navigate=self.navigate_to
        )
        self.nav_sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        # Add navigation items
        for item in create_default_nav_items():
            self.nav_sidebar.add_item(item, is_active=(item.page_key == "dashboard"))
        
        # Separator
        sep = tk.Frame(self.main_container, width=1, bg=Colors.BORDER)
        sep.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right content area
        self.content_area = ttk.Frame(self.main_container, style="TFrame")
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def _create_pages(self):
        """Create all page instances."""
        for page_key, page_class in self.PAGES.items():
            page = page_class(self.content_area, app=self)
            self._pages[page_key] = page
    
    def navigate_to(self, page_key: str):
        """
        Navigate to the specified page.
        
        Args:
            page_key: Key of the page to navigate to
        """
        if page_key not in self._pages:
            print(f"Unknown page: {page_key}")
            return
        
        # Hide current page
        if self._current_page and self._current_page in self._pages:
            current = self._pages[self._current_page]
            current.on_hide()
            current.pack_forget()
        
        # Show new page
        self._current_page = page_key
        new_page = self._pages[page_key]
        new_page.pack(fill=tk.BOTH, expand=True)
        new_page.on_show()
        
        # Update navigation
        self.nav_sidebar._set_active(page_key)
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
    
    def quit(self):
        """Quit the application."""
        self.root.quit()


def main():
    """Entry point for the GUI application."""
    app = SOMAApp()
    app.run()


if __name__ == "__main__":
    main()
