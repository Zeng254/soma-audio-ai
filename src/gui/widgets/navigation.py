"""
Navigation sidebar widget for SOMA GUI.

Provides a vertical navigation menu with icons and labels.
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple, Callable, Optional
from gui.styles import Colors, Fonts


class NavItem:
    """Represents a navigation item."""
    
    def __init__(self, name: str, icon: str, page_key: str, description: str = ""):
        self.name = name
        self.icon = icon
        self.page_key = page_key
        self.description = description


class NavigationSidebar(ttk.Frame):
    """
    Vertical navigation sidebar with icon + label items.
    
    Emits navigation events when items are clicked.
    """
    
    def __init__(self, parent: tk.Widget, on_navigate: Callable[[str], None]):
        """
        Initialize the navigation sidebar.
        
        Args:
            parent: Parent widget
            on_navigate: Callback when navigation item is selected
        """
        super().__init__(parent, style="Nav.TFrame", width=220)
        self.pack_propagate(False)  # Maintain fixed width
        
        self.on_navigate = on_navigate
        self._active_item: Optional[str] = None
        self._buttons: dict = {}
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the navigation sidebar widgets."""
        # Header with logo/title
        header = ttk.Frame(self, style="Nav.TFrame")
        header.pack(fill=tk.X, padx=15, pady=(20, 30))
        
        logo_label = ttk.Label(header, text="🎵", font=("Segoe UI Emoji", 28))
        logo_label.pack(anchor=tk.W)
        
        title_label = ttk.Label(header, text="SOMA AI", 
                               font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"),
                               foreground=Colors.TEXT_PRIMARY)
        title_label.pack(anchor=tk.W, pady=(5, 0))
        
        version_label = ttk.Label(header, text="Cover Workstation v1.0",
                                 font=(Fonts.FAMILY, Fonts.SIZE_TINY),
                                 foreground=Colors.TEXT_MUTED)
        version_label.pack(anchor=tk.W)
        
        # Separator
        sep = tk.Frame(self, height=1, bg=Colors.BORDER)
        sep.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # Navigation items container
        self.nav_container = ttk.Frame(self, style="Nav.TFrame")
        self.nav_container.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Footer with status
        footer = ttk.Frame(self, style="Nav.TFrame")
        footer.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        status_label = ttk.Label(footer, text="● Ready",
                                font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
                                foreground=Colors.ACCENT_SUCCESS)
        status_label.pack(anchor=tk.W)
    
    def add_item(self, item: NavItem, is_active: bool = False):
        """
        Add a navigation item.
        
        Args:
            item: Navigation item to add
            is_active: Whether this item should be initially active
        """
        btn_frame = ttk.Frame(self.nav_container, style="Nav.TFrame")
        btn_frame.pack(fill=tk.X, pady=2)
        
        # Create button with icon and text
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
        
        # Hover effects
        btn.bind("<Enter>", lambda e: self._on_hover(btn, True))
        btn.bind("<Leave>", lambda e: self._on_hover(btn, False))
        
        self._buttons[item.page_key] = btn
        
        if is_active:
            self._set_active(item.page_key)
    
    def _on_item_click(self, page_key: str):
        """Handle navigation item click."""
        if page_key != self._active_item:
            self._set_active(page_key)
            self.on_navigate(page_key)
    
    def _set_active(self, page_key: str):
        """Set the active navigation item."""
        # Reset previous active item
        if self._active_item and self._active_item in self._buttons:
            prev_btn = self._buttons[self._active_item]
            prev_btn.configure(
                bg=Colors.NAV_BG,
                fg=Colors.TEXT_SECONDARY
            )
        
        # Set new active item
        if page_key in self._buttons:
            btn = self._buttons[page_key]
            btn.configure(
                bg=Colors.NAV_HOVER,
                fg=Colors.ACCENT_PRIMARY
            )
            self._active_item = page_key
    
    def _on_hover(self, btn: tk.Button, entering: bool):
        """Handle button hover effects."""
        page_key = None
        for key, b in self._buttons.items():
            if b == btn:
                page_key = key
                break
        
        if page_key == self._active_item:
            return  # Don't change active item appearance on hover
        
        if entering:
            btn.configure(bg=Colors.NAV_HOVER)
        else:
            btn.configure(bg=Colors.NAV_BG)
    
    def get_active_item(self) -> Optional[str]:
        """Get the currently active page key."""
        return self._active_item
    
    def set_status(self, status: str, color: str = None):
        """
        Update the status indicator in the footer.
        
        Args:
            status: Status text
            color: Optional color (defaults to success green)
        """
        # Find and update status label
        for widget in self.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        if "Ready" in child.cget("text") or "●" in child.cget("text"):
                            child.configure(
                                text=f"● {status}",
                                foreground=color or Colors.ACCENT_SUCCESS
                            )
                            return


def create_default_nav_items() -> List[NavItem]:
    """Create the default navigation items for the application."""
    return [
        NavItem("Dashboard", "🏠", "dashboard", "Home and quick actions"),
        NavItem("Voice Clone", "🎤", "training", "Train voice models"),
        NavItem("Song Cover", "🎵", "inference", "Generate covers"),
        NavItem("Separation", "🎼", "separation", "Separate audio tracks"),
        NavItem("Models", "📁", "models", "Manage trained models"),
        NavItem("Settings", "⚙️", "settings", "Application settings"),
    ]
