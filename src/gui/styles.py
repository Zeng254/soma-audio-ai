"""
Custom styles and theme configuration for SOMA GUI.

Provides a modern, dark-themed UI with consistent styling across all widgets.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


# Color palette - Modern dark theme
class Colors:
    """Color constants for the application theme."""
    # Background colors
    BG_PRIMARY = "#1a1b26"      # Main background
    BG_SECONDARY = "#24283b"    # Sidebar, cards
    BG_TERTIARY = "#414868"     # Hover states
    BG_INPUT = "#1f2335"        # Input fields
    
    # Text colors
    TEXT_PRIMARY = "#c0caf5"     # Primary text
    TEXT_SECONDARY = "#a9b1d6"   # Secondary text
    TEXT_MUTED = "#565f89"       # Muted text
    TEXT_DISABLED = "#3b4261"    # Disabled text
    
    # Accent colors
    ACCENT_PRIMARY = "#7aa2f7"   # Primary accent (blue)
    ACCENT_SUCCESS = "#9ece6a"   # Success (green)
    ACCENT_WARNING = "#e0af68"   # Warning (yellow)
    ACCENT_ERROR = "#f7768e"     # Error (red)
    ACCENT_INFO = "#7dcfff"      # Info (cyan)
    
    # Border colors
    BORDER = "#3b4261"
    BORDER_FOCUS = "#7aa2f7"
    
    # Navigation
    NAV_BG = "#16161e"
    NAV_HOVER = "#292e42"
    NAV_ACTIVE = "#7aa2f7"


# Font configuration
class Fonts:
    """Font constants for the application."""
    FAMILY = "Segoe UI"  # Windows default, falls back gracefully
    FAMILY_MONO = "Consolas"
    
    SIZE_TITLE = 18
    SIZE_HEADING = 14
    SIZE_BODY = 11
    SIZE_SMALL = 10
    SIZE_TINY = 9


class Theme:
    """Theme configuration and style application."""
    
    def __init__(self, root: tk.Tk):
        """Initialize theme for the given root window."""
        self.root = root
        self._setup_styles()
    
    def _setup_styles(self):
        """Configure all ttk styles."""
        style = ttk.Style()
        style.theme_use("clam")  # Base theme for customization
        
        # Configure root window
        self.root.configure(bg=Colors.BG_PRIMARY)
        
        # Configure styles
        self._configure_frame()
        self._configure_label()
        self._configure_button()
        self._configure_entry()
        self._configure_combobox()
        self._configure_progressbar()
        self._configure_notebook()
        self._configure_label_frame()
        self._configure_scrollbar()
    
    def _configure_frame(self):
        """Configure TFrame styles."""
        style = ttk.Style()
        
        style.configure("TFrame", background=Colors.BG_PRIMARY)
        style.configure("Card.TFrame", background=Colors.BG_SECONDARY)
        style.configure("Nav.TFrame", background=Colors.NAV_BG)
        style.configure("Input.TFrame", background=Colors.BG_INPUT)
    
    def _configure_label(self):
        """Configure TLabel styles."""
        style = ttk.Style()
        
        style.configure("TLabel", 
                       background=Colors.BG_PRIMARY,
                       foreground=Colors.TEXT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY))
        
        style.configure("Title.TLabel",
                       background=Colors.BG_PRIMARY,
                       foreground=Colors.TEXT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_TITLE, "bold"))
        
        style.configure("Heading.TLabel",
                       background=Colors.BG_PRIMARY,
                       foreground=Colors.ACCENT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_HEADING, "bold"))
        
        style.configure("Card.TLabel",
                       background=Colors.BG_SECONDARY,
                       foreground=Colors.TEXT_PRIMARY)
        
        style.configure("Nav.TLabel",
                       background=Colors.NAV_BG,
                       foreground=Colors.TEXT_SECONDARY)
        
        style.configure("NavActive.TLabel",
                       background=Colors.NAV_BG,
                       foreground=Colors.ACCENT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"))
        
        style.configure("Muted.TLabel",
                       background=Colors.BG_PRIMARY,
                       foreground=Colors.TEXT_MUTED,
                       font=(Fonts.FAMILY, Fonts.SIZE_SMALL))
    
    def _configure_button(self):
        """Configure TButton styles."""
        style = ttk.Style()
        
        # Primary button
        style.configure("Primary.TButton",
                       background=Colors.ACCENT_PRIMARY,
                       foreground=Colors.BG_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"),
                       padding=(16, 8))
        style.map("Primary.TButton",
                 background=[("active", Colors.ACCENT_INFO),
                            ("disabled", Colors.BG_TERTIARY)])
        
        # Secondary button
        style.configure("Secondary.TButton",
                       background=Colors.BG_TERTIARY,
                       foreground=Colors.TEXT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY),
                       padding=(16, 8))
        style.map("Secondary.TButton",
                 background=[("active", Colors.NAV_HOVER)])
        
        # Danger button
        style.configure("Danger.TButton",
                       background=Colors.ACCENT_ERROR,
                       foreground=Colors.BG_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"),
                       padding=(16, 8))
        style.map("Danger.TButton",
                 background=[("active", "#ff9e64")])
        
        # Nav button
        style.configure("Nav.TButton",
                       background=Colors.NAV_BG,
                       foreground=Colors.TEXT_SECONDARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY),
                       padding=(12, 10),
                       anchor="w")
        style.map("Nav.TButton",
                 background=[("active", Colors.NAV_HOVER)])
    
    def _configure_entry(self):
        """Configure TEntry styles."""
        style = ttk.Style()
        
        style.configure("TEntry",
                       fieldbackground=Colors.BG_INPUT,
                       foreground=Colors.TEXT_PRIMARY,
                       bordercolor=Colors.BORDER,
                       lightcolor=Colors.BORDER,
                       darkcolor=Colors.BORDER,
                       insertcolor=Colors.TEXT_PRIMARY,
                       padding=8)
        style.map("TEntry",
                 bordercolor=[("focus", Colors.BORDER_FOCUS)],
                 lightcolor=[("focus", Colors.BORDER_FOCUS)])
    
    def _configure_combobox(self):
        """Configure TCombobox styles."""
        style = ttk.Style()
        
        style.configure("TCombobox",
                       fieldbackground=Colors.BG_INPUT,
                       background=Colors.BG_SECONDARY,
                       foreground=Colors.TEXT_PRIMARY,
                       bordercolor=Colors.BORDER,
                       lightcolor=Colors.BORDER,
                       darkcolor=Colors.BORDER,
                       arrowcolor=Colors.TEXT_SECONDARY,
                       padding=8)
        style.map("TCombobox",
                 bordercolor=[("focus", Colors.BORDER_FOCUS)])
    
    def _configure_progressbar(self):
        """Configure Progressbar styles."""
        style = ttk.Style()
        
        style.configure("Horizontal.TProgressbar",
                       background=Colors.ACCENT_PRIMARY,
                       troughcolor=Colors.BG_TERTIARY,
                       bordercolor=Colors.BORDER,
                       lightcolor=Colors.ACCENT_PRIMARY,
                       darkcolor=Colors.ACCENT_PRIMARY)
    
    def _configure_notebook(self):
        """Configure TNotebook styles."""
        style = ttk.Style()
        
        style.configure("TNotebook",
                       background=Colors.BG_PRIMARY,
                       borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=Colors.BG_SECONDARY,
                       foreground=Colors.TEXT_SECONDARY,
                       padding=(16, 8),
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY))
        style.map("TNotebook.Tab",
                 background=[("selected", Colors.ACCENT_PRIMARY)],
                 foreground=[("selected", Colors.BG_PRIMARY)])
    
    def _configure_label_frame(self):
        """Configure TLabelFrame styles."""
        style = ttk.Style()
        
        style.configure("TLabelframe",
                       background=Colors.BG_SECONDARY,
                       bordercolor=Colors.BORDER,
                       relief="flat")
        style.configure("TLabelframe.Label",
                       background=Colors.BG_SECONDARY,
                       foreground=Colors.ACCENT_PRIMARY,
                       font=(Fonts.FAMILY, Fonts.SIZE_BODY, "bold"))
    
    def _configure_scrollbar(self):
        """Configure Scrollbar styles."""
        style = ttk.Style()
        
        style.configure("Vertical.TScrollbar",
                       background=Colors.BG_TERTIARY,
                       bordercolor=Colors.BG_PRIMARY,
                       troughcolor=Colors.BG_PRIMARY,
                       arrowcolor=Colors.TEXT_SECONDARY)


def create_separator(parent: tk.Widget, orient: str = "horizontal") -> ttk.Frame:
    """Create a visual separator line."""
    sep = ttk.Frame(parent, style="TFrame")
    if orient == "horizontal":
        sep.configure(height=1)
        sep.configure(style="Card.TFrame")
    else:
        sep.configure(width=1)
        sep.configure(style="Card.TFrame")
    return sep
