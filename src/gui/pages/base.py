"""
Base page class for all GUI pages.

Provides common functionality and consistent interface for page implementations.
"""

import tkinter as tk
from tkinter import ttk
from abc import ABC, abstractmethod
from typing import Optional, Callable


class BasePage(ttk.Frame, ABC):
    """
    Abstract base class for all pages in the application.
    
    Each page should inherit from this class and implement:
    - _create_widgets(): Create all UI elements
    - on_show(): Called when page becomes visible
    - on_hide(): Called when page is hidden
    """
    
    # Class attributes to be overridden by subclasses
    PAGE_NAME: str = "Page"
    PAGE_ICON: str = "📄"
    PAGE_DESCRIPTION: str = ""
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """
        Initialize the base page.
        
        Args:
            parent: Parent widget
            app: Reference to the main application (optional)
        """
        super().__init__(parent, style="TFrame")
        self.app = app
        self._is_visible = False
        
        # Create a scrollable content area
        self._create_scrollable_area()
        
        # Let subclasses create their widgets
        self._create_widgets()
    
    def _create_scrollable_area(self):
        """Create a scrollable container for page content."""
        # Main container
        self.content_frame = ttk.Frame(self, style="TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    @abstractmethod
    def _create_widgets(self):
        """Create all widgets for this page. Must be implemented by subclasses."""
        pass
    
    def on_show(self):
        """Called when this page becomes visible."""
        self._is_visible = True
        self._on_visible()
    
    def on_hide(self):
        """Called when this page is hidden."""
        self._is_visible = False
        self._on_hidden()
    
    def _on_visible(self):
        """Hook for subclasses when page becomes visible."""
        pass
    
    def _on_hidden(self):
        """Hook for subclasses when page is hidden."""
        pass
    
    def is_visible(self) -> bool:
        """Check if this page is currently visible."""
        return self._is_visible
    
    def create_title_section(self, parent: tk.Widget, title: str, subtitle: str = "") -> ttk.Frame:
        """
        Create a standard title section for the page.
        
        Args:
            parent: Parent widget
            title: Page title
            subtitle: Optional subtitle/description
            
        Returns:
            The created frame containing the title
        """
        frame = ttk.Frame(parent, style="TFrame")
        frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        title_label = ttk.Label(frame, text=title, style="Title.TLabel")
        title_label.pack(anchor=tk.W)
        
        # Subtitle
        if subtitle:
            subtitle_label = ttk.Label(frame, text=subtitle, style="Muted.TLabel")
            subtitle_label.pack(anchor=tk.W, pady=(5, 0))
        
        return frame
    
    def create_card(self, parent: tk.Widget, title: str = "") -> ttk.Frame:
        """
        Create a card-style container.
        
        Args:
            parent: Parent widget
            title: Optional card title
            
        Returns:
            The created card frame
        """
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=tk.X, pady=(0, 15))
        
        # Add padding inside card
        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        if title:
            title_label = ttk.Label(inner, text=title, style="Heading.TLabel")
            title_label.pack(anchor=tk.W, pady=(0, 10))
        
        return inner
    
    def create_form_row(self, parent: tk.Widget, label: str, widget: tk.Widget) -> ttk.Frame:
        """
        Create a form row with label and widget.
        
        Args:
            parent: Parent widget
            label: Label text
            widget: The input widget
            
        Returns:
            The created row frame
        """
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=5)
        
        label_widget = ttk.Label(row, text=label, style="Card.TLabel", width=20)
        label_widget.pack(side=tk.LEFT, padx=(0, 10))
        
        widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        return row
    
    def create_button_row(self, parent: tk.Widget, buttons: list) -> ttk.Frame:
        """
        Create a row of buttons.
        
        Args:
            parent: Parent widget
            buttons: List of (text, command, style) tuples
            
        Returns:
            The created button row frame
        """
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=tk.X, pady=(15, 0))
        
        for text, command, btn_style in buttons:
            btn = ttk.Button(row, text=text, command=command, style=btn_style)
            btn.pack(side=tk.LEFT, padx=(0, 10))
        
        return row
