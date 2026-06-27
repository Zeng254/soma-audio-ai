"""
Settings page for SOMA GUI.

Provides interface for application configuration.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class SettingsPage(BasePage):
    """
    Settings page for application configuration.
    
    Features:
    - Default output directory
    - Device selection (CPU/GPU)
    - Cache directory
    - About information
    """
    
    PAGE_NAME = "Settings"
    PAGE_ICON = "⚙️"
    PAGE_DESCRIPTION = "Application settings"
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the settings page."""
        super().__init__(parent, app)
        
        # Settings variables
        self.output_dir = tk.StringVar(value="./output")
        self.cache_dir = tk.StringVar(value="./cache")
        self.device = tk.StringVar(value="auto")
        self.auto_save = tk.BooleanVar(value=True)
    
    def _create_widgets(self):
        """Create settings page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Settings",
            "Configure application preferences"
        )
        
        # General settings
        general_card = self.create_card(self.content_frame, "General")
        
        # Output directory
        row1 = ttk.Frame(general_card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Output Directory:", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.output_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(row1, text="Browse...", style="Secondary.TButton",
                  command=self._browse_output_dir).pack(side=tk.RIGHT)
        
        # Cache directory
        row2 = ttk.Frame(general_card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Cache Directory:", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.cache_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(row2, text="Browse...", style="Secondary.TButton",
                  command=self._browse_cache_dir).pack(side=tk.RIGHT)
        
        # Auto-save checkbox
        row3 = ttk.Frame(general_card, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(row3, text="Auto-save outputs",
                       variable=self.auto_save).pack(anchor=tk.W)
        
        # Device settings
        device_card = self.create_card(self.content_frame, "Device")
        
        # Device selection
        row = ttk.Frame(device_card, style="Card.TFrame")
        row.pack(fill=tk.X, pady=5)
        ttk.Label(row, text="Compute Device:", style="Card.TLabel",
                 width=20).pack(side=tk.LEFT)
        
        device_combo = ttk.Combobox(row, textvariable=self.device,
                                   values=["auto", "cpu", "cuda"],
                                   state="readonly")
        device_combo.pack(side=tk.LEFT)
        
        # Device info
        info_label = ttk.Label(device_card,
                              text=self._get_device_info(),
                              style="Muted.TLabel")
        info_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Save button
        button_frame = ttk.Frame(self.content_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="💾 Save Settings",
                  style="Primary.TButton",
                  command=self._save_settings).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Reset to Defaults",
                  style="Secondary.TButton",
                  command=self._reset_defaults).pack(side=tk.LEFT)
        
        # About section
        about_card = self.create_card(self.content_frame, "About")
        
        about_text = """SOMA AI - Cover Workstation
Version: 1.0.0

An AI-powered audio processing workstation for voice cloning,
song cover generation, and audio source separation.

Built with Python + tkinter for offline-first operation.

© 2024 SOMA AI Team"""
        
        about_label = ttk.Label(about_card, text=about_text,
                               style="Card.TLabel",
                               justify=tk.LEFT)
        about_label.pack(anchor=tk.W)
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        folder = filedialog.askdirectory(title="Select Output Directory")
        if folder:
            self.output_dir.set(folder)
    
    def _browse_cache_dir(self):
        """Browse for cache directory."""
        folder = filedialog.askdirectory(title="Select Cache Directory")
        if folder:
            self.cache_dir.set(folder)
    
    def _get_device_info(self) -> str:
        """Get current device information."""
        try:
            import torch
            if torch.cuda.is_available():
                return f"CUDA available: {torch.cuda.get_device_name(0)}"
            return "CUDA not available - using CPU"
        except ImportError:
            return "PyTorch not installed"
    
    def _save_settings(self):
        """Save settings to configuration file."""
        from tkinter import messagebox
        # Would save to actual config file
        messagebox.showinfo("Info", "Settings saved! (Coming soon)")
    
    def _reset_defaults(self):
        """Reset settings to defaults."""
        self.output_dir.set("./output")
        self.cache_dir.set("./cache")
        self.device.set("auto")
        self.auto_save.set(True)
