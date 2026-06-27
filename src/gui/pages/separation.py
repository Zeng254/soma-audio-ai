"""
Separation page for SOMA GUI.

Provides interface for audio source separation (vocals, instruments, etc.).
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class SeparationPage(BasePage):
    """
    Separation page for audio source separation.
    
    Features:
    - Source audio selection
    - Separation mode selection
    - Progress display
    - Output paths
    """
    
    PAGE_NAME = "Separation"
    PAGE_ICON = "🎼"
    PAGE_DESCRIPTION = "Separate audio tracks"
    
    # Separation modes
    MODES = {
        "Vocals + Accompaniment": "Extract vocals and instrumental",
        "Vocals + Drums + Bass + Other": "4-stem separation",
        "Vocals Only": "Extract only vocals",
    }
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the separation page."""
        super().__init__(parent, app)
        
        # Variables
        self.source_path = tk.StringVar()
        self.separation_mode = tk.StringVar(value="Vocals + Accompaniment")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
    
    def _create_widgets(self):
        """Create separation page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Audio Separation",
            "Separate vocals, instruments, and other audio components"
        )
        
        # Source selection
        source_card = self.create_card(self.content_frame, "Source Audio")
        
        path_frame = ttk.Frame(source_card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        ttk.Entry(path_frame, textvariable=self.source_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(path_frame, text="Browse...", style="Secondary.TButton",
                  command=self._browse_source).pack(side=tk.RIGHT)
        
        # Separation mode
        mode_card = self.create_card(self.content_frame, "Separation Mode")
        
        for mode, desc in self.MODES.items():
            row = ttk.Frame(mode_card, style="Card.TFrame")
            row.pack(fill=tk.X, pady=5)
            
            ttk.Radiobutton(row, text=mode, variable=self.separation_mode,
                           value=mode).pack(side=tk.LEFT)
            ttk.Label(row, text=f"- {desc}", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0))
        
        # Start button
        button_frame = ttk.Frame(self.content_frame, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(button_frame, text="▶ Start Separation",
                  style="Primary.TButton",
                  command=self._start_separation).pack(side=tk.LEFT)
        
        # Progress
        progress_card = self.create_card(self.content_frame, "Progress")
        
        ttk.Progressbar(progress_card, variable=self.progress_var,
                       maximum=100).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(progress_card, textvariable=self.status_var,
                 style="Card.TLabel").pack(anchor=tk.W)
    
    def _browse_source(self):
        """Browse for source audio file."""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac")]
        )
        if filename:
            self.source_path.set(filename)
    
    def _start_separation(self):
        """Start the separation process."""
        if not self.source_path.get():
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please select a source audio file.")
            return
        
        self.status_var.set("Separating... (Coming soon)")
        # TODO: Implement actual separation
