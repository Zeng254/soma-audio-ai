"""
Inference page for SOMA GUI.

Provides interface for AI cover generation (voice conversion).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from typing import Optional, List
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class InferencePage(BasePage):
    """
    Inference page for generating AI covers.
    
    Features:
    - Source audio selection
    - Model selection
    - Pitch shift control
    - Quality options
    - Progress display
    """
    
    PAGE_NAME = "Song Cover"
    PAGE_ICON = "🎵"
    PAGE_DESCRIPTION = "Generate covers"
    
    # Quality presets
    QUALITY_PRESETS = {
        "Standard": {"sample_rate": 44100, "description": "Good quality, faster"},
        "High": {"sample_rate": 48000, "description": "Better quality, moderate speed"},
        "Best": {"sample_rate": 48000, "description": "Highest quality, slower"},
    }
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the inference page."""
        super().__init__(parent, app)
        
        # State
        self._is_processing = False
        self._processing_thread: Optional[threading.Thread] = None
        
        # Variables
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.quality = tk.StringVar(value="High")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        
        # Available models (would be populated from actual model storage)
        self._available_models: List[str] = []
    
    def _create_widgets(self):
        """Create inference page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Song Cover Generation",
            "Convert audio using trained voice models"
        )
        
        # Main content area with two columns
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._create_source_section(left_frame)
        self._create_model_section(left_frame)
        self._create_options_section(left_frame)
        self._create_output_section(left_frame)
        
        # Right column - Progress
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._create_progress_section(right_frame)
        self._create_preview_section(right_frame)
    
    def _create_source_section(self, parent: tk.Widget):
        """Create source audio selection section."""
        card = self.create_card(parent, "Source Audio")
        
        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="Browse...",
                               style="Secondary.TButton",
                               command=self._browse_source)
        browse_btn.pack(side=tk.RIGHT)
        
        # Help text
        help_label = ttk.Label(card,
                              text="Select audio file (WAV, MP3, FLAC supported)",
                              style="Muted.TLabel")
        help_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_model_section(self, parent: tk.Widget):
        """Create model selection section."""
        card = self.create_card(parent, "Voice Model")
        
        # Model dropdown
        model_frame = ttk.Frame(card, style="Card.TFrame")
        model_frame.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Model:", style="Card.TLabel").pack(side=tk.LEFT)
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.selected_model,
                                  state="readonly")
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Refresh button
        refresh_btn = ttk.Button(model_frame, text="🔄",
                                style="Secondary.TButton",
                                command=self._refresh_models)
        refresh_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Update combobox values
        self._model_combo = model_combo
        self._refresh_models()
    
    def _create_options_section(self, parent: tk.Widget):
        """Create conversion options section."""
        card = self.create_card(parent, "Conversion Options")
        
        # Pitch shift
        pitch_frame = ttk.Frame(card, style="Card.TFrame")
        pitch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pitch_frame, text="Pitch Shift:", style="Card.TLabel",
                 width=15).pack(side=tk.LEFT)
        
        pitch_spinbox = tk.Spinbox(
            pitch_frame,
            from_=-12,
            to=12,
            textvariable=self.pitch_shift,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_PRIMARY,
            buttonbackground=Colors.BG_TERTIARY,
            width=10
        )
        pitch_spinbox.pack(side=tk.LEFT)
        
        ttk.Label(pitch_frame, text="semitones", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0))
        
        # Quality preset
        quality_frame = ttk.Frame(card, style="Card.TFrame")
        quality_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(quality_frame, text="Quality:", style="Card.TLabel",
                 width=15).pack(side=tk.LEFT)
        
        quality_combo = ttk.Combobox(quality_frame, textvariable=self.quality,
                                    values=list(self.QUALITY_PRESETS.keys()),
                                    state="readonly")
        quality_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Quality description
        self.quality_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.quality_desc.pack(anchor=tk.W, pady=(5, 0))
        self._update_quality_desc()
        
        # Bind quality change
        self.quality.trace_add("write", lambda *args: self._update_quality_desc())
        
        # Convert button
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.convert_btn = ttk.Button(button_frame, text="▶ Start Conversion",
                                     style="Primary.TButton",
                                     command=self._start_conversion)
        self.convert_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop",
                                  style="Danger.TButton",
                                  command=self._stop_conversion,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
    
    def _create_output_section(self, parent: tk.Widget):
        """Create output configuration section."""
        card = self.create_card(parent, "Output")
        
        # Output path
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        ttk.Label(path_frame, text="Output:", style="Card.TLabel").pack(side=tk.LEFT)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.output_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        browse_btn = ttk.Button(path_frame, text="Browse...",
                               style="Secondary.TButton",
                               command=self._browse_output)
        browse_btn.pack(side=tk.RIGHT)
    
    def _create_progress_section(self, parent: tk.Widget):
        """Create progress display section."""
        card = self.create_card(parent, "Conversion Progress")
        
        # Progress bar
        progress_frame = ttk.Frame(card, style="Card.TFrame")
        progress_frame.pack(fill=tk.X)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:", style="Card.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     style="Card.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Time estimate
        self.time_label = ttk.Label(card, text="Estimated time: N/A",
                                   style="Muted.TLabel")
        self.time_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_preview_section(self, parent: tk.Widget):
        """Create preview/info section."""
        card = self.create_card(parent, "Source Info")
        
        # Audio info display
        info_frame = ttk.Frame(card, style="Card.TFrame")
        info_frame.pack(fill=tk.X)
        
        self.info_labels = {}
        info_items = [
            ("Duration", "N/A"),
            ("Sample Rate", "N/A"),
            ("Channels", "N/A"),
            ("File Size", "N/A"),
        ]
        
        for label, value in info_items:
            row = ttk.Frame(info_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=f"{label}:", style="Card.TLabel", width=15).pack(side=tk.LEFT)
            value_label = ttk.Label(row, text=value, style="Muted.TLabel")
            value_label.pack(side=tk.LEFT)
            self.info_labels[label] = value_label
    
    def _browse_source(self):
        """Open file dialog to select source audio."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Source Audio",
            filetypes=filetypes
        )
        
        if filename:
            self.source_path.set(filename)
            self._update_source_info(filename)
            
            # Auto-generate output path
            import os
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_cover{ext}")
    
    def _browse_output(self):
        """Open file dialog to select output path."""
        filetypes = [
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Output Audio",
            filetypes=filetypes,
            defaultextension=".wav"
        )
        
        if filename:
            self.output_path.set(filename)
    
    def _refresh_models(self):
        """Refresh the list of available models."""
        # Would query actual model storage
        self._available_models = ["No models available"]
        self._model_combo.configure(values=self._available_models)
        
        if self._available_models:
            self._model_combo.set(self._available_models[0])
    
    def _update_quality_desc(self):
        """Update quality description label."""
        quality = self.quality.get()
        if quality in self.QUALITY_PRESETS:
            desc = self.QUALITY_PRESETS[quality]["description"]
            self.quality_desc.configure(text=desc)
    
    def _update_source_info(self, filepath: str):
        """Update source audio information display."""
        # Would read actual audio file info
        self.info_labels["Duration"].configure(text="Loading...")
        self.info_labels["Sample Rate"].configure(text="Loading...")
        self.info_labels["Channels"].configure(text="Loading...")
        self.info_labels["File Size"].configure(text="Loading...")
        
        # Simulate loading
        def _load_info():
            import os
            import time
            time.sleep(0.5)  # Simulate work
            
            try:
                size = os.path.getsize(filepath)
                size_str = f"{size / 1024 / 1024:.2f} MB"
                
                self.after(0, lambda: self.info_labels["File Size"].configure(text=size_str))
                self.after(0, lambda: self.info_labels["Duration"].configure(text="3:45"))
                self.after(0, lambda: self.info_labels["Sample Rate"].configure(text="44100 Hz"))
                self.after(0, lambda: self.info_labels["Channels"].configure(text="Stereo"))
            except Exception:
                self.after(0, lambda: self.info_labels["File Size"].configure(text="Error"))
        
        threading.Thread(target=_load_info, daemon=True).start()
    
    def _start_conversion(self):
        """Start the conversion process."""
        # Validate inputs
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file.")
            return
        
        if not self.selected_model.get() or self.selected_model.get() == "No models available":
            messagebox.showwarning("Warning", "Please select a voice model.")
            return
        
        if not self.output_path.get():
            messagebox.showwarning("Warning", "Please specify an output path.")
            return
        
        # Update UI state
        self._is_processing = True
        self.convert_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Processing...")
        
        # Start processing in background
        self._processing_thread = threading.Thread(target=self._conversion_worker, daemon=True)
        self._processing_thread.start()
    
    def _stop_conversion(self):
        """Stop the conversion process."""
        self._is_processing = False
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")
    
    def _conversion_worker(self):
        """Background worker for conversion (simulated)."""
        steps = ["Loading model...", "Extracting features...", "Converting voice...",
                "Applying effects...", "Saving output..."]
        
        for i, step in enumerate(steps):
            if not self._is_processing:
                break
            
            self.after(0, lambda s=step: self.status_var.set(s))
            
            # Simulate work
            for j in range(20):
                if not self._is_processing:
                    return
                progress = ((i * 20 + j) / (len(steps) * 20)) * 100
                self.after(0, lambda p=progress: self.progress_var.set(p))
                threading.Event().wait(0.05)
        
        if self._is_processing:
            self.after(0, self._conversion_complete)
    
    def _conversion_complete(self):
        """Handle conversion completion."""
        self._is_processing = False
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Completed")
        self.progress_var.set(100)
        
        messagebox.showinfo("Success", f"Cover generated successfully!\n\nOutput: {self.output_path.get()}")
