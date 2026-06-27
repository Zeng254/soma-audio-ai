"""
Separation page for SOMA GUI.

Provides interface for audio source separation (vocals, instruments, etc.).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
from typing import Optional, List
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


class SeparationPage(BasePage):
    """
    Separation page for audio source separation.
    
    Features:
    - Source audio selection
    - Separation mode selection (2-stem, 4-stem, HPSS, dereverb)
    - Backend selection (librosa, demucs, msst)
    - Output directory selection
    - Progress display with real-time log output
    """
    
    PAGE_NAME = "Separation"
    PAGE_ICON = "🎼"
    PAGE_DESCRIPTION = "Separate audio tracks"
    
    # Separation modes
    MODES = {
        "2-stem (Vocals + Accompaniment)": "2stems",
        "4-stem (Vocals + Drums + Bass + Other)": "4stems",
        "Dereverb Only": "dereverb",
        "HPSS (Harmonic + Percussive)": "hpss",
    }
    
    # Backends
    BACKENDS = {
        "librosa": "Lightweight, offline-first",
        "demucs": "High quality, requires demucs package",
        "msst": "MSST models (coming soon)",
    }
    
    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the separation page."""
        super().__init__(parent, app)
        
        # State
        self._is_processing = False
        self._processing_thread: Optional[threading.Thread] = None
        
        # Variables
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.separation_mode = tk.StringVar(value="2-stem (Vocals + Accompaniment)")
        self.backend = tk.StringVar(value="librosa")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.log_text = tk.StringVar(value="")
        
        # Output files
        self._output_files: List[str] = []
    
    def _create_widgets(self):
        """Create separation page widgets."""
        # Title section
        self.create_title_section(
            self.content_frame,
            "Audio Separation",
            "Separate vocals, instruments, and other audio components"
        )
        
        # Main content area with two columns
        main_frame = ttk.Frame(self.content_frame, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Configuration
        left_frame = ttk.Frame(main_frame, style="TFrame")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self._create_source_section(left_frame)
        self._create_mode_section(left_frame)
        self._create_output_section(left_frame)
        self._create_action_section(left_frame)
        
        # Right column - Progress and Log
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)
        self._create_output_section(right_frame)
    
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
    
    def _create_mode_section(self, parent: tk.Widget):
        """Create separation mode selection section."""
        card = self.create_card(parent, "Separation Settings")
        
        # Mode selection
        mode_frame = ttk.Frame(card, style="Card.TFrame")
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Mode:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.separation_mode,
                                  values=list(self.MODES.keys()),
                                  state="readonly")
        mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Backend selection
        backend_frame = ttk.Frame(card, style="Card.TFrame")
        backend_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(backend_frame, text="Backend:", style="Card.TLabel", width=10).pack(side=tk.LEFT)
        
        backend_combo = ttk.Combobox(backend_frame, textvariable=self.backend,
                                     values=list(self.BACKENDS.keys()),
                                     state="readonly")
        backend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Backend description
        self.backend_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.backend_desc.pack(anchor=tk.W, pady=(5, 0))
        self._update_backend_desc()
        
        # Bind backend change
        self.backend.trace_add("write", lambda *args: self._update_backend_desc())
    
    def _create_output_section(self, parent: tk.Widget):
        """Create output directory selection section."""
        card = self.create_card(parent, "Output Directory")
        
        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.output_dir)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = ttk.Button(path_frame, text="Browse...",
                               style="Secondary.TButton",
                               command=self._browse_output_dir)
        browse_btn.pack(side=tk.RIGHT)
        
        # Help text
        help_label = ttk.Label(card,
                              text="Output files will be saved here",
                              style="Muted.TLabel")
        help_label.pack(anchor=tk.W, pady=(10, 0))
    
    def _create_action_section(self, parent: tk.Widget):
        """Create action buttons section."""
        button_frame = ttk.Frame(parent, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start Separation",
                                   style="Primary.TButton",
                                   command=self._start_separation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop",
                                  style="Danger.TButton",
                                  command=self._stop_separation,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
    
    def _create_progress_section(self, parent: tk.Widget):
        """Create progress display section."""
        card = self.create_card(parent, "Progress")
        
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
    
    def _create_log_section(self, parent: tk.Widget):
        """Create log output section."""
        card = self.create_card(parent, "Log Output")
        
        # Log text area
        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_display = tk.Text(
            log_frame,
            height=10,
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_SMALL),
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)
    
    def _create_output_section(self, parent: tk.Widget):
        """Create output files section."""
        card = self.create_card(parent, "Output Files")
        
        # Output files list
        self.output_listbox = tk.Listbox(
            card,
            height=5,
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            selectbackground=Colors.PRIMARY,
            selectforeground=Colors.TEXT_ON_PRIMARY
        )
        self.output_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Open folder button
        open_btn = ttk.Button(card, text="📁 Open Output Folder",
                             style="Secondary.TButton",
                             command=self._open_output_folder)
        open_btn.pack(anchor=tk.W)
    
    def _browse_source(self):
        """Browse for source audio file."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            self.source_path.set(filename)
            
            # Auto-set output directory
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(filename))
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
    
    def _update_backend_desc(self):
        """Update backend description label."""
        backend = self.backend.get()
        if backend in self.BACKENDS:
            desc = self.BACKENDS[backend]
            self.backend_desc.configure(text=desc)
    
    def _log(self, message: str):
        """Add message to log display."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)
    
    def _start_separation(self):
        """Start the separation process."""
        # Validate inputs
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file.")
            return
        
        if not os.path.exists(self.source_path.get()):
            messagebox.showerror("Error", "Source file does not exist.")
            return
        
        if not self.output_dir.get():
            messagebox.showwarning("Warning", "Please specify an output directory.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir.get(), exist_ok=True)
        
        # Update UI state
        self._is_processing = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Starting...")
        self.progress_var.set(0)
        self._output_files = []
        self.output_listbox.delete(0, tk.END)
        
        # Clear log
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)
        self.log_display.configure(state=tk.DISABLED)
        
        self._log(f"Starting separation...")
        self._log(f"Source: {self.source_path.get()}")
        self._log(f"Mode: {self.separation_mode.get()}")
        self._log(f"Backend: {self.backend.get()}")
        self._log(f"Output: {self.output_dir.get()}")
        self._log("")
        
        # Start processing in background
        self._processing_thread = threading.Thread(target=self._separation_worker, daemon=True)
        self._processing_thread.start()
    
    def _stop_separation(self):
        """Stop the separation process."""
        self._is_processing = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self._log("Separation stopped by user.")
    
    def _separation_worker(self):
        """Background worker for separation."""
        try:
            # Import here to avoid blocking UI on import
            from separators.audio_separator import AudioSeparator, SeparationMode, SeparationBackend
            
            # Map UI values to enum values
            mode_map = {
                "2-stem (Vocals + Accompaniment)": SeparationMode.TWO_STEMS,
                "4-stem (Vocals + Drums + Bass + Other)": SeparationMode.FOUR_STEMS,
                "Dereverb Only": "dereverb",
                "HPSS (Harmonic + Percussive)": SeparationMode.HPSS,
            }
            
            backend_map = {
                "librosa": SeparationBackend.LIBROSA,
                "demucs": SeparationBackend.DEMUCS,
                "msst": SeparationBackend.MSST,
            }
            
            mode = mode_map.get(self.separation_mode.get())
            backend = backend_map.get(self.backend.get())
            
            if mode is None or backend is None:
                raise ValueError(f"Invalid mode or backend selection")
            
            self.after(0, lambda: self._log("Initializing AudioSeparator..."))
            self.after(0, lambda: self.status_var.set("Initializing..."))
            
            # Create separator
            separator = AudioSeparator(backend=backend.value)
            
            # Load audio
            self.after(0, lambda: self._log("Loading audio..."))
            self.after(0, lambda: self.status_var.set("Loading audio..."))
            self.after(0, lambda: self.progress_var.set(10))
            
            import numpy as np
            import soundfile as sf
            
            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)  # Mono to stereo
            
            self.after(0, lambda: self._log(f"Audio loaded: {audio.shape}, {sr}Hz"))
            self.after(0, lambda: self.progress_var.set(20))
            
            # Perform separation
            if mode == "dereverb":
                self.after(0, lambda: self._log("Performing dereverberation..."))
                self.after(0, lambda: self.status_var.set("Dereverberating..."))
                result = separator.dereverb(audio)
                results = [("dereverb", result)]
            else:
                self.after(0, lambda: self._log(f"Performing {mode.value} separation..."))
                self.after(0, lambda: self.status_var.set("Separating..."))
                results = separator.separate(audio, mode=mode)
            
            self.after(0, lambda: self.progress_var.set(80))
            self.after(0, lambda: self._log("Separation complete, saving files..."))
            self.after(0, lambda: self.status_var.set("Saving files..."))
            
            # Save output files
            source_name = os.path.splitext(os.path.basename(self.source_path.get()))[0]
            
            for stem_name, stem_audio in results:
                output_path = os.path.join(
                    self.output_dir.get(),
                    f"{source_name}_{stem_name}.wav"
                )
                sf.write(output_path, stem_audio, sr)
                self._output_files.append(output_path)
                self.after(0, lambda p=output_path: self._log(f"Saved: {p}"))
                self.after(0, lambda p=output_path: self.output_listbox.insert(tk.END, os.path.basename(p)))
            
            self.after(0, lambda: self.progress_var.set(100))
            self.after(0, self._separation_complete)
            
        except ImportError as e:
            self.after(0, lambda: self._log(f"Import error: {e}"))
            self.after(0, lambda: self._log("Please install required packages: pip install librosa soundfile"))
            self.after(0, lambda: self.status_var.set("Error"))
            self.after(0, self._separation_error)
            
        except Exception as e:
            self.after(0, lambda: self._log(f"Error: {e}"))
            self.after(0, lambda: self.status_var.set("Error"))
            self.after(0, self._separation_error)
    
    def _separation_complete(self):
        """Handle separation completion."""
        self._is_processing = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Completed")
        
        self._log("")
        self._log(f"Separation complete! {len(self._output_files)} files saved.")
        
        messagebox.showinfo(
            "Success",
            f"Separation completed successfully!\n\n"
            f"Output files saved to:\n{self.output_dir.get()}"
        )
    
    def _separation_error(self):
        """Handle separation error."""
        self._is_processing = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        
        messagebox.showerror(
            "Error",
            "Separation failed. Check log for details."
        )
    
    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        if self.output_dir.get() and os.path.exists(self.output_dir.get()):
            import subprocess
            import sys
            
            folder = os.path.normpath(self.output_dir.get())
            
            if sys.platform == 'win32':
                subprocess.run(['explorer', folder])
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder])
            else:
                subprocess.run(['xdg-open', folder])
        else:
            messagebox.showwarning("Warning", "Output folder does not exist.")
