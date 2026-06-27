"""
Separation page for SOMA GUI.

Provides interface for audio source separation (vocals, instruments, etc.).
Enhanced with: parameter controls, progress display, file info, drag-drop zone,
directory memory, auto-naming, completion dialog, and error handling.

Code quality fixes applied:
- SettingsManager singleton for thread-safe settings access
- threading.Event for cancel mechanism (unified with comparison page)
- Widget alive guards on all after() callbacks
- Unified UI reset logic after processing completes/stops/errors
- Common open_folder utility
- File info loading in background thread
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
from typing import Optional, List
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts
from gui.utils import SettingsManager, open_folder, AUDIO_FILETYPES


class SeparationPage(BasePage):
    """
    Separation page for audio source separation.

    Features:
    - Source audio selection with file info display and drop zone
    - Separation mode: 2-stem, 4-stem, HPSS, Dereverb
    - Backend selection: librosa (default), demucs, HPSS
    - Dereverb toggle
    - Output format: wav, mp3, flac
    - Output directory selection with directory memory
    - Progress bar with status text and elapsed time
    - Completion dialog with open folder button
    - Friendly error handling
    """

    PAGE_NAME = "Separation"
    PAGE_ICON = "\U0001f3bc"
    PAGE_DESCRIPTION = "Separate audio tracks"

    # Separation modes
    MODES = {
        "2-stem (Vocals + Accompaniment)": "2stems",
        "4-stem (Vocals + Drums + Bass + Other)": "4stems",
        "HPSS (Harmonic + Percussive)": "hpss",
        "Dereverb Only": "dereverb",
    }

    # Backends
    BACKENDS = {
        "librosa": "Lightweight, offline-first (default)",
        "demucs": "High quality deep learning model",
        "HPSS": "Spectral harmonic-percussive separation",
    }

    # Output formats
    OUTPUT_FORMATS = {
        "WAV": ".wav",
        "FLAC": ".flac",
        "MP3": ".mp3",
    }

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the separation page."""
        super().__init__(parent, app)

        # Settings manager (singleton, thread-safe)
        self._settings = SettingsManager()

        # Cancel event (threading.Event for unified cancel mechanism)
        self._cancel_event = threading.Event()

        # State
        self._processing_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # Variables
        self.source_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.separation_mode = tk.StringVar(value="2-stem (Vocals + Accompaniment)")
        self.backend = tk.StringVar(value="librosa")
        self.dereverb_enabled = tk.BooleanVar(value=False)
        self.output_format = tk.StringVar(value="WAV")
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="")

        # File info variables
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="No file selected")

        # Remembered last directory (from SettingsManager)
        self._last_directory = self._settings.get(
            "separation_last_dir", os.path.expanduser("~")
        )

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

        self._create_dropzone_section(left_frame)
        self._create_file_info_section(left_frame)
        self._create_mode_section(left_frame)
        self._create_output_section(left_frame)
        self._create_action_section(left_frame)

        # Right column - Progress and Log
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)
        self._create_output_files_section(right_frame)

    # ── Drop Zone ──────────────────────────────────────────────────────

    def _create_dropzone_section(self, parent: tk.Widget):
        """Create a visual drop zone for audio files."""
        card = self.create_card(parent, "Source Audio")

        # Drop zone frame with dashed border effect
        self.dropzone = tk.Frame(
            card,
            bg=Colors.BG_INPUT,
            highlightbackground=Colors.BORDER,
            highlightthickness=2,
            padx=20,
            pady=20,
        )
        self.dropzone.pack(fill=tk.X, pady=(0, 10))

        # Drop zone icon and text
        icon_label = tk.Label(
            self.dropzone,
            text="\U0001f4c1",  # folder emoji
            font=(Fonts.FAMILY, 28),
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_MUTED,
        )
        icon_label.pack(pady=(5, 0))

        hint_label = tk.Label(
            self.dropzone,
            text="Click 'Browse' or drag audio file here",
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT,
            fg=Colors.TEXT_SECONDARY,
        )
        hint_label.pack(pady=(0, 5))

        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton",
            command=self._browse_source
        )
        browse_btn.pack(side=tk.RIGHT)

        # Help text
        ttk.Label(
            card,
            text="Supported formats: WAV, MP3, FLAC, OGG",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

        # Bind trace on source_path to update file info
        self.source_path.trace_add("write", self._on_source_path_changed)

    # ── File Info ──────────────────────────────────────────────────────

    def _create_file_info_section(self, parent: tk.Widget):
        """Create file information display section."""
        card = self.create_card(parent, "File Info")

        info_frame = ttk.Frame(card, style="Card.TFrame")
        info_frame.pack(fill=tk.X)

        info_items = [
            ("File", self.file_info_filename),
            ("Duration", self.file_info_duration),
            ("Sample Rate", self.file_info_samplerate),
            ("Channels", self.file_info_channels),
            ("File Size", self.file_info_filesize),
        ]

        for label_text, var in info_items:
            row = ttk.Frame(info_frame, style="Card.TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{label_text}:", style="Card.TLabel", width=12).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Separation Settings ────────────────────────────────────────────

    def _create_mode_section(self, parent: tk.Widget):
        """Create separation mode and backend selection section."""
        card = self.create_card(parent, "Separation Settings")

        # Mode selection
        mode_frame = ttk.Frame(card, style="Card.TFrame")
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(mode_frame, text="Mode:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            mode_frame, textvariable=self.separation_mode,
            values=list(self.MODES.keys()), state="readonly"
        )
        mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Backend selection
        backend_frame = ttk.Frame(card, style="Card.TFrame")
        backend_frame.pack(fill=tk.X, pady=5)
        ttk.Label(backend_frame, text="Backend:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        backend_combo = ttk.Combobox(
            backend_frame, textvariable=self.backend,
            values=list(self.BACKENDS.keys()), state="readonly"
        )
        backend_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Backend description
        self.backend_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.backend_desc.pack(anchor=tk.W, pady=(5, 0))
        self._update_backend_desc()
        self.backend.trace_add("write", lambda *a: self._update_backend_desc())

        # Separator
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Dereverb toggle
        dereverb_frame = ttk.Frame(card, style="Card.TFrame")
        dereverb_frame.pack(fill=tk.X, pady=5)
        dereverb_check = ttk.Checkbutton(
            dereverb_frame,
            text="Enable Dereverberation (reduce room echo)",
            variable=self.dereverb_enabled,
        )
        dereverb_check.pack(side=tk.LEFT)

        # Output format
        fmt_frame = ttk.Frame(card, style="Card.TFrame")
        fmt_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fmt_frame, text="Output Format:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        fmt_combo = ttk.Combobox(
            fmt_frame, textvariable=self.output_format,
            values=list(self.OUTPUT_FORMATS.keys()), state="readonly", width=10
        )
        fmt_combo.pack(side=tk.LEFT)

    # ── Output Directory ───────────────────────────────────────────────

    def _create_output_section(self, parent: tk.Widget):
        """Create output directory selection section."""
        card = self.create_card(parent, "Output Directory")

        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.output_dir)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton",
            command=self._browse_output_dir
        )
        browse_btn.pack(side=tk.RIGHT)

        ttk.Label(
            card,
            text="Output files are auto-named: <source>_<stem>.<format>",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

    # ── Action Buttons ─────────────────────────────────────────────────

    def _create_action_section(self, parent: tk.Widget):
        """Create action buttons section."""
        button_frame = ttk.Frame(parent, style="TFrame")
        button_frame.pack(fill=tk.X, pady=20)

        self.start_btn = ttk.Button(
            button_frame, text="\u25b6 Start Separation",
            style="Primary.TButton",
            command=self._start_separation
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame, text="\u23f9 Stop",
            style="Danger.TButton",
            command=self._stop_separation,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

    # ── Progress ───────────────────────────────────────────────────────

    def _create_progress_section(self, parent: tk.Widget):
        """Create progress display section with elapsed time."""
        card = self.create_card(parent, "Progress")

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            card, variable=self.progress_var,
            maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Status row
        status_frame = ttk.Frame(card, style="Card.TFrame")
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, text="Status:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.status_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Elapsed time row
        time_frame = ttk.Frame(card, style="Card.TFrame")
        time_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(time_frame, text="Elapsed:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(time_frame, textvariable=self.elapsed_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

    # ── Log ────────────────────────────────────────────────────────────

    def _create_log_section(self, parent: tk.Widget):
        """Create log output section."""
        card = self.create_card(parent, "Log Output")

        log_frame = ttk.Frame(card, style="Card.TFrame")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_display = tk.Text(
            log_frame, height=10,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)

    # ── Output Files ───────────────────────────────────────────────────

    def _create_output_files_section(self, parent: tk.Widget):
        """Create output files list section."""
        card = self.create_card(parent, "Output Files")

        self.output_listbox = tk.Listbox(
            card, height=5,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            selectbackground=Colors.ACCENT_PRIMARY,
            selectforeground=Colors.BG_PRIMARY,
        )
        self.output_listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        open_btn = ttk.Button(
            card, text="\U0001f4c1 Open Output Folder",
            style="Secondary.TButton",
            command=self._open_output_folder
        )
        open_btn.pack(anchor=tk.W)

    # ── Browse Handlers ────────────────────────────────────────────────

    def _browse_source(self):
        """Browse for source audio file, remembering last directory."""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=AUDIO_FILETYPES,
            initialdir=self._last_directory,
        )

        if filename:
            self.source_path.set(filename)
            # Remember directory via SettingsManager
            self._last_directory = os.path.dirname(filename)
            self._settings.set("separation_last_dir", self._last_directory)

            # Auto-set output directory if empty
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(filename))

    def _browse_output_dir(self):
        """Browse for output directory, remembering last directory."""
        dirname = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self._last_directory,
        )
        if dirname:
            self.output_dir.set(dirname)
            self._last_directory = dirname
            self._settings.set("separation_last_dir", dirname)

    # ── File Info Update (background thread) ───────────────────────────

    def _on_source_path_changed(self, *args):
        """Triggered when source_path variable changes."""
        filepath = self.source_path.get()
        if not filepath:
            self._reset_file_info()
            return
        if not os.path.isfile(filepath):
            self._reset_file_info()
            return
        # Update file info in background thread (fix #12)
        threading.Thread(target=self._load_file_info, args=(filepath,), daemon=True).start()

    def _reset_file_info(self):
        """Reset file info to defaults."""
        self.file_info_filename.set("No file selected")
        self.file_info_duration.set("--")
        self.file_info_samplerate.set("--")
        self.file_info_channels.set("--")
        self.file_info_filesize.set("--")

    def _load_file_info(self, filepath: str):
        """Load and display audio file information (runs in background thread)."""
        try:
            # File size
            size_bytes = os.path.getsize(filepath)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / 1024 / 1024:.2f} MB"

            filename = os.path.basename(filepath)
            self.safe_after(0, lambda: self.file_info_filename.set(filename))
            self.safe_after(0, lambda: self.file_info_filesize.set(size_str))

            # Try to read audio metadata
            try:
                import soundfile as sf
                info = sf.info(filepath)
                duration_sec = info.duration
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)
                duration_str = f"{minutes}:{seconds:02d}"

                self.safe_after(0, lambda: self.file_info_duration.set(duration_str))
                self.safe_after(0, lambda: self.file_info_samplerate.set(f"{info.samplerate} Hz"))
                ch_map = {1: "Mono", 2: "Stereo"}
                self.safe_after(0, lambda: self.file_info_channels.set(
                    ch_map.get(info.channels, f"{info.channels} ch")
                ))
            except Exception:
                self.safe_after(0, lambda: self.file_info_duration.set("N/A"))
                self.safe_after(0, lambda: self.file_info_samplerate.set("N/A"))
                self.safe_after(0, lambda: self.file_info_channels.set("N/A"))

        except Exception:
            self.safe_after(0, self._reset_file_info)

    # ── Backend Description ────────────────────────────────────────────

    def _update_backend_desc(self):
        """Update backend description label."""
        backend = self.backend.get()
        desc = self.BACKENDS.get(backend, "")
        self.backend_desc.configure(text=desc)

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Add message to log display (must be called from main thread)."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)

    # ── Elapsed Timer ──────────────────────────────────────────────────

    def _start_elapsed_timer(self):
        """Start the elapsed time counter."""
        self._start_time = time.time()
        self._tick_elapsed()

    def _tick_elapsed(self):
        """Update the elapsed time display every second."""
        if self._start_time is not None and not self._cancel_event.is_set():
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d}")
            self._elapsed_timer_id = self.safe_after(1000, self._tick_elapsed)

    def _stop_elapsed_timer(self):
        """Stop the elapsed time counter."""
        if self._elapsed_timer_id is not None:
            try:
                self.after_cancel(self._elapsed_timer_id)
            except Exception:
                pass
            self._elapsed_timer_id = None
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d} (done)")
            self._start_time = None

    # ── Unified UI Reset (fix #5) ──────────────────────────────────────

    def _reset_ui_after_processing(self, status_text: str):
        """
        Unified UI state reset after processing completes, stops, or errors.

        This ensures all buttons, progress bar, and status text are
        consistently reset regardless of how processing ended.
        """
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set(status_text)
        self._stop_elapsed_timer()

    # ── Separation Logic ───────────────────────────────────────────────

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

        # Create output directory
        try:
            os.makedirs(self.output_dir.get(), exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Cannot create output directory:\n{e}")
            return

        # Reset cancel event and update UI state
        self._cancel_event.clear()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Starting...")
        self.progress_var.set(0)
        self.elapsed_var.set("0:00")
        self._output_files = []
        self.output_listbox.delete(0, tk.END)

        # Clear log
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)
        self.log_display.configure(state=tk.DISABLED)

        mode_label = self.separation_mode.get()
        backend_label = self.backend.get()
        fmt_label = self.output_format.get()
        dereverb_on = self.dereverb_enabled.get()

        self._log(f"Starting separation...")
        self._log(f"Source: {os.path.basename(self.source_path.get())}")
        self._log(f"Mode: {mode_label}")
        self._log(f"Backend: {backend_label}")
        self._log(f"Dereverb: {'ON' if dereverb_on else 'OFF'}")
        self._log(f"Output format: {fmt_label}")
        self._log(f"Output dir: {self.output_dir.get()}")
        self._log("")

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start processing in background
        self._processing_thread = threading.Thread(target=self._separation_worker, daemon=True)
        self._processing_thread.start()

    def _stop_separation(self):
        """Stop the separation process via cancel event."""
        self._cancel_event.set()
        self._log("Stop requested...")
        # UI will be reset by the worker thread's finally block

    def _separation_worker(self):
        """Background worker for separation."""
        try:
            from separators.audio_separator import AudioSeparator, SeparationMode

            # Map UI values to API values
            mode_str = self.MODES.get(self.separation_mode.get(), "2stems")
            backend_str = self.backend.get().lower()
            fmt_ext = self.OUTPUT_FORMATS.get(self.output_format.get(), ".wav")
            dereverb_on = self.dereverb_enabled.get()

            # Map backend names to AudioSeparator backend parameter
            if backend_str == "librosa":
                sep_backend = "auto"
            elif backend_str == "demucs":
                sep_backend = "demucs"
            elif backend_str == "hpss":
                sep_backend = "auto"
            else:
                sep_backend = "auto"

            self.safe_after(0, lambda: self._log("Initializing AudioSeparator..."))
            self.safe_after(0, lambda: self.status_var.set("Loading model..."))
            self.safe_after(0, lambda: self.progress_var.set(5))

            separator = AudioSeparator(backend=sep_backend)

            # Load audio
            self.safe_after(0, lambda: self._log("Loading audio..."))
            self.safe_after(0, lambda: self.status_var.set("Loading audio..."))
            self.safe_after(0, lambda: self.progress_var.set(15))

            import numpy as np
            import soundfile as sf

            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)  # Mono to stereo

            self.safe_after(0, lambda: self._log(f"Audio loaded: {audio.shape[0]} samples, {sr}Hz"))
            self.safe_after(0, lambda: self.progress_var.set(25))

            # Optional dereverberation
            if dereverb_on:
                self.safe_after(0, lambda: self._log("Applying dereverberation..."))
                self.safe_after(0, lambda: self.status_var.set("Dereverberating..."))
                self.safe_after(0, lambda: self.progress_var.set(35))
                audio = separator.dereverb(audio)
                self.safe_after(0, lambda: self._log("Dereverberation complete."))
                self.safe_after(0, lambda: self.progress_var.set(45))

            if self._cancel_event.is_set():
                return

            # Perform separation
            if mode_str == "dereverb":
                self.safe_after(0, lambda: self._log("Performing dereverberation..."))
                self.safe_after(0, lambda: self.status_var.set("Dereverberating..."))
                self.safe_after(0, lambda: self.progress_var.set(60))
                result_audio = separator.dereverb(audio)
                results = [("dereverb", result_audio)]
            elif mode_str == "hpss":
                self.safe_after(0, lambda: self._log("Performing HPSS separation..."))
                self.safe_after(0, lambda: self.status_var.set("Separating (HPSS)..."))
                self.safe_after(0, lambda: self.progress_var.set(60))
                harmonic, percussive = separator.hpss(audio, sample_rate=sr)
                results = [("harmonic", harmonic), ("percussive", percussive)]
            else:
                mode_enum = SeparationMode(mode_str)
                self.safe_after(0, lambda: self._log(f"Performing {mode_str} separation..."))
                self.safe_after(0, lambda: self.status_var.set("Separating..."))
                self.safe_after(0, lambda: self.progress_var.set(60))
                stems = separator.separate(audio, mode=mode_enum, sample_rate=sr)

                # Map stem names
                if mode_str == "2stems":
                    stem_names = ["vocals", "accompaniment"]
                elif mode_str == "4stems":
                    stem_names = ["vocals", "drums", "bass", "other"]
                else:
                    stem_names = [f"stem_{i}" for i in range(len(stems))]

                results = list(zip(stem_names, stems))

            if self._cancel_event.is_set():
                return

            self.safe_after(0, lambda: self._log("Separation complete, saving files..."))
            self.safe_after(0, lambda: self.status_var.set("Saving files..."))
            self.safe_after(0, lambda: self.progress_var.set(80))

            # Save output files
            source_name = os.path.splitext(os.path.basename(self.source_path.get()))[0]

            for stem_name, stem_audio in results:
                if self._cancel_event.is_set():
                    return
                output_path = os.path.join(
                    self.output_dir.get(),
                    f"{source_name}_{stem_name}{fmt_ext}"
                )
                sf.write(output_path, stem_audio, sr)
                self._output_files.append(output_path)
                self.safe_after(0, lambda p=output_path: self._log(f"Saved: {os.path.basename(p)}"))
                self.safe_after(0, lambda p=output_path: self.output_listbox.insert(
                    tk.END, os.path.basename(p)
                ))

            self.safe_after(0, lambda: self.progress_var.set(100))
            self.safe_after(0, self._separation_complete)

        except ImportError as e:
            err_msg = f"Missing dependency: {e}"
            self.safe_after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.safe_after(0, lambda: self._log("Please install required packages."))
            self.safe_after(0, lambda: self.status_var.set("Error"))
            self.safe_after(0, self._separation_error)

        except Exception as e:
            err_msg = str(e)
            self.safe_after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.safe_after(0, lambda: self.status_var.set("Error"))
            self.safe_after(0, self._separation_error)

        finally:
            # Always reset UI state when worker exits (fix #5)
            if self._cancel_event.is_set():
                self.safe_after(0, lambda: self._reset_ui_after_processing("Stopped"))
                self.safe_after(0, lambda: self._log("Separation stopped by user."))

    # ── Completion / Error ─────────────────────────────────────────────

    def _separation_complete(self):
        """Handle separation completion."""
        self._reset_ui_after_processing("Completed")

        self._log("")
        self._log(f"Done! {len(self._output_files)} file(s) saved.")

        # Show completion dialog with option to open folder
        result = messagebox.askyesno(
            "Separation Complete",
            f"Separation completed successfully!\n\n"
            f"{len(self._output_files)} file(s) saved to:\n{self.output_dir.get()}\n\n"
            f"Open output folder?"
        )
        if result:
            self._open_output_folder()

    def _separation_error(self):
        """Handle separation error."""
        self._reset_ui_after_processing("Error")

        messagebox.showerror(
            "Separation Failed",
            "Audio separation failed.\n\n"
            "Please check the log for details.\n"
            "Common issues: missing dependencies, unsupported format, or insufficient memory."
        )

    def _open_output_folder(self):
        """Open the output folder in file explorer using common utility."""
        if self.output_dir.get() and os.path.exists(self.output_dir.get()):
            if not open_folder(self.output_dir.get()):
                messagebox.showwarning("Warning", "Could not open folder.")
        else:
            messagebox.showwarning("Warning", "Output folder does not exist.")
