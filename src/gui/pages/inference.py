"""
Inference page for SOMA GUI.

Provides interface for AI cover generation (voice conversion).
Enhanced with: advanced parameter controls, multi-stage progress, file info,
directory memory, auto-naming, completion dialog, and error handling.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import json
from typing import Optional, List
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


# Persistent settings file
_SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".soma_gui_settings.json")


def _load_settings() -> dict:
    """Load persistent GUI settings from disk."""
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict):
    """Save persistent GUI settings to disk."""
    try:
        existing = _load_settings()
        existing.update(data)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


class InferencePage(BasePage):
    """
    Inference page for generating AI covers.

    Features:
    - Source audio selection with file info and drop zone
    - Model selection with refresh
    - Pitch shift: +/-12 semitones
    - Feature extractor: hubert, contentvec
    - F0 method: dio, harvest, rmvpe, crepe
    - Device: auto / cpu / cuda
    - Sample rate selection
    - Clustering ratio slider
    - Multi-stage progress (load model -> feature extract -> vocoder)
    - Elapsed time display
    - Preprocessing options (separate vocals, dereverb)
    - Output path with auto-naming
    - Completion dialog with open folder button
    - Friendly error handling
    """

    PAGE_NAME = "Song Cover"
    PAGE_ICON = "\U0001f3b5"
    PAGE_DESCRIPTION = "Generate covers"

    # Feature extractors
    FEATURE_EXTRACTORS = {
        "hubert": "HuBERT Base (default, good quality)",
        "contentvec": "ContentVec (alternative features)",
    }

    # F0 extraction methods
    F0_METHODS = {
        "dio": "DIO (fast, default)",
        "harvest": "Harvest (better quality, slower)",
        "rmvpe": "RMVPE (deep learning, best quality)",
        "crepe": "Crepe (neural pitch, high quality)",
    }

    # Devices
    DEVICES = {
        "auto": "Auto-detect (GPU if available)",
        "cpu": "CPU (always available)",
        "cuda": "CUDA (NVIDIA GPU)",
    }

    # Sample rates
    SAMPLE_RATES = ["16000", "32000", "40000", "44100", "48000"]

    # Quality presets (for backward compatibility)
    QUALITY_PRESETS = {
        "Standard": {"sample_rate": "40000", "description": "Good quality, faster"},
        "High": {"sample_rate": "44100", "description": "Better quality, moderate speed"},
        "Best": {"sample_rate": "48000", "description": "Highest quality, slower"},
    }

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        """Initialize the inference page."""
        super().__init__(parent, app)

        # State
        self._is_processing = False
        self._processing_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._elapsed_timer_id: Optional[str] = None

        # Variables
        self.source_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.pitch_shift = tk.IntVar(value=0)
        self.quality = tk.StringVar(value="High")

        # Advanced parameters
        self.feature_extractor = tk.StringVar(value="hubert")
        self.f0_method = tk.StringVar(value="dio")
        self.device = tk.StringVar(value="auto")
        self.output_sample_rate = tk.StringVar(value="40000")
        self.cluster_ratio = tk.DoubleVar(value=0.0)

        # Preprocessing options
        self.separate_vocals = tk.BooleanVar(value=True)
        self.dereverb_audio = tk.BooleanVar(value=False)
        self.separation_mode = tk.StringVar(value="2-stem")

        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_var = tk.StringVar(value="")
        self.stage_var = tk.StringVar(value="")

        # File info variables
        self.file_info_duration = tk.StringVar(value="--")
        self.file_info_samplerate = tk.StringVar(value="--")
        self.file_info_channels = tk.StringVar(value="--")
        self.file_info_filesize = tk.StringVar(value="--")
        self.file_info_filename = tk.StringVar(value="No file selected")

        # Remembered last directory
        settings = _load_settings()
        self._last_directory = settings.get("inference_last_dir", os.path.expanduser("~"))

        # Available models
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

        self._create_dropzone_section(left_frame)
        self._create_file_info_section(left_frame)
        self._create_model_section(left_frame)
        self._create_basic_options_section(left_frame)
        self._create_advanced_options_section(left_frame)
        self._create_preprocessing_section(left_frame)
        self._create_output_section(left_frame)

        # Right column - Progress
        right_frame = ttk.Frame(main_frame, style="TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self._create_progress_section(right_frame)
        self._create_log_section(right_frame)

    # ── Drop Zone ──────────────────────────────────────────────────────

    def _create_dropzone_section(self, parent: tk.Widget):
        """Create a visual drop zone for source audio files."""
        card = self.create_card(parent, "Source Audio")

        # Drop zone visual
        self.dropzone = tk.Frame(
            card, bg=Colors.BG_INPUT,
            highlightbackground=Colors.BORDER, highlightthickness=2,
            padx=20, pady=15,
        )
        self.dropzone.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            self.dropzone, text="\U0001f3a4",
            font=(Fonts.FAMILY, 28), bg=Colors.BG_INPUT, fg=Colors.TEXT_MUTED,
        ).pack(pady=(5, 0))

        tk.Label(
            self.dropzone,
            text="Click 'Browse' or drag dry vocal file here",
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_SECONDARY,
        ).pack(pady=(0, 5))

        # Path entry with browse button
        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        path_entry = ttk.Entry(path_frame, textvariable=self.source_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton", command=self._browse_source
        )
        browse_btn.pack(side=tk.RIGHT)

        ttk.Label(
            card, text="Supported: WAV, MP3, FLAC, OGG",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(10, 0))

        # Bind trace to update file info
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

    # ── Model Selection ────────────────────────────────────────────────

    def _create_model_section(self, parent: tk.Widget):
        """Create model selection section."""
        card = self.create_card(parent, "Voice Model")

        model_frame = ttk.Frame(card, style="Card.TFrame")
        model_frame.pack(fill=tk.X)

        ttk.Label(model_frame, text="Model:", style="Card.TLabel").pack(side=tk.LEFT)

        self._model_combo = ttk.Combobox(
            model_frame, textvariable=self.selected_model, state="readonly"
        )
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

        refresh_btn = ttk.Button(
            model_frame, text="\U0001f504",
            style="Secondary.TButton", command=self._refresh_models
        )
        refresh_btn.pack(side=tk.RIGHT, padx=(10, 0))

        self._refresh_models()

    # ── Basic Options ──────────────────────────────────────────────────

    def _create_basic_options_section(self, parent: tk.Widget):
        """Create basic conversion options (pitch, quality)."""
        card = self.create_card(parent, "Basic Options")

        # Pitch shift
        pitch_frame = ttk.Frame(card, style="Card.TFrame")
        pitch_frame.pack(fill=tk.X, pady=5)
        ttk.Label(pitch_frame, text="Pitch Shift:", style="Card.TLabel", width=14).pack(side=tk.LEFT)

        pitch_spinbox = tk.Spinbox(
            pitch_frame, from_=-12, to=12,
            textvariable=self.pitch_shift,
            font=(Fonts.FAMILY, Fonts.SIZE_BODY),
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            buttonbackground=Colors.BG_TERTIARY, width=6
        )
        pitch_spinbox.pack(side=tk.LEFT)
        ttk.Label(pitch_frame, text="semitones (-12 to +12)", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

        # Output sample rate
        sr_frame = ttk.Frame(card, style="Card.TFrame")
        sr_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sr_frame, text="Sample Rate:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        sr_combo = ttk.Combobox(
            sr_frame, textvariable=self.output_sample_rate,
            values=self.SAMPLE_RATES, state="readonly", width=10
        )
        sr_combo.pack(side=tk.LEFT)
        ttk.Label(sr_frame, text="Hz (output)", style="Muted.TLabel").pack(side=tk.LEFT, padx=(10, 0))

    # ── Advanced Options ───────────────────────────────────────────────

    def _create_advanced_options_section(self, parent: tk.Widget):
        """Create advanced parameter controls."""
        card = self.create_card(parent, "Advanced Options")

        # Feature extractor
        fe_frame = ttk.Frame(card, style="Card.TFrame")
        fe_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fe_frame, text="Feature Extractor:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        fe_combo = ttk.Combobox(
            fe_frame, textvariable=self.feature_extractor,
            values=list(self.FEATURE_EXTRACTORS.keys()), state="readonly", width=14
        )
        fe_combo.pack(side=tk.LEFT)

        # Feature extractor description
        self.fe_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.fe_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_fe_desc()
        self.feature_extractor.trace_add("write", lambda *a: self._update_fe_desc())

        # F0 method
        f0_frame = ttk.Frame(card, style="Card.TFrame")
        f0_frame.pack(fill=tk.X, pady=5)
        ttk.Label(f0_frame, text="F0 Method:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        f0_combo = ttk.Combobox(
            f0_frame, textvariable=self.f0_method,
            values=list(self.F0_METHODS.keys()), state="readonly", width=14
        )
        f0_combo.pack(side=tk.LEFT)

        # F0 method description
        self.f0_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.f0_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_f0_desc()
        self.f0_method.trace_add("write", lambda *a: self._update_f0_desc())

        # Device
        dev_frame = ttk.Frame(card, style="Card.TFrame")
        dev_frame.pack(fill=tk.X, pady=5)
        ttk.Label(dev_frame, text="Device:", style="Card.TLabel", width=16).pack(side=tk.LEFT)
        dev_combo = ttk.Combobox(
            dev_frame, textvariable=self.device,
            values=list(self.DEVICES.keys()), state="readonly", width=14
        )
        dev_combo.pack(side=tk.LEFT)

        # Device description
        self.dev_desc = ttk.Label(card, text="", style="Muted.TLabel")
        self.dev_desc.pack(anchor=tk.W, pady=(0, 5))
        self._update_dev_desc()
        self.device.trace_add("write", lambda *a: self._update_dev_desc())

        # Separator
        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Clustering ratio slider
        cluster_frame = ttk.Frame(card, style="Card.TFrame")
        cluster_frame.pack(fill=tk.X, pady=5)
        ttk.Label(cluster_frame, text="Cluster Ratio:", style="Card.TLabel", width=16).pack(side=tk.LEFT)

        self.cluster_scale = ttk.Scale(
            cluster_frame, from_=0.0, to=1.0,
            variable=self.cluster_ratio, orient=tk.HORIZONTAL,
        )
        self.cluster_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        self.cluster_value_label = ttk.Label(cluster_frame, text="0.0", style="Card.TLabel", width=5)
        self.cluster_value_label.pack(side=tk.LEFT)
        self.cluster_ratio.trace_add("write", self._update_cluster_label)

        ttk.Label(
            card,
            text="Timbre clustering ratio (0 = original, 1 = fully clustered)",
            style="Muted.TLabel"
        ).pack(anchor=tk.W, pady=(5, 0))

    # ── Preprocessing ──────────────────────────────────────────────────

    def _create_preprocessing_section(self, parent: tk.Widget):
        """Create audio preprocessing options."""
        card = self.create_card(parent, "Audio Preprocessing")

        # Separate vocals checkbox
        separate_frame = ttk.Frame(card, style="Card.TFrame")
        separate_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            separate_frame,
            text="Separate vocals first (recommended for songs with instruments)",
            variable=self.separate_vocals,
        ).pack(side=tk.LEFT)

        # Dereverb checkbox
        dereverb_frame = ttk.Frame(card, style="Card.TFrame")
        dereverb_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            dereverb_frame,
            text="Remove reverb (improves voice conversion quality)",
            variable=self.dereverb_audio,
        ).pack(side=tk.LEFT)

        # Separation mode (if separating)
        sep_mode_frame = ttk.Frame(card, style="Card.TFrame")
        sep_mode_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sep_mode_frame, text="Separation:", style="Card.TLabel", width=14).pack(side=tk.LEFT)
        sep_mode_combo = ttk.Combobox(
            sep_mode_frame, textvariable=self.separation_mode,
            values=["2-stem", "4-stem"], state="readonly", width=10
        )
        sep_mode_combo.pack(side=tk.LEFT)
        ttk.Label(sep_mode_frame, text="(vocals + accompaniment)", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

    # ── Output ─────────────────────────────────────────────────────────

    def _create_output_section(self, parent: tk.Widget):
        """Create output configuration section."""
        card = self.create_card(parent, "Output")

        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        ttk.Label(path_frame, text="Output:", style="Card.TLabel").pack(side=tk.LEFT)

        path_entry = ttk.Entry(path_frame, textvariable=self.output_path)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))

        browse_btn = ttk.Button(
            path_frame, text="Browse...",
            style="Secondary.TButton", command=self._browse_output
        )
        browse_btn.pack(side=tk.RIGHT)

        # Action buttons
        button_frame = ttk.Frame(card, style="Card.TFrame")
        button_frame.pack(fill=tk.X, pady=(15, 0))

        self.convert_btn = ttk.Button(
            button_frame, text="\u25b6 Start Conversion",
            style="Primary.TButton", command=self._start_conversion
        )
        self.convert_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame, text="\u23f9 Stop",
            style="Danger.TButton", command=self._stop_conversion,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

    # ── Progress ───────────────────────────────────────────────────────

    def _create_progress_section(self, parent: tk.Widget):
        """Create multi-stage progress display."""
        card = self.create_card(parent, "Conversion Progress")

        # Stage indicator
        stage_frame = ttk.Frame(card, style="Card.TFrame")
        stage_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(stage_frame, text="Stage:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Label(stage_frame, textvariable=self.stage_var, style="Card.TLabel").pack(
            side=tk.LEFT, padx=(10, 0)
        )

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
            log_frame, height=12,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self.log_display.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.log_display, command=self.log_display.yview)
        self.log_display.configure(yscrollcommand=scrollbar.set)

    # ── Browse Handlers ────────────────────────────────────────────────

    def _browse_source(self):
        """Open file dialog to select source audio."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Source Audio",
            filetypes=filetypes,
            initialdir=self._last_directory,
        )

        if filename:
            self.source_path.set(filename)
            self._last_directory = os.path.dirname(filename)
            _save_settings({"inference_last_dir": self._last_directory})

            # Auto-generate output path
            base, ext = os.path.splitext(filename)
            self.output_path.set(f"{base}_cover.wav")

    def _browse_output(self):
        """Open file dialog to select output path."""
        filetypes = [
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        ]

        filename = filedialog.asksaveasfilename(
            title="Save Output Audio",
            filetypes=filetypes,
            defaultextension=".wav",
            initialdir=self._last_directory,
        )

        if filename:
            self.output_path.set(filename)
            self._last_directory = os.path.dirname(filename)
            _save_settings({"inference_last_dir": self._last_directory})

    # ── File Info ──────────────────────────────────────────────────────

    def _on_source_path_changed(self, *args):
        """Triggered when source_path variable changes."""
        filepath = self.source_path.get()
        if not filepath or not os.path.isfile(filepath):
            self._reset_file_info()
            return
        threading.Thread(target=self._load_file_info, args=(filepath,), daemon=True).start()

    def _reset_file_info(self):
        """Reset file info to defaults."""
        self.file_info_filename.set("No file selected")
        self.file_info_duration.set("--")
        self.file_info_samplerate.set("--")
        self.file_info_channels.set("--")
        self.file_info_filesize.set("--")

    def _load_file_info(self, filepath: str):
        """Load and display audio file information."""
        try:
            size_bytes = os.path.getsize(filepath)
            if size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / 1024 / 1024:.2f} MB"

            filename = os.path.basename(filepath)
            self.after(0, lambda: self.file_info_filename.set(filename))
            self.after(0, lambda: self.file_info_filesize.set(size_str))

            try:
                import soundfile as sf
                info = sf.info(filepath)
                duration_sec = info.duration
                minutes = int(duration_sec // 60)
                seconds = int(duration_sec % 60)

                self.after(0, lambda: self.file_info_duration.set(f"{minutes}:{seconds:02d}"))
                self.after(0, lambda: self.file_info_samplerate.set(f"{info.samplerate} Hz"))
                ch_map = {1: "Mono", 2: "Stereo"}
                self.after(0, lambda: self.file_info_channels.set(
                    ch_map.get(info.channels, f"{info.channels} ch")
                ))
            except Exception:
                self.after(0, lambda: self.file_info_duration.set("N/A"))
                self.after(0, lambda: self.file_info_samplerate.set("N/A"))
                self.after(0, lambda: self.file_info_channels.set("N/A"))
        except Exception:
            self.after(0, self._reset_file_info)

    # ── Description Updaters ───────────────────────────────────────────

    def _update_fe_desc(self):
        fe = self.feature_extractor.get()
        self.fe_desc.configure(text=self.FEATURE_EXTRACTORS.get(fe, ""))

    def _update_f0_desc(self):
        f0 = self.f0_method.get()
        self.f0_desc.configure(text=self.F0_METHODS.get(f0, ""))

    def _update_dev_desc(self):
        dev = self.device.get()
        self.dev_desc.configure(text=self.DEVICES.get(dev, ""))

    def _update_cluster_label(self, *args):
        val = self.cluster_ratio.get()
        self.cluster_value_label.configure(text=f"{val:.2f}")

    # ── Model Management ───────────────────────────────────────────────

    def _refresh_models(self):
        """Refresh the list of available models."""
        # Scan model directory for .pth files
        model_dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]

        models = []
        for model_dir in model_dirs:
            if os.path.isdir(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith(".pth"):
                        models.append(f[:-4])  # Remove .pth extension

        if not models:
            models = ["No models available"]

        self._available_models = models
        self._model_combo.configure(values=self._available_models)

        if self._available_models and self._available_models[0] != "No models available":
            self._model_combo.set(self._available_models[0])

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Add message to log display (main thread only)."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)

    # ── Elapsed Timer ──────────────────────────────────────────────────

    def _start_elapsed_timer(self):
        self._start_time = time.time()
        self._tick_elapsed()

    def _tick_elapsed(self):
        if self._start_time is not None and self._is_processing:
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d}")
            self._elapsed_timer_id = self.after(1000, self._tick_elapsed)

    def _stop_elapsed_timer(self):
        if self._elapsed_timer_id is not None:
            self.after_cancel(self._elapsed_timer_id)
            self._elapsed_timer_id = None
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d} (done)")
            self._start_time = None

    # ── Conversion Logic ───────────────────────────────────────────────

    def _start_conversion(self):
        """Start the conversion process."""
        # Validate inputs
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file.")
            return

        if not os.path.exists(self.source_path.get()):
            messagebox.showerror("Error", "Source file does not exist.")
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
        self.status_var.set("Starting...")
        self.stage_var.set("Initializing")
        self.progress_var.set(0)
        self.elapsed_var.set("0:00")

        # Clear log
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.delete(1.0, tk.END)
        self.log_display.configure(state=tk.DISABLED)

        self._log("Starting voice conversion...")
        self._log(f"Source: {os.path.basename(self.source_path.get())}")
        self._log(f"Model: {self.selected_model.get()}")
        self._log(f"Pitch: {self.pitch_shift.get()} semitones")
        self._log(f"Feature extractor: {self.feature_extractor.get()}")
        self._log(f"F0 method: {self.f0_method.get()}")
        self._log(f"Device: {self.device.get()}")
        self._log(f"Output sample rate: {self.output_sample_rate.get()} Hz")
        self._log(f"Cluster ratio: {self.cluster_ratio.get():.2f}")
        self._log(f"Output: {self.output_path.get()}")
        self._log("")

        # Start elapsed timer
        self._start_elapsed_timer()

        # Start processing in background
        self._processing_thread = threading.Thread(target=self._conversion_worker, daemon=True)
        self._processing_thread.start()

    def _stop_conversion(self):
        """Stop the conversion process."""
        self._is_processing = False
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")
        self.stage_var.set("")
        self._stop_elapsed_timer()
        self._log("Conversion stopped by user.")

    def _conversion_worker(self):
        """Background worker for voice conversion."""
        try:
            import numpy as np
            import soundfile as sf

            # Stage 1: Load model
            self.after(0, lambda: self._set_stage("Loading model"))
            self.after(0, lambda: self.status_var.set("Loading model..."))
            self.after(0, lambda: self.progress_var.set(5))
            self.after(0, lambda: self._log("[1/3] Loading voice model..."))

            from training.inference import RVCInference

            # Determine device
            device_str = self.device.get()
            if device_str == "auto":
                try:
                    import torch
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device_str = "cpu"

            # Find model file
            model_name = self.selected_model.get()
            model_path = self._find_model_file(model_name)

            if model_path is None:
                raise FileNotFoundError(
                    f"Model '{model_name}' not found. "
                    "Please place .pth model files in ~/.soma/models/ or assets/models/"
                )

            output_sr = int(self.output_sample_rate.get())

            pipeline = RVCInference(
                model_path=model_path,
                device=device_str,
                output_sample_rate=output_sr,
                f0_method=self.f0_method.get(),
            )

            if not self._is_processing:
                return

            self.after(0, lambda: self.progress_var.set(20))
            self.after(0, lambda: self._log(f"Model loaded on {device_str}"))

            # Stage 2: Load and preprocess audio
            self.after(0, lambda: self._set_stage("Loading audio"))
            self.after(0, lambda: self.status_var.set("Loading audio..."))
            self.after(0, lambda: self.progress_var.set(30))
            self.after(0, lambda: self._log("[2/3] Loading source audio..."))

            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 2 and audio.shape[1] > 2:
                audio = np.mean(audio, axis=1)  # Downmix to mono

            self.after(0, lambda: self._log(f"Audio loaded: {len(audio)} samples, {sr}Hz"))

            # Optional preprocessing
            if self.separate_vocals.get() or self.dereverb_audio.get():
                self.after(0, lambda: self._set_stage("Preprocessing"))
                self.after(0, lambda: self.status_var.set("Preprocessing audio..."))
                self.after(0, lambda: self.progress_var.set(35))

                try:
                    from separators.audio_separator import AudioSeparator, SeparationMode

                    separator = AudioSeparator()

                    if self.dereverb_audio.get():
                        self.after(0, lambda: self._log("Applying dereverberation..."))
                        audio = separator.dereverb(audio)

                    if self.separate_vocals.get():
                        self.after(0, lambda: self._log("Separating vocals..."))
                        mode = SeparationMode.TWO_STEMS
                        vocals, _ = separator.separate(audio, mode=mode, sample_rate=sr)
                        audio = vocals
                        self.after(0, lambda: self._log("Vocals separated."))
                except ImportError:
                    self.after(0, lambda: self._log("Warning: Preprocessing skipped (missing deps)"))

            if not self._is_processing:
                return

            # Stage 3: Voice conversion
            self.after(0, lambda: self._set_stage("Voice conversion"))
            self.after(0, lambda: self.status_var.set("Converting voice..."))
            self.after(0, lambda: self.progress_var.set(50))
            self.after(0, lambda: self._log("[3/3] Running voice conversion..."))

            transpose = float(self.pitch_shift.get())

            # Use chunked conversion for long audio
            duration_sec = len(audio) / sr
            if duration_sec > 30:
                self.after(0, lambda: self._log(f"Long audio ({duration_sec:.1f}s), using chunked mode..."))
                result = pipeline.convert_long_audio(
                    audio, sample_rate=sr, transpose=transpose
                )
            else:
                result = pipeline.convert(audio, sample_rate=sr, transpose=transpose)

            if not self._is_processing:
                return

            # Save output
            self.after(0, lambda: self._set_stage("Saving output"))
            self.after(0, lambda: self.status_var.set("Saving output..."))
            self.after(0, lambda: self.progress_var.set(90))

            output_dir = os.path.dirname(self.output_path.get())
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            sf.write(self.output_path.get(), result, output_sr)

            self.after(0, lambda: self.progress_var.set(100))
            self.after(0, lambda: self._log(f"Output saved: {self.output_path.get()}"))
            self.after(0, self._conversion_complete)

        except ImportError as e:
            err_msg = f"Missing dependency: {e}"
            self.after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.after(0, lambda: self.status_var.set("Error"))
            self.after(0, self._conversion_error)

        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.after(0, lambda: self.status_var.set("Error"))
            self.after(0, self._conversion_error)

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """Find a model file by name in known model directories."""
        search_dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]

        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            # Direct match: model_name.pth
            direct = os.path.join(search_dir, f"{model_name}.pth")
            if os.path.isfile(direct):
                return direct
            # Recursive search
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f == f"{model_name}.pth" or f == model_name:
                        return os.path.join(root, f)

        return None

    def _set_stage(self, stage_text: str):
        """Update the stage indicator (must be called via self.after from worker)."""
        self.stage_var.set(stage_text)

    # ── Completion / Error ─────────────────────────────────────────────

    def _conversion_complete(self):
        """Handle conversion completion."""
        self._is_processing = False
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set("Completed")
        self.stage_var.set("Done")
        self.progress_var.set(100)
        self._stop_elapsed_timer()

        self._log("")
        self._log("Voice conversion completed successfully!")

        output_dir = os.path.dirname(self.output_path.get())
        result = messagebox.askyesno(
            "Conversion Complete",
            f"Voice conversion completed successfully!\n\n"
            f"Output: {self.output_path.get()}\n\n"
            f"Open output folder?"
        )
        if result and output_dir and os.path.exists(output_dir):
            self._open_folder(output_dir)

    def _conversion_error(self):
        """Handle conversion error."""
        self._is_processing = False
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.stage_var.set("")
        self._stop_elapsed_timer()

        messagebox.showerror(
            "Conversion Failed",
            "Voice conversion failed.\n\n"
            "Please check the log for details.\n"
            "Common issues: missing model file, incompatible audio format, or insufficient memory."
        )

    @staticmethod
    def _open_folder(folder: str):
        """Open a folder in the system file explorer."""
        import subprocess
        import sys

        folder = os.path.normpath(folder)
        try:
            if sys.platform == 'win32':
                subprocess.run(['explorer', folder], check=False)
            elif sys.platform == 'darwin':
                subprocess.run(['open', folder], check=False)
            else:
                subprocess.run(['xdg-open', folder], check=False)
        except Exception:
            pass
