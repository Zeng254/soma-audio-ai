"""
Comparison page for SOMA GUI.

Provides interface for comparing voice conversion results across different
models and parameter settings. Supports multi-task queue, A/B playback
switching, batch export, and configuration save/load.

Layout:
- Left panel:  Source audio + parameter configuration + action buttons
- Right panel: Task list (top) + result comparison / playback (bottom)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import time
import json
import uuid
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any
from gui.pages.base import BasePage
from gui.styles import Colors, Fonts


# ── Persistent helpers ─────────────────────────────────────────────────

_SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".soma_gui_settings.json")


def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(data: dict):
    try:
        existing = _load_settings()
        existing.update(data)
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ── Task status constants ──────────────────────────────────────────────

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class ComparisonPage(BasePage):
    """
    Comparison page for A/B testing voice conversion results.

    Features:
    - Multi-task queue with independent parameter sets
    - Background thread pool for parallel processing
    - A/B audio switching via system player
    - Batch export with auto-naming
    - Configuration save / load
    """

    PAGE_NAME = "Compare"
    PAGE_ICON = "\U0001f500"  # 🔀
    PAGE_DESCRIPTION = "Compare results"

    # Options mirrors the inference page
    FEATURE_EXTRACTORS = ["hubert", "contentvec"]
    F0_METHODS = ["dio", "harvest", "rmvpe", "crepe"]
    DEVICES = ["auto", "cpu", "cuda"]
    SAMPLE_RATES = ["16000", "32000", "40000", "44100", "48000"]

    def __init__(self, parent: tk.Widget, app: Optional[object] = None):
        super().__init__(parent, app)

        # ── State ──
        self._tasks: List[Dict[str, Any]] = []  # task queue
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cmp")
        self._playback_process: Optional[subprocess.Popen] = None
        self._task_id_counter = 0

        # ── Tkinter variables ──
        self.source_path = tk.StringVar()
        self.cfg_model = tk.StringVar()
        self.cfg_pitch = tk.IntVar(value=0)
        self.cfg_f0 = tk.StringVar(value="dio")
        self.cfg_feature = tk.StringVar(value="hubert")
        self.cfg_cluster = tk.DoubleVar(value=0.0)
        self.cfg_device = tk.StringVar(value="auto")
        self.cfg_sr = tk.StringVar(value="40000")

        # File info
        self.file_info_text = tk.StringVar(value="No source loaded")

        # Cluster label
        self.cluster_label_var = tk.StringVar(value="0.00")

        # Available models
        self._available_models: List[str] = []

        # Remembered directory
        settings = _load_settings()
        self._last_dir = settings.get("comparison_last_dir", os.path.expanduser("~"))

    # ═══════════════════════════════════════════════════════════════════
    #  Widget creation
    # ═══════════════════════════════════════════════════════════════════

    def _create_widgets(self):
        self.create_title_section(
            self.content_frame,
            "Effect Comparison",
            "Compare voice conversion results across models and parameters"
        )

        # Two-column layout
        main = ttk.Frame(self.content_frame, style="TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, style="TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 8))

        right = ttk.Frame(main, style="TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # ── Left: source + config + actions ──
        self._build_source_card(left)
        self._build_config_card(left)
        self._build_action_card(left)

        # ── Right: task list + results ──
        self._build_task_list_card(right)
        self._build_result_card(right)

    # ── Source audio ───────────────────────────────────────────────────

    def _build_source_card(self, parent: tk.Widget):
        card = self.create_card(parent, "Source Audio")

        path_frame = ttk.Frame(card, style="Card.TFrame")
        path_frame.pack(fill=tk.X)

        ttk.Entry(path_frame, textvariable=self.source_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(path_frame, text="Browse...", style="Secondary.TButton",
                   command=self._browse_source).pack(side=tk.RIGHT)

        ttk.Label(card, textvariable=self.file_info_text,
                  style="Muted.TLabel").pack(anchor=tk.W, pady=(8, 0))

    # ── Parameter config ───────────────────────────────────────────────

    def _build_config_card(self, parent: tk.Widget):
        card = self.create_card(parent, "Task Parameters")

        rows = [
            ("Model", self._build_model_row),
            ("Pitch", self._build_pitch_row),
            ("F0 Method", self._build_f0_row),
            ("Feature Ext.", self._build_feature_row),
            ("Cluster Ratio", self._build_cluster_row),
            ("Device", self._build_device_row),
            ("Sample Rate", self._build_sr_row),
        ]

        for label_text, builder in rows:
            row = ttk.Frame(card, style="Card.TFrame")
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label_text + ":", style="Card.TLabel", width=13).pack(side=tk.LEFT)
            builder(row)

    def _build_model_row(self, parent: tk.Widget):
        self._model_combo = ttk.Combobox(parent, textvariable=self.cfg_model, state="readonly")
        self._model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(parent, text="\U0001f504", style="Secondary.TButton",
                   command=self._refresh_models, width=3).pack(side=tk.RIGHT, padx=(5, 0))
        self._refresh_models()

    def _build_pitch_row(self, parent: tk.Widget):
        tk.Spinbox(parent, from_=-12, to=12, textvariable=self.cfg_pitch,
                   font=(Fonts.FAMILY, Fonts.SIZE_BODY),
                   bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
                   buttonbackground=Colors.BG_TERTIARY, width=6).pack(side=tk.LEFT)
        ttk.Label(parent, text="semitones", style="Muted.TLabel").pack(side=tk.LEFT, padx=(8, 0))

    def _build_f0_row(self, parent: tk.Widget):
        ttk.Combobox(parent, textvariable=self.cfg_f0,
                     values=self.F0_METHODS, state="readonly", width=12).pack(side=tk.LEFT)

    def _build_feature_row(self, parent: tk.Widget):
        ttk.Combobox(parent, textvariable=self.cfg_feature,
                     values=self.FEATURE_EXTRACTORS, state="readonly", width=12).pack(side=tk.LEFT)

    def _build_cluster_row(self, parent: tk.Widget):
        ttk.Scale(parent, from_=0.0, to=1.0, variable=self.cfg_cluster,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Label(parent, textvariable=self.cluster_label_var, style="Card.TLabel", width=5).pack(side=tk.LEFT)
        self.cfg_cluster.trace_add("write", self._update_cluster_label)

    def _build_device_row(self, parent: tk.Widget):
        ttk.Combobox(parent, textvariable=self.cfg_device,
                     values=self.DEVICES, state="readonly", width=10).pack(side=tk.LEFT)

    def _build_sr_row(self, parent: tk.Widget):
        ttk.Combobox(parent, textvariable=self.cfg_sr,
                     values=self.SAMPLE_RATES, state="readonly", width=10).pack(side=tk.LEFT)
        ttk.Label(parent, text="Hz", style="Muted.TLabel").pack(side=tk.LEFT, padx=(5, 0))

    # ── Action buttons ─────────────────────────────────────────────────

    def _build_action_card(self, parent: tk.Widget):
        card = self.create_card(parent, "Actions")

        # Row 1: add / duplicate
        row1 = ttk.Frame(card, style="Card.TFrame")
        row1.pack(fill=tk.X, pady=3)
        ttk.Button(row1, text="\u2795 Add Task", style="Primary.TButton",
                   command=self._add_task).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row1, text="\U0001f4cb Duplicate Last", style="Secondary.TButton",
                   command=self._duplicate_last_task).pack(side=tk.LEFT)

        # Row 2: batch start / cancel all / clear done
        row2 = ttk.Frame(card, style="Card.TFrame")
        row2.pack(fill=tk.X, pady=3)
        ttk.Button(row2, text="\u25b6 Start All", style="Primary.TButton",
                   command=self._start_all_tasks).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="\u23f9 Cancel All", style="Danger.TButton",
                   command=self._cancel_all_tasks).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row2, text="\U0001f5d1 Clear Done", style="Secondary.TButton",
                   command=self._clear_done_tasks).pack(side=tk.LEFT)

        # Row 3: save / load config
        row3 = ttk.Frame(card, style="Card.TFrame")
        row3.pack(fill=tk.X, pady=3)
        ttk.Button(row3, text="\U0001f4be Save Config", style="Secondary.TButton",
                   command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row3, text="\U0001f4c2 Load Config", style="Secondary.TButton",
                   command=self._load_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(row3, text="\U0001f4e6 Export All", style="Secondary.TButton",
                   command=self._export_all).pack(side=tk.LEFT)

    # ── Task list ──────────────────────────────────────────────────────

    def _build_task_list_card(self, parent: tk.Widget):
        card = self.create_card(parent, "Task Queue")

        # Treeview for task list
        columns = ("id", "model", "pitch", "f0", "feature", "status", "time")
        self._task_tree = ttk.Treeview(card, columns=columns, show="headings", height=8)

        col_config = [
            ("id", "#", 40),
            ("model", "Model", 120),
            ("pitch", "Pitch", 55),
            ("f0", "F0", 65),
            ("feature", "Feature", 75),
            ("status", "Status", 80),
            ("time", "Time", 60),
        ]
        for cid, heading, width in col_config:
            self._task_tree.heading(cid, text=heading)
            self._task_tree.column(cid, width=width, minwidth=40)

        self._task_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Per-task action buttons
        btn_row = ttk.Frame(card, style="Card.TFrame")
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="\u25b6 Play", style="Primary.TButton",
                   command=self._play_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="\U0001f504 A/B Switch", style="Secondary.TButton",
                   command=self._ab_switch).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="\u274c Cancel", style="Danger.TButton",
                   command=self._cancel_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_row, text="\U0001f5d1 Remove", style="Secondary.TButton",
                   command=self._remove_selected).pack(side=tk.LEFT)

    # ── Result / playback area ─────────────────────────────────────────

    def _build_result_card(self, parent: tk.Widget):
        card = self.create_card(parent, "Playback & Results")

        # Now-playing info
        self.now_playing_var = tk.StringVar(value="Select a completed task and click Play")
        ttk.Label(card, textvariable=self.now_playing_var,
                  style="Card.TLabel").pack(anchor=tk.W, pady=(0, 8))

        # Playback controls
        ctrl = ttk.Frame(card, style="Card.TFrame")
        ctrl.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(ctrl, text="\u25b6 Play", style="Primary.TButton",
                   command=self._play_selected).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctrl, text="\u23f9 Stop", style="Danger.TButton",
                   command=self._stop_playback).pack(side=tk.LEFT, padx=(0, 5))

        # Volume
        ttk.Label(ctrl, text="Vol:", style="Card.TLabel").pack(side=tk.LEFT, padx=(15, 5))
        self._volume_var = tk.IntVar(value=80)
        ttk.Scale(ctrl, from_=0, to=100, variable=self._volume_var,
                  orient=tk.HORIZONTAL, length=120).pack(side=tk.LEFT)

        # Result list (text area showing completed results)
        self._result_text = tk.Text(
            card, height=6,
            bg=Colors.BG_INPUT, fg=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY_MONO, Fonts.SIZE_SMALL),
            wrap=tk.WORD, state=tk.DISABLED
        )
        self._result_text.pack(fill=tk.BOTH, expand=True)

    # ═══════════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _update_cluster_label(self, *args):
        self.cluster_label_var.set(f"{self.cfg_cluster.get():.2f}")

    def _browse_source(self):
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(
            title="Select Source Audio", filetypes=filetypes,
            initialdir=self._last_dir
        )
        if path:
            self.source_path.set(path)
            self._last_dir = os.path.dirname(path)
            _save_settings({"comparison_last_dir": self._last_dir})
            self._update_file_info(path)

    def _update_file_info(self, filepath: str):
        """Load and display basic audio file info."""
        def _load():
            try:
                size = os.path.getsize(filepath)
                size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
                name = os.path.basename(filepath)
                try:
                    import soundfile as sf
                    info = sf.info(filepath)
                    dur = info.duration
                    m, s = int(dur // 60), int(dur % 60)
                    text = f"{name}  |  {m}:{s:02d}  |  {info.samplerate}Hz  |  {size_str}"
                except Exception:
                    text = f"{name}  |  {size_str}"
                self.after(0, lambda: self.file_info_text.set(text))
            except Exception:
                self.after(0, lambda: self.file_info_text.set("Cannot read file info"))
        threading.Thread(target=_load, daemon=True).start()

    def _refresh_models(self):
        """Scan model directories for .pth files."""
        dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]
        models = []
        for d in dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".pth"):
                        models.append(f[:-4])
        if not models:
            models = ["No models available"]
        self._available_models = models
        self._model_combo.configure(values=models)
        if models and models[0] != "No models available":
            self._model_combo.set(models[0])

    def _find_model_file(self, model_name: str) -> Optional[str]:
        """Locate model .pth file by name."""
        dirs = [
            os.path.join(os.path.expanduser("~"), ".soma", "models"),
            os.path.join(os.getcwd(), "assets", "models"),
        ]
        for d in dirs:
            if not os.path.isdir(d):
                continue
            direct = os.path.join(d, f"{model_name}.pth")
            if os.path.isfile(direct):
                return direct
            for root, _, files in os.walk(d):
                for f in files:
                    if f == f"{model_name}.pth":
                        return os.path.join(root, f)
        return None

    def _get_current_config(self) -> Dict[str, Any]:
        """Capture current parameter panel as a config dict."""
        return {
            "model": self.cfg_model.get(),
            "pitch": self.pitch_shift_val(),
            "f0_method": self.cfg_f0.get(),
            "feature_extractor": self.cfg_feature.get(),
            "cluster_ratio": round(self.cfg_cluster.get(), 2),
            "device": self.cfg_device.get(),
            "sample_rate": int(self.cfg_sr.get()),
        }

    def pitch_shift_val(self) -> int:
        return self.cfg_pitch.get()

    # ═══════════════════════════════════════════════════════════════════
    #  Task management
    # ═══════════════════════════════════════════════════════════════════

    def _next_task_id(self) -> int:
        self._task_id_counter += 1
        return self._task_id_counter

    def _add_task(self, config: Optional[Dict] = None):
        """Add a new task to the queue with current (or given) parameters."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return
        if not os.path.isfile(self.source_path.get()):
            messagebox.showerror("Error", "Source file does not exist.")
            return

        cfg = config or self._get_current_config()
        if cfg["model"] == "No models available":
            messagebox.showwarning("Warning", "No voice model selected.")
            return

        task = {
            "id": self._next_task_id(),
            "source": self.source_path.get(),
            "config": cfg,
            "status": STATUS_QUEUED,
            "output_path": None,
            "elapsed": None,
            "error": None,
            "future": None,
            "cancel_flag": threading.Event(),
        }
        self._tasks.append(task)
        self._refresh_task_tree()

    def _duplicate_last_task(self):
        """Duplicate the last task's config into a new task."""
        if not self._tasks:
            messagebox.showinfo("Info", "No tasks to duplicate.")
            return
        last_cfg = self._tasks[-1]["config"].copy()
        # Apply to UI so user can tweak
        self.cfg_model.set(last_cfg["model"])
        self.cfg_pitch.set(last_cfg["pitch"])
        self.cfg_f0.set(last_cfg["f0_method"])
        self.cfg_feature.set(last_cfg["feature_extractor"])
        self.cfg_cluster.set(last_cfg["cluster_ratio"])
        self.cfg_device.set(last_cfg["device"])
        self.cfg_sr.set(str(last_cfg["sample_rate"]))
        self._add_task(last_cfg)

    def _start_all_tasks(self):
        """Submit all queued tasks to the thread pool."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return

        queued = [t for t in self._tasks if t["status"] == STATUS_QUEUED]
        if not queued:
            messagebox.showinfo("Info", "No queued tasks to start.")
            return

        for task in queued:
            task["status"] = STATUS_RUNNING
            future = self._executor.submit(self._run_task, task)
            task["future"] = future
            future.add_done_callback(lambda f, t=task: self._on_task_done(t, f))

        self._refresh_task_tree()

    def _cancel_all_tasks(self):
        """Cancel all queued/running tasks."""
        for task in self._tasks:
            if task["status"] in (STATUS_QUEUED, STATUS_RUNNING):
                task["cancel_flag"].set()
                task["status"] = STATUS_CANCELLED
                if task["future"] and not task["future"].done():
                    task["future"].cancel()
        self._refresh_task_tree()

    def _cancel_selected(self):
        """Cancel the selected task."""
        task = self._get_selected_task()
        if task and task["status"] in (STATUS_QUEUED, STATUS_RUNNING):
            task["cancel_flag"].set()
            task["status"] = STATUS_CANCELLED
            if task["future"] and not task["future"].done():
                task["future"].cancel()
            self._refresh_task_tree()

    def _remove_selected(self):
        """Remove the selected task from the list."""
        task = self._get_selected_task()
        if task:
            task["cancel_flag"].set()
            self._tasks.remove(task)
            self._refresh_task_tree()

    def _clear_done_tasks(self):
        """Remove all completed/failed/cancelled tasks."""
        self._tasks = [t for t in self._tasks if t["status"] in (STATUS_QUEUED, STATUS_RUNNING)]
        self._refresh_task_tree()

    def _get_selected_task(self) -> Optional[Dict]:
        """Get the task corresponding to the selected tree row."""
        sel = self._task_tree.selection()
        if not sel:
            return None
        item = self._task_tree.item(sel[0])
        task_id = int(item["values"][0])
        for t in self._tasks:
            if t["id"] == task_id:
                return t
        return None

    def _refresh_task_tree(self):
        """Rebuild the treeview from self._tasks."""
        self._task_tree.delete(*self._task_tree.get_children())
        status_icons = {
            STATUS_QUEUED: "\u23f3",     # ⏳
            STATUS_RUNNING: "\u2699",     # ⚙
            STATUS_DONE: "\u2705",        # ✅
            STATUS_FAILED: "\u274c",      # ❌
            STATUS_CANCELLED: "\u23f9",   # ⏹
        }
        for t in self._tasks:
            cfg = t["config"]
            icon = status_icons.get(t["status"], "")
            time_str = f"{t['elapsed']:.1f}s" if t["elapsed"] is not None else "--"
            self._task_tree.insert("", tk.END, values=(
                t["id"],
                cfg["model"],
                f"{cfg['pitch']:+d}",
                cfg["f0_method"],
                cfg["feature_extractor"],
                f"{icon} {t['status']}",
                time_str,
            ))

    # ═══════════════════════════════════════════════════════════════════
    #  Task execution (runs in worker thread)
    # ═══════════════════════════════════════════════════════════════════

    def _run_task(self, task: Dict):
        """Execute a single comparison task. Runs in thread pool."""
        if task["cancel_flag"].is_set():
            return

        cfg = task["config"]
        source = task["source"]
        t0 = time.time()

        try:
            import numpy as np
            import soundfile as sf

            # Resolve device
            device_str = cfg["device"]
            if device_str == "auto":
                try:
                    import torch
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device_str = "cpu"

            # Find model
            model_path = self._find_model_file(cfg["model"])
            if model_path is None:
                raise FileNotFoundError(f"Model '{cfg['model']}' not found")

            if task["cancel_flag"].is_set():
                return

            # Load inference pipeline
            from training.inference import RVCInference
            pipeline = RVCInference(
                model_path=model_path,
                device=device_str,
                output_sample_rate=cfg["sample_rate"],
                f0_method=cfg["f0_method"],
            )

            if task["cancel_flag"].is_set():
                return

            # Load audio
            audio, sr = sf.read(source)
            if audio.ndim == 2 and audio.shape[1] > 2:
                audio = np.mean(audio, axis=1)

            if task["cancel_flag"].is_set():
                return

            # Convert
            transpose = float(cfg["pitch"])
            duration = len(audio) / sr
            if duration > 30:
                result = pipeline.convert_long_audio(audio, sample_rate=sr, transpose=transpose)
            else:
                result = pipeline.convert(audio, sample_rate=sr, transpose=transpose)

            if task["cancel_flag"].is_set():
                return

            # Save to temp location
            out_dir = os.path.join("/tmp", "soma_comparison")
            os.makedirs(out_dir, exist_ok=True)

            # Auto-name: model_pitch_f0_feature.wav
            pitch_str = f"p{cfg['pitch']:+d}" if cfg["pitch"] != 0 else "p0"
            out_name = f"{cfg['model']}_{pitch_str}_{cfg['f0_method']}_{cfg['feature_extractor']}_t{task['id']}.wav"
            out_path = os.path.join(out_dir, out_name)
            sf.write(out_path, result, cfg["sample_rate"])

            elapsed = time.time() - t0
            task["output_path"] = out_path
            task["elapsed"] = elapsed
            task["status"] = STATUS_DONE

        except Exception as e:
            task["elapsed"] = time.time() - t0
            task["error"] = str(e)
            task["status"] = STATUS_FAILED

    def _on_task_done(self, task: Dict, future: Future):
        """Callback when a task future completes."""
        # Schedule UI update on main thread
        self.after(0, self._refresh_task_tree)
        self.after(0, lambda: self._append_result_log(task))

    def _append_result_log(self, task: Dict):
        """Append a result line to the result text area."""
        cfg = task["config"]
        self._result_text.configure(state=tk.NORMAL)
        if task["status"] == STATUS_DONE:
            line = (
                f"[#{task['id']}] {cfg['model']}  pitch={cfg['pitch']:+d}  "
                f"f0={cfg['f0_method']}  feat={cfg['feature_extractor']}  "
                f"SR={cfg['sample_rate']}  |  {task['elapsed']:.1f}s  |  "
                f"{os.path.basename(task['output_path'])}"
            )
        elif task["status"] == STATUS_FAILED:
            line = f"[#{task['id']}] FAILED: {task.get('error', 'unknown error')}"
        else:
            line = f"[#{task['id']}] {task['status']}"

        self._result_text.insert(tk.END, line + "\n")
        self._result_text.see(tk.END)
        self._result_text.configure(state=tk.DISABLED)

    # ═══════════════════════════════════════════════════════════════════
    #  Playback
    # ═══════════════════════════════════════════════════════════════════

    def _play_selected(self):
        """Play the selected completed task's output audio."""
        task = self._get_selected_task()
        if not task or task["status"] != STATUS_DONE or not task["output_path"]:
            messagebox.showinfo("Info", "Select a completed task to play.")
            return
        if not os.path.isfile(task["output_path"]):
            messagebox.showerror("Error", f"Output file not found:\n{task['output_path']}")
            return

        self._stop_playback()  # Stop any current playback

        cfg = task["config"]
        self.now_playing_var.set(
            f"\U0001f3b5 Playing #{task['id']}: {cfg['model']}  "
            f"pitch={cfg['pitch']:+d}  f0={cfg['f0_method']}  "
            f"({task['elapsed']:.1f}s)"
        )

        self._playback_process = self._open_audio_file(task["output_path"])

    def _ab_switch(self):
        """A/B switch: rapidly toggle between two selected completed tasks.

        Plays the first selected task, then after a short delay plays the second.
        If only one task is selected, plays it. If none, shows info.
        """
        sel = self._task_tree.selection()
        completed_tasks = []
        for item_id in sel:
            vals = self._task_tree.item(item_id)["values"]
            task_id = int(vals[0])
            for t in self._tasks:
                if t["id"] == task_id and t["status"] == STATUS_DONE and t["output_path"]:
                    completed_tasks.append(t)
                    break

        if not completed_tasks:
            messagebox.showinfo("Info", "Select one or two completed tasks for A/B switch.")
            return

        def _ab_worker():
            for i, task in enumerate(completed_tasks[:2]):
                label = "A" if i == 0 else "B"
                cfg = task["config"]
                self.after(0, lambda l=label, t=task: self.now_playing_var.set(
                    f"\U0001f500 [{l}] Playing #{t['id']}: {cfg['model']}  pitch={cfg['pitch']:+d}"
                ))
                proc = self._open_audio_file(task["output_path"])
                # Wait for playback to roughly finish (estimate ~duration + 1s buffer)
                try:
                    import soundfile as sf
                    info = sf.info(task["output_path"])
                    wait = min(info.duration + 1.0, 30.0)
                except Exception:
                    wait = 5.0
                time.sleep(wait)
                if proc and proc.poll() is None:
                    proc.terminate()

            self.after(0, lambda: self.now_playing_var.set("A/B switch complete"))

        self._stop_playback()
        threading.Thread(target=_ab_worker, daemon=True).start()

    def _stop_playback(self):
        """Stop any currently playing audio."""
        if self._playback_process and self._playback_process.poll() is None:
            self._playback_process.terminate()
            self._playback_process = None
        self.now_playing_var.set("Playback stopped")

    @staticmethod
    def _open_audio_file(filepath: str) -> Optional[subprocess.Popen]:
        """Open an audio file with the system default player."""
        try:
            if sys.platform == "win32":
                os.startfile(filepath)
                return None
            elif sys.platform == "darwin":
                return subprocess.Popen(["afplay", filepath], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Linux: try xdg-open, fall back to aplay/aplay
                return subprocess.Popen(
                    ["xdg-open", filepath],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  Export
    # ═══════════════════════════════════════════════════════════════════

    def _export_all(self):
        """Export all completed task outputs to a user-chosen directory."""
        done_tasks = [t for t in self._tasks if t["status"] == STATUS_DONE and t["output_path"]]
        if not done_tasks:
            messagebox.showinfo("Info", "No completed results to export.")
            return

        target_dir = filedialog.askdirectory(
            title="Export Results To", initialdir=self._last_dir
        )
        if not target_dir:
            return

        self._last_dir = target_dir
        _save_settings({"comparison_last_dir": target_dir})

        copied = 0
        for task in done_tasks:
            cfg = task["config"]
            src = task["output_path"]
            pitch_str = f"pitch{cfg['pitch']:+d}" if cfg["pitch"] != 0 else "pitch0"
            name = f"{cfg['model']}_{pitch_str}_{cfg['f0_method']}_{cfg['feature_extractor']}_t{task['id']}.wav"
            dst = os.path.join(target_dir, name)
            try:
                import shutil
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                pass

        messagebox.showinfo(
            "Export Complete",
            f"Exported {copied}/{len(done_tasks)} result(s) to:\n{target_dir}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Config save / load
    # ═══════════════════════════════════════════════════════════════════

    def _save_config(self):
        """Save current task configs to a JSON file."""
        configs = [t["config"] for t in self._tasks]
        if not configs:
            configs = [self._get_current_config()]

        path = filedialog.asksaveasfilename(
            title="Save Comparison Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialdir=self._last_dir,
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"tasks": configs, "source": self.source_path.get()}, f,
                          ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Config saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def _load_config(self):
        """Load task configs from a JSON file."""
        path = filedialog.askopenfilename(
            title="Load Comparison Config",
            filetypes=[("JSON files", "*.json")],
            initialdir=self._last_dir,
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            configs = data.get("tasks", [])
            source = data.get("source", "")

            if source and os.path.isfile(source):
                self.source_path.set(source)
                self._update_file_info(source)

            for cfg in configs:
                self._add_task(cfg)

            messagebox.showinfo("Loaded", f"Loaded {len(configs)} task(s) from config.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")
