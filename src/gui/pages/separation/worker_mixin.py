"""
SeparationPage Worker Mixin - background processing, state management, and completion handling.

Contains all worker thread logic, elapsed timer, logging, UI reset,
and separation start/stop/complete/error methods.

Required attributes (initialized in SeparationPage.__init__):
    - _cancel_event: threading.Event - cancel signal
    - _processing_thread: Optional[threading.Thread] - background thread reference
    - _start_time: Optional[float] - processing start timestamp
    - _elapsed_timer_id: Optional[str] - timer ID for elapsed display
    - source_path: tk.StringVar - source audio file path (read)
    - output_dir: tk.StringVar - output directory path (read)
    - separation_mode: tk.StringVar - separation mode (read)
    - backend: tk.StringVar - backend selection (read)
    - dereverb_enabled: tk.BooleanVar - dereverb toggle (read)
    - output_format: tk.StringVar - output format (read)
    - progress_var: tk.DoubleVar - progress bar value (write)
    - status_var: tk.StringVar - status text (write)
    - elapsed_var: tk.StringVar - elapsed time display (write)
    - _last_directory: str - remembered directory (read/write)
    - _settings: SettingsManager - settings manager instance

Methods provided by this mixin:
    - _log(message)
    - _start_elapsed_timer()
    - _tick_elapsed()
    - _stop_elapsed_timer()
    - _reset_ui_after_processing()
    - _start_separation()
    - _stop_separation()
    - _separation_worker(source_path, output_dir, mode, backend, dereverb, fmt)
    - _separation_complete(output_path)
    - _separation_error(error_msg)
    - _open_output_folder()
"""

import tkinter as tk
from tkinter import messagebox
import threading
import os
import time
from typing import Optional, List

from gui.utils import open_folder


class SeparationWorkerMixin:
    """Mixin providing worker/processing methods for SeparationPage."""

    # Class-level constants (accessed via self when mixed in)
    MODES: dict
    BACKENDS: dict
    OUTPUT_FORMATS: dict

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
