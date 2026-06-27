"""
Inference Worker Mixin.

Contains all background processing and worker logic for the InferencePage.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import os
import time
from typing import Optional

from gui.utils import open_folder


class InferenceWorkerMixin:
    """Mixin class providing worker/processing methods for InferencePage."""

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Add message to log display (main thread only)."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)

    def _set_stage(self, stage: str):
        """Set the current processing stage (main thread only)."""
        self.stage_var.set(stage)

    # ── Elapsed Timer ──────────────────────────────────────────────────

    def _start_elapsed_timer(self):
        self._start_time = time.time()
        self._tick_elapsed()

    def _tick_elapsed(self):
        if self._start_time is not None and not self._cancel_event.is_set():
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d}")
            self._elapsed_timer_id = self.safe_after(1000, self._tick_elapsed)

    def _stop_elapsed_timer(self):
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
        """
        self.convert_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.status_var.set(status_text)
        self.stage_var.set("")
        self._stop_elapsed_timer()

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

        # Reset cancel event and update UI state
        self._cancel_event.clear()
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
        """Stop the conversion process via cancel event."""
        self._cancel_event.set()
        self._log("Stop requested...")
        # UI will be reset by the worker thread's finally block

    def _conversion_worker(self):
        """Background worker for voice conversion."""
        try:
            import numpy as np
            import soundfile as sf

            # Stage 1: Load model
            self.safe_after(0, lambda: self._set_stage("Loading model"))
            self.safe_after(0, lambda: self.status_var.set("Loading model..."))
            self.safe_after(0, lambda: self.progress_var.set(5))
            self.safe_after(0, lambda: self._log("[1/3] Loading voice model..."))

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

            if self._cancel_event.is_set():
                return

            self.safe_after(0, lambda: self.progress_var.set(20))
            self.safe_after(0, lambda: self._log(f"Model loaded on {device_str}"))

            # Stage 2: Load and preprocess audio
            self.safe_after(0, lambda: self._set_stage("Loading audio"))
            self.safe_after(0, lambda: self.status_var.set("Loading audio..."))
            self.safe_after(0, lambda: self.progress_var.set(30))
            self.safe_after(0, lambda: self._log("[2/3] Loading source audio..."))

            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 2 and audio.shape[1] > 2:
                audio = np.mean(audio, axis=1)  # Downmix to mono

            self.safe_after(0, lambda: self._log(f"Audio loaded: {len(audio)} samples, {sr}Hz"))

            # Optional preprocessing
            if self.separate_vocals.get() or self.dereverb_audio.get():
                self.safe_after(0, lambda: self._set_stage("Preprocessing"))
                self.safe_after(0, lambda: self.status_var.set("Preprocessing audio..."))
                self.safe_after(0, lambda: self.progress_var.set(35))

                try:
                    from separators.audio_separator import AudioSeparator, SeparationMode

                    separator = AudioSeparator()

                    if self.dereverb_audio.get():
                        self.safe_after(0, lambda: self._log("Applying dereverberation..."))
                        audio = separator.dereverb(audio)

                    if self.separate_vocals.get():
                        self.safe_after(0, lambda: self._log("Separating vocals..."))
                        mode = SeparationMode.TWO_STEMS
                        vocals, _ = separator.separate(audio, mode=mode, sample_rate=sr)
                        audio = vocals
                        self.safe_after(0, lambda: self._log("Vocals separated."))
                except ImportError:
                    self.safe_after(0, lambda: self._log("Warning: Preprocessing skipped (missing deps)"))

            if self._cancel_event.is_set():
                return

            # Stage 3: Voice conversion
            self.safe_after(0, lambda: self._set_stage("Voice conversion"))
            self.safe_after(0, lambda: self.status_var.set("Converting voice..."))
            self.safe_after(0, lambda: self.progress_var.set(50))
            self.safe_after(0, lambda: self._log("[3/3] Running voice conversion..."))

            transpose = float(self.pitch_shift.get())

            # Use chunked conversion for long audio
            duration_sec = len(audio) / sr
            if duration_sec > 30:
                self.safe_after(0, lambda: self._log(f"Long audio ({duration_sec:.1f}s), using chunked mode..."))
                result = pipeline.convert_long_audio(
                    audio, sample_rate=sr, transpose=transpose
                )
            else:
                result = pipeline.convert(audio, sample_rate=sr, transpose=transpose)

            if self._cancel_event.is_set():
                return

            # Save output
            self.safe_after(0, lambda: self._set_stage("Saving output"))
            self.safe_after(0, lambda: self.status_var.set("Saving output..."))
            self.safe_after(0, lambda: self.progress_var.set(90))

            output_dir = os.path.dirname(self.output_path.get())
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            sf.write(self.output_path.get(), result, output_sr)

            self.safe_after(0, lambda: self.progress_var.set(100))
            self.safe_after(0, lambda: self._log(f"Output saved: {self.output_path.get()}"))
            self.safe_after(0, self._conversion_complete)

        except ImportError as e:
            err_msg = f"Missing dependency: {e}"
            self.safe_after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.safe_after(0, lambda: self.status_var.set("Error"))
            self.safe_after(0, self._conversion_error)

        except Exception as e:
            err_msg = str(e)
            self.safe_after(0, lambda: self._log(f"ERROR: {err_msg}"))
            self.safe_after(0, lambda: self.status_var.set("Error"))
            self.safe_after(0, self._conversion_error)

        finally:
            # Always reset UI state when worker exits (fix #5)
            if self._cancel_event.is_set():
                self.safe_after(0, lambda: self._reset_ui_after_processing("Stopped"))
                self.safe_after(0, lambda: self._log("Conversion stopped by user."))

    def _conversion_complete(self):
        """Handle conversion completion."""
        self._reset_ui_after_processing("Completed")

        self._log("")
        self._log("Conversion completed successfully!")

        # Show completion dialog
        output_dir = os.path.dirname(self.output_path.get())
        result = messagebox.askyesno(
            "Conversion Complete",
            f"Voice conversion completed successfully!\n\n"
            f"Output saved to:\n{self.output_path.get()}\n\n"
            f"Open output folder?"
        )
        if result and output_dir:
            open_folder(output_dir)

    def _conversion_error(self):
        """Handle conversion error."""
        self._reset_ui_after_processing("Error")

        messagebox.showerror(
            "Conversion Failed",
            "Voice conversion failed.\n\n"
            "Please check the log for details.\n"
            "Common issues: missing dependencies, model not found, or insufficient memory."
        )
