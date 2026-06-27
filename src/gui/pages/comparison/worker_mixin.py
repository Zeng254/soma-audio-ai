"""
Comparison page - Worker Mixin.

Contains all task management, execution, config save/load, and export methods.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

from gui.utils import (
    SettingsManager, open_folder,
    STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED,
    STATUS_CANCELLED, STATUS_DISPLAY,
    ComparisonTask,
    DEFAULT_MAX_WORKERS_CPU, DEFAULT_MAX_WORKERS_GPU,
    SETTING_KEY_MAX_WORKERS, SETTING_KEY_DEVICE_TYPE,
)


class ComparisonWorkerMixin:
    """Mixin class for comparison page worker/task management methods."""

    def _get_max_workers(self) -> int:
        """Get configured max workers for thread pool (fix #6).

        Falls back to defaults if stored value is invalid (<=0 or non-numeric).
        Logs a warning when fallback is triggered.
        """
        logger = logging.getLogger(__name__)
        try:
            stored = self._settings.get(SETTING_KEY_MAX_WORKERS, None)
            if stored is not None:
                value = int(stored)
                if value <= 0:
                    logger.warning(
                        "Invalid max_workers=%d (<=0), falling back to default", value
                    )
                else:
                    return value
        except (ValueError, TypeError) as e:
            logger.warning(
                "Cannot parse max_workers setting: %s, falling back to default", e
            )
        # Auto-detect based on device
        device_type = self._settings.get(SETTING_KEY_DEVICE_TYPE, "auto")
        if device_type == "cuda":
            return DEFAULT_MAX_WORKERS_GPU
        return DEFAULT_MAX_WORKERS_CPU

    # ── Task Management (thread-safe, fix #6) ──────────────────────────

    def _get_current_config(self) -> Dict:
        """Get current parameter configuration."""
        return {
            "model": self.selected_model.get(),
            "pitch": self.pitch_shift.get(),
            "feature_extractor": self.feature_extractor.get(),
            "f0_method": self.f0_method.get(),
            "device": self.device.get(),
            "sample_rate": self.output_sample_rate.get(),
            "cluster_ratio": round(self.cluster_ratio.get(), 2),
        }

    def _add_task(self):
        """Add a new task with current configuration."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return

        config = self._get_current_config()
        if config["model"] == "No models available":
            messagebox.showwarning("Warning", "No voice models available.")
            return

        with self._tasks_lock:
            self._task_counter += 1
            task_id = self._task_counter
            task: ComparisonTask = {
                "id": task_id,
                "config": config,
                "status": STATUS_QUEUED,
                "result_path": None,
                "error": None,
                "duration": None,
                "cancel_flag": threading.Event(),
                "uuid": uuid.uuid4().hex[:8],  # fix #9: unique ID for filenames
            }
            self._tasks.append(task)

        # Insert into treeview and track in item map (fix #4)
        iid = str(task_id)
        self.task_tree.insert("", tk.END, iid=iid, values=(
            task_id,
            config["model"],
            f"{config['pitch']:+d}",
            config["f0_method"],
            config["feature_extractor"],
            STATUS_DISPLAY[STATUS_QUEUED],
            "--",
        ))
        self._tree_item_map[task_id] = iid

        self._update_task_count()
        self._log(f"Task #{task_id} added: model={config['model']}, pitch={config['pitch']:+d}, "
                  f"f0={config['f0_method']}, feature={config['feature_extractor']}")

    def _duplicate_last_task(self):
        """Duplicate the last task's configuration for quick setup."""
        with self._tasks_lock:
            if not self._tasks:
                messagebox.showinfo("Info", "No tasks to duplicate.")
                return
            last_config = self._tasks[-1]["config"].copy()

        # Apply last config to current controls
        self.selected_model.set(last_config["model"])
        self.pitch_shift.set(last_config["pitch"])
        self.feature_extractor.set(last_config["feature_extractor"])
        self.f0_method.set(last_config["f0_method"])
        self.device.set(last_config["device"])
        self.output_sample_rate.set(last_config["sample_rate"])
        self.cluster_ratio.set(last_config["cluster_ratio"])

        self._log(f"Duplicated config from last task: model={last_config['model']}")

    def _remove_selected_task(self):
        """Remove selected task from queue."""
        selection = self.task_tree.selection()
        if not selection:
            return

        for item_id in selection:
            task_id = int(item_id)
            with self._tasks_lock:
                task = next((t for t in self._tasks if t["id"] == task_id), None)
                if task:
                    if task["status"] == STATUS_RUNNING:
                        task["cancel_flag"].set()
                    self._tasks.remove(task)
            try:
                self.task_tree.delete(item_id)
            except tk.TclError:
                pass
            # Clean up item map (fix #4)
            self._tree_item_map.pop(task_id, None)

        self._update_task_count()

    def _clear_done_tasks(self):
        """Clear all completed/failed/cancelled tasks."""
        with self._tasks_lock:
            done_tasks = [t for t in self._tasks if t["status"] in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)]
            for task in done_tasks:
                self._tasks.remove(task)
                task_id = task["id"]
                try:
                    self.task_tree.delete(str(task_id))
                except tk.TclError:
                    pass
                # Clean up item map (fix #4)
                self._tree_item_map.pop(task_id, None)

        self._update_task_count()

    def _update_task_count(self):
        """Update task count label."""
        with self._tasks_lock:
            total = len(self._tasks)
            done = sum(1 for t in self._tasks if t["status"] == STATUS_DONE)
            running = sum(1 for t in self._tasks if t["status"] == STATUS_RUNNING)
        self.task_count_label.configure(text=f"{total} tasks ({done} done, {running} running)")

    def _update_task_in_tree(self, task_id: int, task: ComparisonTask):
        """Update a task's display in the treeview (main thread, incremental fix #4).

        Uses the item mapping to update only the changed row instead of
        deleting and re-inserting all rows.
        """
        try:
            iid = str(task_id)
            values = (
                task_id,
                task["config"]["model"],
                f"{task['config']['pitch']:+d}",
                task["config"]["f0_method"],
                task["config"]["feature_extractor"],
                STATUS_DISPLAY.get(task["status"], task["status"]),
                f"{task['duration']:.1f}s" if task["duration"] else "--",
            )
            if iid in self._tree_item_map:
                # Incremental update: just update the existing row
                self.task_tree.item(iid, values=values)
            else:
                # Row doesn't exist yet, insert it
                new_iid = self.task_tree.insert("", tk.END, iid=iid, values=values)
                self._tree_item_map[task_id] = new_iid
        except tk.TclError:
            pass

    # ── Batch Operations ───────────────────────────────────────────────

    def _start_all_tasks(self):
        """Start all queued tasks."""
        if not self.source_path.get():
            messagebox.showwarning("Warning", "Please select a source audio file first.")
            return

        if not self.output_dir.get():
            messagebox.showwarning("Warning", "Please specify an output directory first.")
            return

        try:
            os.makedirs(self.output_dir.get(), exist_ok=True)
        except OSError as e:
            messagebox.showerror("Error", f"Cannot create output directory:\n{e}")
            return

        with self._tasks_lock:
            queued = [t for t in self._tasks if t["status"] == STATUS_QUEUED]

        if not queued:
            messagebox.showinfo("Info", "No queued tasks to start.")
            return

        self._processing = True
        self._start_time = time.time()
        self._tick_elapsed()

        for task in queued:
            task["cancel_flag"].clear()
            task["status"] = STATUS_RUNNING
            self.safe_after(0, lambda t=task: self._update_task_in_tree(t["id"], t))
            self._executor.submit(self._run_task, task)

        self._update_task_count()
        self._log(f"Started {len(queued)} task(s)")

    def _cancel_all_tasks(self):
        """Cancel all running tasks."""
        with self._tasks_lock:
            running = [t for t in self._tasks if t["status"] == STATUS_RUNNING]

        for task in running:
            task["cancel_flag"].set()

        self._log(f"Cancelled {len(running)} task(s)")

    # ── Task Execution ─────────────────────────────────────────────────

    def _run_task(self, task: Dict):
        """Execute a single conversion task (runs in thread pool)."""
        task_id = task["id"]
        config = task["config"]
        cancel_flag = task["cancel_flag"]
        start_time = time.time()

        try:
            import numpy as np
            import soundfile as sf
            from training.inference import RVCInference

            # Find model file
            model_path = self._find_model_file(config["model"])
            if model_path is None:
                raise FileNotFoundError(f"Model '{config['model']}' not found")

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Determine device
            device_str = config["device"]
            if device_str == "auto":
                try:
                    import torch
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device_str = "cpu"

            output_sr = int(config["sample_rate"])

            pipeline = RVCInference(
                model_path=model_path,
                device=device_str,
                output_sample_rate=output_sr,
                f0_method=config["f0_method"],
            )

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Load audio
            audio, sr = sf.read(self.source_path.get())
            if audio.ndim == 2 and audio.shape[1] > 2:
                audio = np.mean(audio, axis=1)

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Convert
            transpose = float(config["pitch"])
            result = pipeline.convert(audio, sample_rate=sr, transpose=transpose)

            if cancel_flag.is_set():
                raise InterruptedError("Cancelled")

            # Save with task ID in filename (fix #9)
            source_name = os.path.splitext(os.path.basename(self.source_path.get()))[0]
            output_filename = (
                f"{source_name}_t{task_id}_{config['model']}_"
                f"pitch{config['pitch']:+d}_{config['f0_method']}_"
                f"{config['feature_extractor']}.wav"
            )
            output_path = os.path.join(self.output_dir.get(), output_filename)
            sf.write(output_path, result, output_sr)

            task["status"] = STATUS_DONE
            task["result_path"] = output_path
            task["duration"] = time.time() - start_time

            self.safe_after(0, lambda: self._log(
                f"Task #{task_id} completed in {task['duration']:.1f}s"
            ))

        except InterruptedError:
            task["status"] = STATUS_CANCELLED
            task["duration"] = time.time() - start_time
            self.safe_after(0, lambda: self._log(f"Task #{task_id} cancelled"))

        except Exception as e:
            task["status"] = STATUS_FAILED
            task["error"] = str(e)
            task["duration"] = time.time() - start_time
            self.safe_after(0, lambda: self._log(f"Task #{task_id} failed: {e}"))

        finally:
            # Update treeview and count from main thread
            self.safe_after(0, lambda: self._update_task_in_tree(task_id, task))
            self.safe_after(0, self._update_task_count)
            self.safe_after(0, self._check_all_done)

    def _check_all_done(self):
        """Check if all tasks are done and stop the timer."""
        with self._tasks_lock:
            all_done = all(
                t["status"] in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)
                for t in self._tasks
            )
        if all_done and self._processing:
            self._processing = False
            if self._start_time:
                elapsed = time.time() - self._start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                self.elapsed_var.set(f"{minutes}:{seconds:02d} (all done)")

            with self._tasks_lock:
                done_count = sum(1 for t in self._tasks if t["status"] == STATUS_DONE)
            self._log(f"All tasks finished. {done_count} succeeded.")

            if done_count > 0:
                result = messagebox.askyesno(
                    "Comparison Complete",
                    f"All comparison tasks finished!\n\n"
                    f"{done_count} result(s) saved to:\n{self.output_dir.get()}\n\n"
                    f"Open output folder?"
                )
                if result:
                    self._open_output_folder()

    # ── Export ──────────────────────────────────────────────────────────

    def _export_all_results(self):
        """Export all completed results to a chosen directory."""
        with self._tasks_lock:
            done_tasks = [t for t in self._tasks if t["status"] == STATUS_DONE and t["result_path"]]

        if not done_tasks:
            messagebox.showinfo("Info", "No completed results to export.")
            return

        target_dir = filedialog.askdirectory(
            title="Select Export Directory",
            initialdir=self._last_directory,
        )
        if not target_dir:
            return

        import shutil
        exported = 0
        for task in done_tasks:
            source_path = task["result_path"]
            if source_path and os.path.isfile(source_path):
                filename = os.path.basename(source_path)
                dest_path = os.path.join(target_dir, filename)
                try:
                    shutil.copy2(source_path, dest_path)
                    exported += 1
                except Exception as e:
                    self._log(f"Export failed for {filename}: {e}")

        self._log(f"Exported {exported} file(s) to {target_dir}")
        messagebox.showinfo("Export Complete", f"Exported {exported} file(s) to:\n{target_dir}")

    def _open_output_folder(self):
        """Open output folder using common utility."""
        if self.output_dir.get() and os.path.exists(self.output_dir.get()):
            if not open_folder(self.output_dir.get()):
                messagebox.showwarning("Warning", "Could not open folder.")
        else:
            messagebox.showwarning("Warning", "Output folder does not exist.")

    # ── Config Save/Load ───────────────────────────────────────────────

    def _save_config(self):
        """Save current task configurations to JSON."""
        with self._tasks_lock:
            configs = [t["config"].copy() for t in self._tasks]

        if not configs:
            messagebox.showinfo("Info", "No tasks to save.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Comparison Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self._last_directory,
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({"tasks": configs, "source": self.source_path.get()}, f, indent=2)
            self._log(f"Config saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{e}")

    def _load_config(self):
        """Load task configurations from JSON."""
        filepath = filedialog.askopenfilename(
            title="Load Comparison Config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self._last_directory,
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            configs = data.get("tasks", [])
            source = data.get("source", "")

            if source and os.path.isfile(source):
                self.source_path.set(source)

            for config in configs:
                with self._tasks_lock:
                    self._task_counter += 1
                    task_id = self._task_counter
                    task: ComparisonTask = {
                        "id": task_id,
                        "config": config,
                        "status": STATUS_QUEUED,
                        "result_path": None,
                        "error": None,
                        "duration": None,
                        "cancel_flag": threading.Event(),
                        "uuid": uuid.uuid4().hex[:8],
                    }
                    self._tasks.append(task)

                iid = str(task_id)
                self.task_tree.insert("", tk.END, iid=iid, values=(
                    task_id,
                    config.get("model", "?"),
                    f"{config.get('pitch', 0):+d}",
                    config.get("f0_method", "?"),
                    config.get("feature_extractor", "?"),
                    STATUS_DISPLAY[STATUS_QUEUED],
                    "--",
                ))
                self._tree_item_map[task_id] = iid

            self._update_task_count()
            self._log(f"Loaded {len(configs)} task(s) from {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")

    # ── Logging ────────────────────────────────────────────────────────

    def _log(self, message: str):
        """Add message to log display (main thread only)."""
        self.log_display.configure(state=tk.NORMAL)
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END)
        self.log_display.configure(state=tk.DISABLED)

    # ── Elapsed Timer ──────────────────────────────────────────────────

    def _tick_elapsed(self):
        """Update elapsed time display."""
        if self._start_time is not None and self._processing:
            elapsed = time.time() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.elapsed_var.set(f"{minutes}:{seconds:02d}")
            self._elapsed_timer_id = self.safe_after(1000, self._tick_elapsed)
