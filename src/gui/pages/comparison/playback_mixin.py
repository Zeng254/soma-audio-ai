"""
Comparison page - Playback Mixin.

Contains all audio playback methods for A/B comparison.

Required attributes (initialized in ComparisonPage.__init__):
    - _current_player: subprocess.Popen - current playback process
    - _playback_lock: threading.Lock - protects _current_player access
    - volume_var: tk.DoubleVar - volume slider value
    - _tasks: list - list of ComparisonTask dicts (read for completed tasks)
    - _tasks_lock: threading.Lock - protects _tasks list (read access)

Methods provided by this mixin:
    - _play_audio_file(path)
    - _stop_playback()
    - _play_selected_a()
    - _play_selected_b()
    - _ab_switch_play()
    - _play_b_after_a()
"""

import tkinter as tk
from tkinter import messagebox
import os
import subprocess
import sys


class ComparisonPlaybackMixin:
    """Mixin class for comparison page audio playback methods."""

    def _play_audio_file(self, filepath: str):
        """Play audio file using system default player (cross-platform, fix #3).

        Uses subprocess.Popen on all platforms for consistent stop behavior.
        On macOS: afplay (can be killed)
        On Linux: xdg-open
        On Windows: powershell -c (Start-Player) or ffplay if available
        """
        if not os.path.isfile(filepath):
            messagebox.showwarning("Warning", f"File not found:\n{filepath}")
            return

        self._stop_playback()

        try:
            if sys.platform == "darwin":
                # macOS: afplay can be killed via process group
                proc = subprocess.Popen(
                    ["afplay", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif sys.platform == "win32":
                # Windows: use PowerShell to play via Windows Media Player COM
                # This allows us to get a process handle and kill it
                ps_cmd = (
                    f'$player = New-Object System.Media.SoundPlayer("{filepath}"); '
                    f'$player.Play(); '
                    f'Start-Sleep -Seconds 999'
                )
                proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
                )
            else:
                # Linux: xdg-open
                proc = subprocess.Popen(
                    ["xdg-open", filepath],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            with self._playback_lock:
                self._current_player = proc

        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio:\n{e}")

    def _stop_playback(self):
        """Stop current playback (cross-platform, fix #3)."""
        with self._playback_lock:
            if self._current_player is not None:
                try:
                    self._current_player.terminate()
                    self._current_player.wait(timeout=2)
                except Exception:
                    try:
                        self._current_player.kill()
                    except Exception:
                        pass
                self._current_player = None

    def _play_selected_a(self):
        """Play the first selected completed task."""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select a completed task to play.")
            return

        task_id = int(selection[0])
        with self._tasks_lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)

        if not task or task["status"] != "done" or not task["result_path"]:
            messagebox.showwarning("Warning", "Selected task has no completed result.")
            return

        self._play_audio_file(task["result_path"])
        self._log(f"Playing A: Task #{task_id}")

    def _play_selected_b(self):
        """Play the second selected completed task (or first if only one selected)."""
        selection = self.task_tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Select a completed task to play.")
            return

        # Use second selected item if available, otherwise first
        idx = min(1, len(selection) - 1)
        task_id = int(selection[idx])
        with self._tasks_lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)

        if not task or task["status"] != "done" or not task["result_path"]:
            messagebox.showwarning("Warning", "Selected task has no completed result.")
            return

        self._play_audio_file(task["result_path"])
        self._log(f"Playing B: Task #{task_id}")

    def _ab_switch_play(self):
        """A/B switch playback: play first selected, then second after a delay."""
        selection = self.task_tree.selection()
        if len(selection) < 2:
            messagebox.showinfo("Info", "Select exactly 2 completed tasks for A/B comparison.")
            return

        task_a_id = int(selection[0])
        task_b_id = int(selection[1])

        with self._tasks_lock:
            task_a = next((t for t in self._tasks if t["id"] == task_a_id), None)
            task_b = next((t for t in self._tasks if t["id"] == task_b_id), None)

        if not task_a or not task_b:
            messagebox.showwarning("Warning", "Could not find selected tasks.")
            return

        if task_a["status"] != "done" or task_b["status"] != "done":
            messagebox.showwarning("Warning", "Both tasks must be completed.")
            return

        if not task_a["result_path"] or not task_b["result_path"]:
            messagebox.showwarning("Warning", "Result files not found.")
            return

        # Play A first, then B after a delay
        self._play_audio_file(task_a["result_path"])
        self._log(f"A/B Switch: Playing A (Task #{task_a_id})...")

        # Schedule B playback after 5 seconds
        self.safe_after(5000, lambda: self._play_b_after_a(task_b_id, task_b["result_path"]))

    def _play_b_after_a(self, task_id: int, filepath: str):
        """Play task B after A has played (scheduled callback)."""
        self._stop_playback()
        self._play_audio_file(filepath)
        self._log(f"A/B Switch: Playing B (Task #{task_id})...")
