"""
Common utility functions shared across SOMA GUI pages.

Provides:
- open_folder: Open a directory in the system file explorer
- safe_after: Schedule a callback on the tkinter main thread with widget alive check
- widget_alive: Check if a tkinter widget still exists and is not destroyed
"""

import os
import subprocess
import sys
from typing import Optional

import tkinter as tk


def open_folder(folder_path: str) -> bool:
    """
    Open a folder in the system's default file explorer.

    Args:
        folder_path: Absolute path to the folder to open.

    Returns:
        True if the command was launched successfully, False otherwise.
    """
    if not folder_path or not os.path.isdir(folder_path):
        return False

    folder = os.path.normpath(folder_path)
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["explorer", folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "darwin":
            subprocess.Popen(
                ["open", folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["xdg-open", folder],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False


def open_audio_file(filepath: str) -> Optional[subprocess.Popen]:
    """
    Open an audio file with the system default player.

    Uses subprocess.Popen on all platforms (including Windows) so that
    the process handle can be terminated later if needed.

    Args:
        filepath: Path to the audio file.

    Returns:
        A subprocess.Popen handle, or None if playback could not start.
    """
    if not filepath or not os.path.isfile(filepath):
        return None

    try:
        if sys.platform == "win32":
            # Use cmd /c start to get a Popen handle on Windows
            return subprocess.Popen(
                ["cmd", "/c", "start", "", filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "darwin":
            return subprocess.Popen(
                ["afplay", filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            return subprocess.Popen(
                ["xdg-open", filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        return None


def widget_alive(widget: tk.Widget) -> bool:
    """
    Check if a tkinter widget still exists and has not been destroyed.

    This is used as a guard in after() callbacks to prevent crashes
    when a callback fires after its target widget has been destroyed
    (e.g., when switching pages).

    Args:
        widget: The tkinter widget to check.

    Returns:
        True if the widget is still alive, False otherwise.
    """
    try:
        return widget.winfo_exists()
    except Exception:
        return False


def safe_after(widget: tk.Widget, delay_ms: int, callback):
    """
    Schedule a callback on the tkinter main thread with a widget alive guard.

    If the widget has been destroyed before the callback fires, the callback
    is silently skipped. This prevents "can't invoke 'winfo_exists' command"
    errors when pages are switched while background threads are running.

    Args:
        widget: The tkinter widget to schedule on (must be alive at call time).
        delay_ms: Delay in milliseconds before executing the callback.
        callback: The callable to execute.

    Returns:
        The after() ID, or None if the widget is already destroyed.
    """
    if not widget_alive(widget):
        return None

    def _guarded():
        if widget_alive(widget):
            try:
                callback()
            except Exception:
                pass  # Silently ignore errors in guarded callbacks

    return widget.after(delay_ms, _guarded)
