"""
Thread-safe singleton settings manager for SOMA GUI.

Provides a centralized, thread-safe way to read/write persistent GUI settings
(e.g., last opened directory) to ~/.soma_gui_settings.json.

Usage:
    sm = SettingsManager()
    last_dir = sm.get("separation_last_dir", os.path.expanduser("~"))
    sm.set("separation_last_dir", "/path/to/dir")
"""

import json
import os
import threading
from typing import Any, Optional


class SettingsManager:
    """
    Thread-safe singleton for managing persistent GUI settings.

    All reads/writes are protected by a threading.Lock to prevent
    race conditions when multiple pages or background threads access
    settings concurrently.
    """

    _instance: Optional["SettingsManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "SettingsManager":
        """Ensure only one instance exists (double-checked locking)."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._settings_file = os.path.join(
                        os.path.expanduser("~"), ".soma_gui_settings.json"
                    )
                    inst._lock = threading.Lock()
                    inst._cache: dict = {}
                    inst._loaded = False
                    cls._instance = inst
        return cls._instance

    def _ensure_loaded(self):
        """Load settings from disk if not already loaded. Must hold _lock."""
        if not self._loaded:
            try:
                with open(self._settings_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                self._cache = {}
            self._loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by key, with optional default."""
        with self._lock:
            self._ensure_loaded()
            return self._cache.get(key, default)

    def set(self, key: str, value: Any):
        """Set a setting value and persist to disk."""
        with self._lock:
            self._ensure_loaded()
            self._cache[key] = value
            self._flush()

    def set_many(self, data: dict):
        """Set multiple settings at once and persist to disk."""
        with self._lock:
            self._ensure_loaded()
            self._cache.update(data)
            self._flush()

    def _flush(self):
        """Write current cache to disk. Must hold _lock."""
        try:
            with open(self._settings_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except OSError:
            pass  # Silently ignore save failures

    def reload(self):
        """Force reload from disk (useful for testing)."""
        with self._lock:
            self._loaded = False
            self._ensure_loaded()
