#!/usr/bin/env python3
"""
SOMA AI Workstation - PyInstaller Launcher

This is the entry point for PyInstaller packaging.
It bootstraps the application by setting up paths and launching the GUI.

Usage (development):
    python launcher.py

Usage (packaged):
    SOMA.exe  (Windows)
    ./SOMA    (Linux/macOS)
"""

import sys
import os


def get_base_path():
    """
    Get the base path for the application.
    - When running from PyInstaller bundle: use sys._MEIPASS
    - When running from source: use the project root
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running from source
        return os.path.dirname(os.path.abspath(__file__))


def setup_paths():
    """Set up Python paths for the application."""
    base_path = get_base_path()

    # Add src/ to Python path
    src_path = os.path.join(base_path, 'src')
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Also add base_path for direct imports
    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    return base_path


def setup_environment(base_path):
    """Set up environment variables for the application."""
    # Set SOMA_HOME for user data
    if 'SOMA_HOME' not in os.environ:
        if sys.platform == 'win32':
            soma_home = os.path.join(os.path.expanduser('~'), '.soma')
        else:
            soma_home = os.path.join(os.path.expanduser('~'), '.soma')
        os.environ['SOMA_HOME'] = soma_home

    # Ensure user directories exist
    soma_home = os.environ.get('SOMA_HOME', '')
    if soma_home:
        for subdir in ['models', 'output', 'configs']:
            path = os.path.join(soma_home, subdir)
            os.makedirs(path, exist_ok=True)


def main():
    """Main entry point."""
    base_path = setup_paths()
    setup_environment(base_path)

    # Import and launch the GUI
    # This import must happen after path setup
    from gui.app import main as gui_main
    gui_main()


if __name__ == '__main__':
    main()
