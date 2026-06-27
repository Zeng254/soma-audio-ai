"""
Runtime hook for torch DLL loading in PyInstaller bundle.

This hook is executed at runtime before the main application starts.
It adds torch/lib to the DLL search path on Windows to ensure
torch can find its dependencies.
"""

import sys
import os


def setup_torch_dlls():
    """Set up torch DLL search path for PyInstaller bundle."""
    if not getattr(sys, 'frozen', False):
        # Not running as PyInstaller bundle, no setup needed
        return

    # Get the base path of the PyInstaller bundle
    base_path = sys._MEIPASS

    # Torch DLLs are typically in torch/lib or torch/bin
    torch_lib_paths = [
        os.path.join(base_path, 'torch', 'lib'),
        os.path.join(base_path, 'torch', 'bin'),
        os.path.join(base_path, 'torch'),
    ]

    # Add torch paths to DLL search path (Windows)
    if sys.platform == 'win32':
        # Add to PATH environment variable
        current_path = os.environ.get('PATH', '')
        for torch_path in torch_lib_paths:
            if os.path.isdir(torch_path) and torch_path not in current_path:
                os.environ['PATH'] = torch_path + os.pathsep + current_path

        # Also use os.add_dll_directory for Python 3.8+
        if hasattr(os, 'add_dll_directory'):
            for torch_path in torch_lib_paths:
                if os.path.isdir(torch_path):
                    try:
                        os.add_dll_directory(torch_path)
                    except OSError:
                        pass

    # For Linux/macOS, add to LD_LIBRARY_PATH / DYLD_LIBRARY_PATH
    elif sys.platform.startswith('linux'):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        for torch_path in torch_lib_paths:
            if os.path.isdir(torch_path) and torch_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = torch_path + os.pathsep + current_ld_path

    elif sys.platform == 'darwin':
        current_dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        for torch_path in torch_lib_paths:
            if os.path.isdir(torch_path) and torch_path not in current_dyld_path:
                os.environ['DYLD_LIBRARY_PATH'] = torch_path + os.pathsep + current_dyld_path


# Execute the setup
setup_torch_dlls()
