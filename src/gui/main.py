"""
SOMA AI GUI Entry Point

Launch the SOMA AI Cover Workstation desktop application.

Usage:
    python -m gui.main
    or
    python src/gui/main.py
"""

import sys
import os

# Add src to path if needed
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from gui.app import main

if __name__ == "__main__":
    main()
