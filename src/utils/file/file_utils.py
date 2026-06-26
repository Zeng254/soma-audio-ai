"""
File utilities for SOMA project.

Provides file path manipulation and validation functions.
"""

import os
import re
from pathlib import Path
from typing import Optional


def get_extension(path: str) -> str:
    """
    Get file extension (lowercase, without dot)
    
    Args:
        path: File path
        
    Returns:
        File extension（e.g. "txt", "pdf"）
    """
    if not path:
        return ""
    
    path_obj = Path(path)
    ext = path_obj.suffix.lstrip('.').lower()
    return ext


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not exists
    
    Args:
        path: Directory path
        
    Returns:
        Path Object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_filename(filename: str, max_length: int = 254) -> str:
    """
    Generate secure filename
    
    - Remove illegal characters (replace spaces with underscores)
    - Process path iterator characters (replace .. with underscore)
    - Limit length
    - Preserve extension
    
    Args:
        filename: Original filename
        max_length: Maximum length limit (entire filename including extension)
        
    Returns:
        Secure filename
    """
    if not filename:
        return "unnamed"
    
    # Replace spaces with underscores
    name = filename.replace(' ', '_')
    
    # Remove illegal characters (including path separators)
    safe_chars = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    name = safe_chars.sub('_', name)
    
    # Process ".." - Replace with single underscore
    name = name.replace('..', '_')
    
    # Separate extension (last dot)
    if '.' in name:
        last_dot = name.rfind('.')
        name_without_ext = name[:last_dot]
        ext = name[last_dot:]  # Contains dot
    else:
        name_without_ext = name
        ext = ""
    
    # Remove extra underscores and dots
    name_without_ext = name_without_ext.strip('_.')
    
    if not name_without_ext:
        name_without_ext = "unnamed"
    
    # Limit length (entire filename including extension not exceeding max_length)
    max_name_len = max_length - len(ext)
    if len(name_without_ext) > max_name_len:
        name_without_ext = name_without_ext[:max_name_len]
    
    return name_without_ext + ext


def ensure_parent_dir(file_path: str) -> Path:
    """
    Ensure file parent directory exists
    
    Args:
        file_path: File path
        
    Returns:
        Path Object
    """
    path_obj = Path(file_path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj
