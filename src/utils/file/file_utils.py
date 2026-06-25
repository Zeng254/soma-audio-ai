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
    获取文件扩展名（小写，不含点）
    
    Args:
        path: 文件路径
        
    Returns:
        文件扩展名（如 "txt", "pdf"）
    """
    if not path:
        return ""
    
    path_obj = Path(path)
    ext = path_obj.suffix.lstrip('.').lower()
    return ext


def ensure_dir(path: str) -> Path:
    """
    确保目录存在，如不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path 对象
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_filename(filename: str, max_length: int = 254) -> str:
    """
    生成安全的文件名
    
    - 移除非法字符（空格替换为下划线）
    - 处理路径遍历符（.. 替换为下划线）
    - 限制长度
    - 保留扩展名
    
    Args:
        filename: 原始文件名
        max_length: 最大长度限制（整个文件名含扩展名）
        
    Returns:
        安全的文件名
    """
    if not filename:
        return "unnamed"
    
    # 空格转下划线
    name = filename.replace(' ', '_')
    
    # 移除非法字符（包括路径分隔符）
    safe_chars = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    name = safe_chars.sub('_', name)
    
    # 处理 ".." - 替换为单个下划线
    name = name.replace('..', '_')
    
    # 分离扩展名（最后一个点）
    if '.' in name:
        last_dot = name.rfind('.')
        name_without_ext = name[:last_dot]
        ext = name[last_dot:]  # 包含点
    else:
        name_without_ext = name
        ext = ""
    
    # 去掉多余的下划线和点
    name_without_ext = name_without_ext.strip('_.')
    
    if not name_without_ext:
        name_without_ext = "unnamed"
    
    # 限制长度（整个文件名含扩展名不超过 max_length）
    max_name_len = max_length - len(ext)
    if len(name_without_ext) > max_name_len:
        name_without_ext = name_without_ext[:max_name_len]
    
    return name_without_ext + ext
    
    if not name:
        name = "unnamed"
    
    # 限制长度（保留扩展名）
    max_name_len = max_length - len(ext)
    if len(name) > max_name_len:
        name = name[:max_name_len]
    
    return name + ext


def ensure_parent_dir(file_path: str) -> Path:
    """
    确保文件的父目录存在
    
    Args:
        file_path: 文件路径
        
    Returns:
        Path 对象
    """
    path_obj = Path(file_path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj
