"""
SOMA 日志系统

提供统一的日志管理，支持：
- 控制台和文件双输出
- 按天滚动的日志文件
- 模块级别的日志控制
- 灵活的日志格式配置

使用方式:
    from src.utils.logger import get_logger, setup_logging

    # 设置日志系统（应用启动时调用一次）
    setup_logging(level="DEBUG")

    # 获取模块 logger
    logger = get_logger(__name__)
    logger.info("This is an info message")

    # 在模块中直接使用
    from src.utils.logger import logger
    logger.info("Module-level logging")
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from src.config import LoggingDefaults, get_config


# 日志级别映射
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# 全局 logger 实例
_logger: Optional[logging.Logger] = None
_module_loggers: Dict[str, logging.Logger] = {}


def get_log_level(level_str: str) -> int:
    """将字符串日志级别转换为 int"""
    return LOG_LEVELS.get(level_str.upper(), logging.INFO)


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m',
    }

    def __init__(self, fmt: str, datefmt: str):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "soma.log",
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    backup_count: int = 7,
    max_bytes: int = 10 * 1024 * 1024,
    config: Optional[object] = None
) -> None:
    """
    设置全局日志系统

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 日志目录，默认 ~/.soma/logs
        log_file: 日志文件名
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        format_string: 日志格式字符串
        date_format: 日期格式字符串
        backup_count: 保留的日志文件数量
        max_bytes: 单个日志文件最大大小
        config: 配置对象（优先使用）
    """
    global _logger

    # 优先使用配置对象
    if config is None:
        try:
            config = get_config()
            log_config = config.get_section("logging")
        except Exception:
            log_config = None

        if log_config:
            level = log_config.level if not level else level
            log_dir = log_dir or str(Path(log_config.log_dir).expanduser())
            log_file = log_file or log_config.log_file_name
            console_output = console_output if console_output else log_config.console_output
            file_output = file_output if file_output else log_config.file_output
            format_string = format_string or log_config.format
            date_format = date_format or log_config.date_format
            backup_count = backup_count if backup_count != 7 else log_config.backup_count

    # 创建根 logger
    logger = logging.getLogger("soma")
    logger.setLevel(get_log_level(level))
    logger.handlers.clear()

    # 日志格式
    fmt = format_string or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = date_format or "%Y-%m-%d %H:%M:%S"

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(get_log_level(level))
        console_formatter = ColoredFormatter(fmt, datefmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if file_output:
        log_path = Path(log_dir or "~/.soma/logs").expanduser()
        log_path.mkdir(parents=True, exist_ok=True)

        file_path = log_path / log_file

        # 使用 TimedRotatingFileHandler 按天滚动
        file_handler = TimedRotatingFileHandler(
            filename=str(file_path),
            when='midnight',          # 每天午夜滚动
            interval=1,               # 间隔 1 天
            backupCount=backup_count,  # 保留 7 天
            encoding='utf-8',
            delay=False
        )
        file_handler.setLevel(get_log_level(level))

        # 普通格式化器（文件不需要颜色）
        file_formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 保存全局 logger 引用
    _logger = logger

    logger.info("=" * 60)
    logger.info("SOMA 日志系统已初始化")
    logger.info(f"日志级别: {level}")
    logger.info(f"日志目录: {log_path if file_output else 'N/A'}")
    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    获取模块 logger

    Args:
        name: 模块名称，通常使用 __name__

    Returns:
        Logger 实例

    示例:
        logger = get_logger(__name__)
        logger.info("Info message")
        logger.debug("Debug message")
    """
    global _module_loggers

    # 如果模块 logger 已存在，直接返回
    if name in _module_loggers:
        return _module_loggers[name]

    # 确保全局 logger 已初始化
    if _logger is None:
        setup_logging()

    # 创建模块 logger
    logger = logging.getLogger(name)

    # 如果模块没有处理器，使用根 logger 的处理器
    if not logger.handlers and not logger.parent:
        logger = _logger

    # 缓存
    _module_loggers[name] = logger

    return logger


def set_module_level(module: str, level: str) -> None:
    """
    设置模块日志级别

    Args:
        module: 模块名，如 "soma.separators"
        level: 日志级别

    示例:
        set_module_level("soma.separators", "DEBUG")
    """
    logger = logging.getLogger(module)
    logger.setLevel(get_log_level(level))


def get_log_file_path() -> Optional[Path]:
    """获取当前日志文件路径"""
    if _logger is None:
        return None

    for handler in _logger.handlers:
        if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
            return Path(handler.baseFilename)

    return None


# 便捷函数
def debug(message: str, *args, **kwargs) -> None:
    """记录 DEBUG 级别日志"""
    if _logger:
        _logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """记录 INFO 级别日志"""
    if _logger:
        _logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """记录 WARNING 级别日志"""
    if _logger:
        _logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """记录 ERROR 级别日志"""
    if _logger:
        _logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs) -> None:
    """记录 CRITICAL 级别日志"""
    if _logger:
        _logger.critical(message, *args, **kwargs)


# 默认 logger（供其他模块导入使用）
logger = logging.getLogger("soma")
