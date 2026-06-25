"""
SOMA Logging system

Provides unified logging management, supports:
- Console and file dual output
- Daily rotating log files
- Module-level logging control
- Flexible logging format configuration

Usage:
    from src.utils.logger import get_logger, setup_logging

    # Set logging system (called once at application start)
    setup_logging(level="DEBUG")

    # GetModule logger
    logger = get_logger(__name__)
    logger.info("This is an info message")

    # Use directly in module
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


# Logging level mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Global logger instance
_logger: Optional[logging.Logger] = None
_module_loggers: Dict[str, logging.Logger] = {}


def get_log_level(level_str: str) -> int:
    """Convert string logging level to int"""
    return LOG_LEVELS.get(level_str.upper(), logging.INFO)


class ColoredFormatter(logging.Formatter):
    """Colored logging formatter"""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Purple
        'RESET': '\033[0m',
    }

    def __init__(self, fmt: str, datefmt: str):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Add color
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
    Set global logging system

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Logging directory, default ~/.soma/logs
        log_file: Log file name
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_string: Logging formatString
        date_format: DateFormatString
        backup_count: Keep log file count
        max_bytes: Single log file maximum size
        config: Configuration object (preferred)
    """
    global _logger

    # Prefer configuration object
    if config is None:
        try:
            config = get_config()
            log_config = config.get_section("logging")
        except Exception:
            log_config = None

        if log_config:
            level = level if level != "INFO" else log_config.level
            log_dir = log_dir or str(Path(log_config.log_dir).expanduser())
            log_file = log_file or log_config.log_file_name
            console_output = console_output if console_output else log_config.console_output
            file_output = file_output if file_output else log_config.file_output
            format_string = format_string or log_config.format
            date_format = date_format or log_config.date_format
            backup_count = backup_count if backup_count != 7 else log_config.backup_count

    # Create root logger
    logger = logging.getLogger("soma")
    logger.setLevel(get_log_level(level))
    logger.handlers.clear()

    # Logging format
    fmt = format_string or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = date_format or "%Y-%m-%d %H:%M:%S"

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(get_log_level(level))
        console_formatter = ColoredFormatter(fmt, datefmt)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        log_path = Path(log_dir or "~/.soma/logs").expanduser()
        log_path.mkdir(parents=True, exist_ok=True)

        file_path = log_path / log_file

        # Use TimedRotatingFileHandler for daily rotation
        file_handler = TimedRotatingFileHandler(
            filename=str(file_path),
            when='midnight',          # Rotate at midnight every day
            interval=1,               # Interval 1 day
            backupCount=backup_count,  # Keep 7 days
            encoding='utf-8',
            delay=False
        )
        file_handler.setLevel(get_log_level(level))

        # Normal formatter (file does not need color)
        file_formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Save global logger reference
    _logger = logger

    logger.info("=" * 60)
    logger.info("SOMA logging system initialized")
    logger.info(f"Logging level: {level}")
    logger.info(f"loggingDirectory: {log_path if file_output else 'N/A'}")
    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """
    GetModule logger

    Args:
        name: Module name, usually uses __name__

    Returns:
        Logger Instance

    Example:
        logger = get_logger(__name__)
        logger.info("Info message")
        logger.debug("Debug message")
    """
    global _module_loggers

    # If module logger already exists, return directly
    if name in _module_loggers:
        return _module_loggers[name]

    # Ensure global logger is initialized
    if _logger is None:
        setup_logging()

    # CreateModule logger
    logger = logging.getLogger(name)

    # If module has no handler, use root logger handler
    if not logger.handlers and not logger.parent:
        logger = _logger

    # Cache
    _module_loggers[name] = logger

    return logger


def set_module_level(module: str, level: str) -> None:
    """
    SetModuleLogging level

    Args:
        module: Module name, e.g. "soma.separators"
        level: Logging level

    Example:
        set_module_level("soma.separators", "DEBUG")
    """
    logger = logging.getLogger(module)
    logger.setLevel(get_log_level(level))


def get_log_file_path() -> Optional[Path]:
    """Get current logging file path"""
    if _logger is None:
        return None

    for handler in _logger.handlers:
        if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
            return Path(handler.baseFilename)

    return None


# Convenience function
def debug(message: str, *args, **kwargs) -> None:
    """Record DEBUG level logging"""
    if _logger:
        _logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """Record INFO level logging"""
    if _logger:
        _logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """Record WARNING level logging"""
    if _logger:
        _logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """Record ERROR level logging"""
    if _logger:
        _logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs) -> None:
    """Record CRITICAL level logging"""
    if _logger:
        _logger.critical(message, *args, **kwargs)


# Default logger (for other module import use)
logger = logging.getLogger("soma")
