"""
Logging utilities for the CommKit library.

This module provides a unified, colorized logging interface for monitoring
signal processing workflows and debugging complex system failures.
"""

import logging
import sys


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter providing ANSI-colored output based on log levels.

    This formatter enhances readability by using distinct colors for different
    severities (e.g., Cyan for DEBUG, Red for ERROR).
    """

    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s [%(levelname)s] [%(name)s/%(filename)s] %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with ANSI color codes.

        Parameters
        ----------
        record : logging.LogRecord
            The log record containing the message and metadata.

        Returns
        -------
        str
            The formatted log message with embedded ANSI escape sequences.
        """
        log_color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            f"{log_color}{self.FORMAT}{self.RESET}",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def get_logger(name: str = "commkit") -> logging.Logger:
    """
    Retrieves and configures a logger instance for the library.

    If the requested logger has no handlers, a `StreamHandler` with the
    `ColorFormatter` is automatically attached to ensure immediate visibility.

    Parameters
    ----------
    name : str, default "commkit"
        The namespace for the logger.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)

    return logger


# Create a default logger for the package
logger = get_logger()


def set_log_level(level):
    """
    Sets the global log level for the CommKit library.

    Parameters
    ----------
    level : int or str
        The logging level to apply. Accepts standard `logging` constants
        (e.g., `logging.DEBUG`) or string identifiers (e.g., "DEBUG", "INFO").
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
