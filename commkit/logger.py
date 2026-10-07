"""
Logging for the CommKit library.

Importing commkit installs no log handler.  Call :func:`set_log_level` to show
commkit's own diagnostics with a coloured console handler, or configure
``logging`` in your application as usual.
"""

import logging
import sys

__all__ = ["logger", "set_log_level"]


class _ColorFormatter(logging.Formatter):
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


# The package logger.  No handler is installed at import: the library never
# configures logging for the application.  Without any logging configuration,
# Python's last-resort handler still prints WARNING and above to stderr, so
# warnings stay visible; INFO/DEBUG diagnostics appear once the application
# configures logging or calls ``set_log_level``.
logger = logging.getLogger("commkit")


def set_log_level(level: int | str) -> None:
    """
    Show commkit's log messages at ``level`` and above.

    Sets the level of the ``commkit`` logger.  On the first call it also
    attaches a coloured console handler (stdout), unless the logger already
    has handlers, so calling this function is all a notebook needs to see
    commkit's diagnostics.  The handler is attached only on request; importing
    commkit never configures logging.

    Parameters
    ----------
    level : int or str
        A standard logging level, e.g. ``logging.DEBUG`` or ``"INFO"``.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColorFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # avoid duplicate lines via the root logger
