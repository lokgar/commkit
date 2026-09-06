"""Tests for commkit logging functionality."""

import logging

from commkit import logger


class TestCommkitLogger:
    """Tests for CommKit logger configuration, formatting, and level filtering."""

    def test_logger_set_level(self) -> None:
        """Verify setting log level via uppercase and lowercase string."""
        logger.set_log_level("DEBUG")
        assert logger.logger.level == logging.DEBUG
        logger.set_log_level("info")
        assert logger.logger.level == logging.INFO
        logger.set_log_level(logging.WARNING)
        assert logger.logger.level == logging.WARNING

    def test_logger_singleton_instance(self) -> None:
        """get_logger returns configured logger with StreamHandler and ColorFormatter."""
        lg = logger.get_logger("commkit")
        assert lg is not None
        assert len(lg.handlers) >= 1
        assert any(isinstance(h.formatter, logger.ColorFormatter) for h in lg.handlers)

    def test_color_formatter_applies_ansi_codes(self) -> None:
        """ColorFormatter wraps messages in ANSI color codes."""
        formatter = logger.ColorFormatter()
        rec_debug = logging.LogRecord(
            name="commkit",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=10,
            msg="debug message",
            args=(),
            exc_info=None,
        )
        formatted_debug = formatter.format(rec_debug)
        assert logger.ColorFormatter.CYAN in formatted_debug
        assert logger.ColorFormatter.RESET in formatted_debug
