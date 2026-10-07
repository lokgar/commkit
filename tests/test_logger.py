"""Tests for commkit logging: no handlers at import, opt-in console output."""

import logging

import pytest

from commkit import logger


@pytest.fixture
def clean_logger():
    """Restore the commkit logger's handlers, level and propagation afterwards."""
    lg = logger.logger
    saved = (list(lg.handlers), lg.level, lg.propagate)
    lg.handlers.clear()
    yield lg
    lg.handlers[:] = saved[0]
    lg.setLevel(saved[1])
    lg.propagate = saved[2]


class TestCommkitLogger:
    """The library logger is unconfigured until the user asks for output."""

    def test_package_logger_is_named_commkit(self) -> None:
        assert logger.logger is logging.getLogger("commkit")

    def test_set_log_level_accepts_names_and_numbers(self, clean_logger) -> None:
        logger.set_log_level("DEBUG")
        assert clean_logger.level == logging.DEBUG
        logger.set_log_level("info")
        assert clean_logger.level == logging.INFO
        logger.set_log_level(logging.WARNING)
        assert clean_logger.level == logging.WARNING

    def test_set_log_level_attaches_one_colour_handler(self, clean_logger) -> None:
        """The first call adds a console handler; later calls do not add more."""
        logger.set_log_level("INFO")
        logger.set_log_level("DEBUG")
        assert len(clean_logger.handlers) == 1
        assert isinstance(clean_logger.handlers[0].formatter, logger._ColorFormatter)
        assert clean_logger.propagate is False

    def test_set_log_level_keeps_existing_handlers(self, clean_logger) -> None:
        """An application's own handler is left alone."""
        own = logging.NullHandler()
        clean_logger.addHandler(own)
        logger.set_log_level("INFO")
        assert clean_logger.handlers == [own]

    def test_color_formatter_applies_ansi_codes(self) -> None:
        """The console formatter wraps messages in ANSI colour codes."""
        rec = logging.LogRecord(
            name="commkit",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=10,
            msg="debug message",
            args=(),
            exc_info=None,
        )
        formatted = logger._ColorFormatter().format(rec)
        assert logger._ColorFormatter.CYAN in formatted
        assert logger._ColorFormatter.RESET in formatted
