"""Tests for logger functionality."""


def test_logger_set_level():
    """Test setting log level via string."""
    from commkit import logger

    logger.set_log_level("DEBUG")
    assert logger.logger.level == 10
    logger.set_log_level("INFO")
    assert logger.logger.level == 20
