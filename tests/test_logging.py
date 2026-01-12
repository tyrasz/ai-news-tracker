"""Tests for logging configuration."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from ai_news_tracker.logging_config import setup_logging, get_logger, JsonFormatter


class TestSetupLogging:
    """Tests for setup_logging function."""

    def teardown_method(self):
        """Clean up logging after each test."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)

    def test_setup_logging_default_level(self):
        """Test default INFO level."""
        setup_logging()
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_logging_debug_level(self):
        """Test DEBUG level."""
        setup_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_warning_level(self):
        """Test WARNING level."""
        setup_logging(level="WARNING")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.WARNING

    def test_setup_logging_case_insensitive(self):
        """Test level is case insensitive."""
        setup_logging(level="debug")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_adds_console_handler(self):
        """Test that console handler is added."""
        setup_logging()
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_clears_existing_handlers(self):
        """Test that existing handlers are cleared."""
        root_logger = logging.getLogger()
        initial_count = len(root_logger.handlers)
        root_logger.addHandler(logging.StreamHandler())
        root_logger.addHandler(logging.StreamHandler())
        assert len(root_logger.handlers) == initial_count + 2

        setup_logging()
        # After setup, only the console handler should remain
        assert len(root_logger.handlers) == 1

    def test_setup_logging_with_file(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            setup_logging(log_file=str(log_file))

            root_logger = logging.getLogger()
            assert len(root_logger.handlers) == 2

            # Log something
            logger = get_logger("test")
            logger.info("Test message")

            # Check file was created and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logging_creates_log_directory(self):
        """Test that log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "deep" / "test.log"
            setup_logging(log_file=str(log_file))

            logger = get_logger("test")
            logger.info("Test message")

            assert log_file.exists()

    def test_setup_logging_with_json_format(self):
        """Test JSON format logging."""
        setup_logging(json_format=True)
        root_logger = logging.getLogger()

        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

    def test_setup_logging_silences_third_party(self):
        """Test that third-party loggers are silenced."""
        setup_logging()

        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("sentence_transformers").level == logging.WARNING


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_basic_message(self):
        """Test formatting a basic log message."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert data["timestamp"].endswith("Z")

    def test_format_with_args(self):
        """Test formatting message with arguments."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["message"] == "Hello world"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JsonFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "ERROR"
        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        # Add extra field
        record.custom_field = "custom_value"
        record.request_id = "abc123"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["custom_field"] == "custom_value"
        assert data["request_id"] == "abc123"

    def test_format_excludes_standard_fields(self):
        """Test that standard LogRecord fields are excluded."""
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        # These should NOT be in the output
        assert "pathname" not in data
        assert "lineno" not in data
        assert "funcName" not in data
        assert "created" not in data
        assert "msecs" not in data


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_correct_name(self):
        """Test that logger has correct name."""
        logger = get_logger("my.custom.logger")
        assert logger.name == "my.custom.logger"

    def test_get_logger_same_instance(self):
        """Test that same name returns same logger."""
        logger1 = get_logger("same.name")
        logger2 = get_logger("same.name")
        assert logger1 is logger2
