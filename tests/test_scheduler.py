"""Tests for the background task scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from ai_news_tracker.scheduler import FeedRefreshScheduler


class TestFeedRefreshSchedulerInit:
    """Tests for scheduler initialization."""

    def test_init_with_defaults(self):
        """Test scheduler initialization with default values."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        assert scheduler.refresh_callback == callback
        assert scheduler.interval_minutes == 30
        assert scheduler.enabled is True
        assert scheduler._task is None
        assert scheduler._running is False
        assert scheduler.last_refresh is None
        assert scheduler.next_refresh is None
        assert scheduler.refresh_count == 0
        assert scheduler.error_count == 0
        assert scheduler.last_error is None

    def test_init_with_custom_values(self):
        """Test scheduler initialization with custom values."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(
            callback,
            interval_minutes=15,
            enabled=False,
        )

        assert scheduler.interval_minutes == 15
        assert scheduler.enabled is False


class TestFeedRefreshSchedulerStartStop:
    """Tests for scheduler start and stop."""

    def test_start_creates_task(self):
        """Test that start creates a background task."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback)

            await scheduler.start()

            assert scheduler._running is True
            assert scheduler._task is not None

            # Clean up
            await scheduler.stop()

        asyncio.run(run_test())

    def test_start_when_already_running(self):
        """Test that starting when already running logs a warning."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback)

            await scheduler.start()
            # Start again - should just return
            await scheduler.start()

            assert scheduler._running is True

            # Clean up
            await scheduler.stop()

        asyncio.run(run_test())

    def test_stop_cancels_task(self):
        """Test that stop cancels the background task."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback)

            await scheduler.start()
            await scheduler.stop()

            assert scheduler._running is False
            assert scheduler._task is None

        asyncio.run(run_test())

    def test_stop_when_not_running(self):
        """Test that stopping when not running is a no-op."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback)

            await scheduler.stop()

            assert scheduler._running is False
            assert scheduler._task is None

        asyncio.run(run_test())


class TestFeedRefreshSchedulerDoRefresh:
    """Tests for the refresh execution."""

    def test_do_refresh_sync_callback(self):
        """Test refresh with a synchronous callback."""
        async def run_test():
            callback = MagicMock(return_value={"articles": 10})
            scheduler = FeedRefreshScheduler(callback)

            await scheduler._do_refresh()

            callback.assert_called_once()
            assert scheduler.last_refresh is not None
            assert scheduler.refresh_count == 1
            assert scheduler.error_count == 0

        asyncio.run(run_test())

    def test_do_refresh_async_callback(self):
        """Test refresh with an asynchronous callback."""
        async def run_test():
            callback = AsyncMock(return_value={"articles": 10})
            scheduler = FeedRefreshScheduler(callback)

            await scheduler._do_refresh()

            callback.assert_called_once()
            assert scheduler.last_refresh is not None
            assert scheduler.refresh_count == 1

        asyncio.run(run_test())

    def test_do_refresh_callback_error(self):
        """Test refresh handles callback errors gracefully."""
        async def run_test():
            callback = MagicMock(side_effect=Exception("Fetch failed"))
            scheduler = FeedRefreshScheduler(callback)

            await scheduler._do_refresh()

            assert scheduler.error_count == 1
            assert scheduler.last_error == "Fetch failed"
            assert scheduler.refresh_count == 0

        asyncio.run(run_test())

    def test_refresh_now(self):
        """Test manual refresh trigger."""
        async def run_test():
            callback = MagicMock(return_value={"articles": 5})
            scheduler = FeedRefreshScheduler(callback)

            result = await scheduler.refresh_now()

            assert result["status"] == "ok"
            assert "last_refresh" in result
            assert scheduler.refresh_count == 1

        asyncio.run(run_test())


class TestFeedRefreshSchedulerConfiguration:
    """Tests for scheduler configuration methods."""

    def test_set_interval(self):
        """Test setting the refresh interval."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        scheduler.set_interval(45)

        assert scheduler.interval_minutes == 45

    def test_set_interval_minimum(self):
        """Test that interval cannot be less than 1."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        scheduler.set_interval(0)

        assert scheduler.interval_minutes == 1

    def test_set_interval_negative(self):
        """Test that negative interval is clamped to 1."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        scheduler.set_interval(-10)

        assert scheduler.interval_minutes == 1

    def test_set_enabled(self):
        """Test enabling and disabling the scheduler."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        scheduler.set_enabled(False)
        assert scheduler.enabled is False

        scheduler.set_enabled(True)
        assert scheduler.enabled is True


class TestFeedRefreshSchedulerStatus:
    """Tests for scheduler status reporting."""

    def test_get_status_initial(self):
        """Test status of a fresh scheduler."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback)

        status = scheduler.get_status()

        assert status["enabled"] is True
        assert status["running"] is False
        assert status["interval_minutes"] == 30
        assert status["last_refresh"] is None
        assert status["next_refresh"] is None
        assert status["refresh_count"] == 0
        assert status["error_count"] == 0
        assert status["last_error"] is None

    def test_get_status_after_refresh(self):
        """Test status after a refresh."""
        async def run_test():
            callback = MagicMock(return_value={"articles": 10})
            scheduler = FeedRefreshScheduler(callback)

            await scheduler._do_refresh()
            status = scheduler.get_status()

            assert status["last_refresh"] is not None
            assert status["refresh_count"] == 1

        asyncio.run(run_test())

    def test_get_status_disabled(self):
        """Test status when scheduler is disabled."""
        callback = MagicMock()
        scheduler = FeedRefreshScheduler(callback, enabled=False)

        status = scheduler.get_status()

        assert status["enabled"] is False
        assert status["next_refresh"] is None


class TestFeedRefreshSchedulerRunLoop:
    """Tests for the main run loop."""

    def test_run_loop_disabled(self):
        """Test that disabled scheduler waits without refreshing."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback, enabled=False, interval_minutes=1)

            # Start and let it run briefly
            await scheduler.start()
            await asyncio.sleep(0.05)  # Brief wait
            await scheduler.stop()

            # Should not have called the callback
            callback.assert_not_called()

        asyncio.run(run_test())

    def test_run_loop_cancellation(self):
        """Test that the run loop handles cancellation."""
        async def run_test():
            callback = MagicMock()
            scheduler = FeedRefreshScheduler(callback, interval_minutes=60)

            await scheduler.start()
            # Immediately stop
            await scheduler.stop()

            assert scheduler._running is False

        asyncio.run(run_test())

    def test_run_loop_error_recovery(self):
        """Test that the run loop continues after errors."""
        async def run_test():
            call_count = 0

            def error_then_success():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("First call fails")
                return {"articles": 5}

            callback = MagicMock(side_effect=error_then_success)
            scheduler = FeedRefreshScheduler(callback, interval_minutes=1)
            scheduler.enabled = True
            scheduler._running = True

            # Manually trigger refreshes to test error recovery
            await scheduler._do_refresh()  # Should fail
            assert scheduler.error_count == 1

            await scheduler._do_refresh()  # Should succeed
            assert scheduler.refresh_count == 1

        asyncio.run(run_test())
