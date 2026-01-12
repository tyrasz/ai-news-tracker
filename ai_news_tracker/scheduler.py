"""Background task scheduler for periodic feed refreshes."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)


class FeedRefreshScheduler:
    """
    Manages periodic background feed refreshes.

    Can be started/stopped and configured with custom intervals.
    """

    def __init__(
        self,
        refresh_callback: Callable[[], Any],
        interval_minutes: int = 30,
        enabled: bool = True,
    ):
        """
        Initialize the scheduler.

        Args:
            refresh_callback: Function to call for refreshing feeds
            interval_minutes: Minutes between refresh cycles
            enabled: Whether to start enabled
        """
        self.refresh_callback = refresh_callback
        self.interval_minutes = interval_minutes
        self.enabled = enabled
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_refresh: Optional[datetime] = None
        self.next_refresh: Optional[datetime] = None
        self.refresh_count = 0
        self.error_count = 0
        self.last_error: Optional[str] = None

    async def start(self):
        """Start the background scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Feed refresh scheduler started (interval: {self.interval_minutes} minutes)")

    async def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Feed refresh scheduler stopped")

    async def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                if self.enabled:
                    self.next_refresh = datetime.utcnow() + timedelta(minutes=self.interval_minutes)

                    # Wait for the interval
                    await asyncio.sleep(self.interval_minutes * 60)

                    if self._running and self.enabled:
                        await self._do_refresh()
                else:
                    # Check every minute if we've been enabled
                    await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                self.last_error = str(e)
                logger.error(f"Scheduler error: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def _do_refresh(self):
        """Execute a feed refresh."""
        try:
            logger.info("Starting scheduled feed refresh")
            start_time = datetime.utcnow()

            # Run the callback (might be sync or async)
            result = self.refresh_callback()
            if asyncio.iscoroutine(result):
                result = await result

            self.last_refresh = datetime.utcnow()
            self.refresh_count += 1
            duration = (self.last_refresh - start_time).total_seconds()

            logger.info(f"Scheduled feed refresh completed in {duration:.1f}s, result: {result}")

        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Feed refresh failed: {e}")

    async def refresh_now(self) -> dict:
        """Trigger an immediate refresh outside the normal schedule."""
        await self._do_refresh()
        return {
            "status": "ok",
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
        }

    def set_interval(self, minutes: int):
        """Update the refresh interval."""
        self.interval_minutes = max(1, minutes)
        logger.info(f"Refresh interval updated to {self.interval_minutes} minutes")

    def set_enabled(self, enabled: bool):
        """Enable or disable scheduled refreshes."""
        self.enabled = enabled
        logger.info(f"Scheduler {'enabled' if enabled else 'disabled'}")

    def get_status(self) -> dict:
        """Get current scheduler status."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "interval_minutes": self.interval_minutes,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "next_refresh": self.next_refresh.isoformat() if self.next_refresh and self.enabled else None,
            "refresh_count": self.refresh_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }
