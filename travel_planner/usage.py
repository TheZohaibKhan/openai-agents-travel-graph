"""A tiny shared counter that protects your free Gemini quota on a public website."""

from __future__ import annotations

import threading
from datetime import date


class DailyCounter:
    """Counts trips per calendar day, for all visitors together (resets each day)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = ""
        self._count = 0

    def try_take(self, limit: int, today: str | None = None) -> bool:
        """Reserve one trip. Returns False when today's limit is already used up."""
        if limit <= 0:  # 0 means "no limit"
            return True
        today = today or date.today().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= limit:
                return False
            self._count += 1
            return True

    def give_back(self) -> None:
        """Return a reserved trip (used when the trip failed through no fault of the visitor)."""
        with self._lock:
            if self._count > 0:
                self._count -= 1


COUNTER = DailyCounter()


# Separate, smaller daily allowance for paid/limited live-price lookups.
PRICE_COUNTER = DailyCounter()
