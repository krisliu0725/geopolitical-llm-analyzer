"""Rate limiter with token-bucket semantics and exponential backoff."""

import time
import threading
from typing import Callable

from config import MIN_API_INTERVAL_SEC, MAX_RETRIES


class RateLimiter:
    """Enforces a minimum interval between API calls and retries on rate-limit errors."""

    def __init__(
        self,
        min_interval_seconds: float = MIN_API_INTERVAL_SEC,
        max_retries: int = MAX_RETRIES,
    ):
        self._min_interval = min_interval_seconds
        self._max_retries = max_retries
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def acquire(self):
        """Block until the minimum interval since the last call has passed."""
        with self._lock:
            now = time.monotonic()
            wait = self._last_call_time + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_time = time.monotonic()

    def call_with_retry(self, fn: Callable, *args, **kwargs):
        """Call fn() with retry on rate-limit errors using exponential backoff."""
        self.acquire()
        last_exception = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if self._is_rate_limit(e) and attempt < self._max_retries:
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        raise last_exception

    @staticmethod
    def _is_rate_limit(exception: Exception) -> bool:
        msg = str(exception).lower()
        return any(
            keyword in msg
            for keyword in ["429", "rate limit", "too many requests", "rate_limit"]
        )
