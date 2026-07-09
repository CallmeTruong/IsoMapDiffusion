"""Spot-instance retry helper.

Wraps a single tile/batch submission with exponential backoff. The
caller is responsible for being idempotent (e.g. a restart-safe resume
manifest), so retries are safe.

This module deliberately has no async dependency on the rest of the
client so unit tests can drive it with a plain function.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


async def with_spot_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    backoff_base_s: float = 5.0,
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Call ``fn()`` with exponential backoff on any exception.

    Args:
        fn: zero-arg async callable that performs the unit of work.
        max_retries: total attempts (so 3 = 1 try + 2 retries).
        backoff_base_s: first sleep is backoff_base_s; doubles each retry.
        on_retry: optional callback invoked with (attempt_index, exception)
            after a failure, before the sleep. Useful for the resume
            manifest's mark_failed.

    Returns:
        Whatever fn() returns on success.

    Raises:
        The last exception raised by fn() if all attempts fail.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except BaseException as e:  # noqa: BLE001 - we want to retry broadly
            last_exc = e
            if attempt >= max_retries:
                break
            if on_retry is not None:
                try:
                    on_retry(attempt, e)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("on_retry callback raised")
            sleep_s = backoff_base_s * (2 ** (attempt - 1))
            logger.warning(
                "Spot retry: attempt %d/%d failed (%s); sleeping %.1fs",
                attempt, max_retries, e, sleep_s,
            )
            await asyncio.sleep(sleep_s)

    assert last_exc is not None  # loop above always sets this on failure
    raise last_exc
