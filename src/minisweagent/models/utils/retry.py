"""Retry utility for model queries."""

import logging
import os
from dataclasses import dataclass

from tenacity import Retrying, before_sleep_log, retry_if_not_exception_type, stop_after_attempt, wait_exponential


@dataclass
class RetryStats:
    retry_count: int = 0
    sleep_time: float = 0.0


def retry(
    *,
    logger: logging.Logger,
    abort_exceptions: list[type[Exception]],
    stats: RetryStats | None = None,
) -> Retrying:
    """Thin wrapper around tenacity.Retrying to make use of global config etc.

    Args:
        logger: Logger to use for reporting retries
        abort_exceptions: Exceptions to abort on.
        stats: Optional retry stats accumulator.

    Returns:
        A tenacity.Retrying object.
    """
    log_before_sleep = before_sleep_log(logger, logging.WARNING)

    def _before_sleep(retry_state) -> None:
        if stats is not None:
            stats.retry_count += 1
            next_action = getattr(retry_state, "next_action", None)
            sleep = float(getattr(next_action, "sleep", 0.0) or 0.0)
            stats.sleep_time += sleep
        log_before_sleep(retry_state)

    return Retrying(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=_before_sleep,
        retry=retry_if_not_exception_type(tuple(abort_exceptions)),
    )
