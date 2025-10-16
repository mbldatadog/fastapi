"""
Datadog metrics instrumentation utilities for FastAPI.

This module provides utilities for instrumenting FastAPI applications with
Datadog metrics, including decorators for tracking request counts, latency,
and errors.
"""
import time
from functools import wraps
from typing import Any, Callable, List, Optional

from fastapi.observability.config import (
    DATADOG_ENABLED,
    DATADOG_HOST,
    DATADOG_NAMESPACE,
    DATADOG_PORT,
)


class NoOpStatsd:
    """No-op StatsD client when Datadog is disabled."""

    def increment(self, metric: str, value: int = 1, tags: Optional[List[str]] = None) -> None:
        pass

    def histogram(self, metric: str, value: float, tags: Optional[List[str]] = None) -> None:
        pass

    def gauge(self, metric: str, value: float, tags: Optional[List[str]] = None) -> None:
        pass


# Initialize StatsD client
if DATADOG_ENABLED:
    try:
        from datadog import DogStatsd

        statsd = DogStatsd(
            host=DATADOG_HOST,
            port=DATADOG_PORT,
            namespace=DATADOG_NAMESPACE,
        )
    except ImportError:
        # If datadog is not installed, use no-op client
        statsd = NoOpStatsd()  # type: ignore[assignment]
else:
    statsd = NoOpStatsd()  # type: ignore[assignment]


def instrument_sync(metric_prefix: str, tags: Optional[List[str]] = None) -> Callable:
    """
    Decorator to instrument synchronous functions with Datadog metrics.
    Emits: request count, latency, and error count.

    Args:
        metric_prefix: Prefix for the metric name (e.g., "my_function")
        tags: Optional list of tags to apply to all metrics

    Returns:
        Decorator function

    Example:
        @instrument_sync("my_function", tags=["service:api"])
        def my_function(arg1, arg2):
            return arg1 + arg2
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tags_list = tags or []
            start_time = time.time()

            # Increment request counter
            statsd.increment(f"{metric_prefix}.count", tags=tags_list)

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # Increment error counter with error type
                error_tags = tags_list + [f"error_type:{type(e).__name__}"]
                statsd.increment(f"{metric_prefix}.errors", tags=error_tags)
                raise
            finally:
                # Record latency
                duration = time.time() - start_time
                statsd.histogram(f"{metric_prefix}.latency", duration, tags=tags_list)

        return wrapper

    return decorator


def instrument_async(metric_prefix: str, tags: Optional[List[str]] = None) -> Callable:
    """
    Decorator to instrument asynchronous functions with Datadog metrics.
    Emits: request count, latency, and error count.

    Args:
        metric_prefix: Prefix for the metric name (e.g., "my_async_function")
        tags: Optional list of tags to apply to all metrics

    Returns:
        Decorator function

    Example:
        @instrument_async("my_async_function", tags=["service:api"])
        async def my_async_function(arg1, arg2):
            return arg1 + arg2
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tags_list = tags or []
            start_time = time.time()

            # Increment request counter
            statsd.increment(f"{metric_prefix}.count", tags=tags_list)

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Increment error counter with error type
                error_tags = tags_list + [f"error_type:{type(e).__name__}"]
                statsd.increment(f"{metric_prefix}.errors", tags=error_tags)
                raise
            finally:
                # Record latency
                duration = time.time() - start_time
                statsd.histogram(f"{metric_prefix}.latency", duration, tags=tags_list)

        return wrapper

    return decorator
