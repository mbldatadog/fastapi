"""
FastAPI Observability Module

Provides instrumentation utilities for monitoring FastAPI applications
with Datadog metrics.
"""

from fastapi.observability.metrics import statsd, instrument_async, instrument_sync

__all__ = ["statsd", "instrument_async", "instrument_sync"]
