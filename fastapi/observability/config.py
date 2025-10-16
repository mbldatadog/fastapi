"""
Configuration for Datadog metrics instrumentation.
"""
import os

# Enable/disable metrics collection
DATADOG_ENABLED = os.getenv("DATADOG_ENABLED", "true").lower() == "true"

# Datadog agent connection settings
DATADOG_HOST = os.getenv("DATADOG_HOST", "localhost")
DATADOG_PORT = int(os.getenv("DATADOG_PORT", "8125"))
DATADOG_NAMESPACE = os.getenv("DATADOG_NAMESPACE", "fastapi")

# Feature flags for specific metric groups
METRICS_REQUEST_HANDLER = os.getenv("METRICS_REQUEST_HANDLER", "true").lower() == "true"
METRICS_DEPENDENCIES = os.getenv("METRICS_DEPENDENCIES", "true").lower() == "true"
METRICS_SERIALIZATION = os.getenv("METRICS_SERIALIZATION", "true").lower() == "true"
METRICS_AUTH = os.getenv("METRICS_AUTH", "true").lower() == "true"
METRICS_ENCODING = os.getenv("METRICS_ENCODING", "true").lower() == "true"
