# FastAPI Datadog Metrics Instrumentation - Unit Test Plan

This document outlines a comprehensive unit testing strategy for the Datadog metrics instrumentation implemented in FastAPI. The instrumentation adds observability to 10 critical methods across the request lifecycle.

---

## Overview of Testing Approach

### Testing Philosophy

The testing approach focuses on **verification without actual Datadog dependencies**. We will:

1. **Mock the StatsD client** - Use `unittest.mock.patch` to replace `fastapi.observability.metrics.statsd` with a mock object
2. **Verify metric emissions** - Assert that metrics are called with correct names, values, and tags
3. **Test error scenarios** - Ensure error metrics are properly tracked with error_type tags
4. **Test feature flags** - Verify metrics can be disabled via environment variables
5. **Test NoOp behavior** - Ensure the NoOpStatsd class works when Datadog is disabled

### Key Components to Mock

```python
# Primary mock target
fastapi.observability.metrics.statsd

# Methods to verify are called:
- statsd.increment(metric_name, tags=tag_list)
- statsd.histogram(metric_name, value, tags=tag_list)
- statsd.gauge(metric_name, value, tags=tag_list)
```

### Test Structure

All tests will be in: `tests/test_observability_metrics.py`

Each test will follow this pattern:
```python
@patch("fastapi.observability.metrics.statsd")
def test_something(mock_statsd):
    # Setup
    app = FastAPI()
    # ... configure endpoint

    # Execute
    response = client.get("/endpoint")

    # Assert metric calls
    mock_statsd.increment.assert_called_with(
        "metric.name",
        tags=["tag1:value1", "tag2:value2"]
    )
    mock_statsd.histogram.assert_called()
```

---

## List of Unit Tests

### Category A: Request Handler Tests (5 tests)

These tests verify the main request handler (`routing.py:318`) emits correct metrics.

#### Test 1: `test_request_handler_success_metrics`
**Purpose**: Verify successful requests emit count and latency metrics with correct tags

**Setup**:
- Create simple FastAPI app with one GET endpoint
- Mock statsd client
- Make successful request

**Assertions**:
- `increment` called with `"request_handler.count"` and tags `["path:/test", "method:GET"]`
- `increment` called again with status_code tag `["path:/test", "method:GET", "status_code:200"]`
- `histogram` called with `"request_handler.latency"` and duration > 0
- No error metrics emitted

**Code Snippet**:
```python
@patch("fastapi.observability.metrics.statsd")
def test_request_handler_success_metrics(mock_statsd):
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "success"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify count metric with path and method
    mock_statsd.increment.assert_any_call(
        "request_handler.count",
        tags=["path:/test", "method:GET"]
    )

    # Verify latency histogram
    assert mock_statsd.histogram.call_count > 0
    latency_call = [c for c in mock_statsd.histogram.call_args_list
                    if c[0][0] == "request_handler.latency"][0]
    assert latency_call[0][1] >= 0  # duration should be non-negative
```

#### Test 2: `test_request_handler_http_exception_metrics`
**Purpose**: Verify HTTPException errors are tracked with correct error_type and status_code

**Setup**:
- Endpoint that raises HTTPException with status 404
- Mock statsd

**Assertions**:
- `increment` called with `"request_handler.errors"`
- Tags include `error_type:HTTPException` and `status_code:404`
- Latency still tracked despite error

#### Test 3: `test_request_handler_validation_error_metrics`
**Purpose**: Verify validation errors (422) are tracked separately

**Setup**:
- Endpoint with query parameter validation
- Send invalid data

**Assertions**:
- Error metric includes `error_type:RequestValidationError` and `status_code:422`

#### Test 4: `test_request_handler_unexpected_error_metrics`
**Purpose**: Verify unexpected exceptions are tracked with status_code:500

**Setup**:
- Endpoint that raises generic Exception

**Assertions**:
- Error metric includes `error_type:Exception` and `status_code:500`

#### Test 5: `test_request_handler_feature_flag_disabled`
**Purpose**: Verify no metrics when METRICS_REQUEST_HANDLER=false

**Setup**:
- Set environment variable `METRICS_REQUEST_HANDLER=false`
- Make request

**Assertions**:
- Mock statsd NOT called (or verify feature flag checking logic)

---

### Category B: Dependency Resolution Tests (5 tests)

Tests for `solve_dependencies()` in `dependencies/utils.py:606`

#### Test 6: `test_solve_dependencies_success_metrics`
**Purpose**: Verify dependency resolution emits count and latency metrics

**Setup**:
- Endpoint with multiple dependencies
- Mock statsd

**Assertions**:
- `increment` called with `"solve_dependencies.count"` and tag `dependency_count:N`
- `histogram` called with `"solve_dependencies.latency"`
- `gauge` called with `"solve_dependencies.cache_size"` if cache exists

#### Test 7: `test_solve_dependencies_cache_metrics`
**Purpose**: Verify cache size gauge is emitted

**Setup**:
- Make multiple requests to trigger dependency caching

**Assertions**:
- `gauge` called with `"solve_dependencies.cache_size"` with value > 0

#### Test 8: `test_solve_dependencies_validation_errors`
**Purpose**: Verify validation errors in dependencies are tracked

**Setup**:
- Dependency that fails validation

**Assertions**:
- `increment` called with `"solve_dependencies.validation_errors"`
- Tags include `error_count:N`

#### Test 9: `test_solve_dependencies_exception_handling`
**Purpose**: Verify unexpected errors in dependency resolution are tracked

**Setup**:
- Dependency that raises exception

**Assertions**:
- `increment` called with `"solve_dependencies.errors"`
- Tags include `error_type:ExceptionType`

#### Test 10: `test_solve_dependencies_feature_flag`
**Purpose**: Verify metrics disabled when METRICS_DEPENDENCIES=false

---

### Category C: Endpoint Execution Tests (4 tests)

Tests for `run_endpoint_function()` in `routing.py:280`

#### Test 11: `test_endpoint_execution_async_function`
**Purpose**: Verify async endpoint execution metrics

**Setup**:
- Async endpoint function

**Assertions**:
- `increment` called with `"endpoint_execution.count"`
- Tags include `is_coroutine:True` and `endpoint:module.function_name`
- `histogram` called with latency

#### Test 12: `test_endpoint_execution_sync_function`
**Purpose**: Verify sync endpoint execution metrics (threadpool)

**Setup**:
- Sync (non-async) endpoint function

**Assertions**:
- Tags include `is_coroutine:False`

#### Test 13: `test_endpoint_execution_error_tracking`
**Purpose**: Verify errors in endpoint execution are tracked

**Setup**:
- Endpoint that raises exception

**Assertions**:
- `increment` called with `"endpoint_execution.errors"`
- Tags include `error_type:ExceptionType`

#### Test 14: `test_endpoint_execution_latency_accuracy`
**Purpose**: Verify latency measurement is reasonably accurate

**Setup**:
- Endpoint with artificial delay (e.g., time.sleep or asyncio.sleep)

**Assertions**:
- Latency histogram value approximately matches expected delay

---

### Category D: Serialization Tests (4 tests)

Tests for `serialize_response()` in `routing.py:219`

#### Test 15: `test_serialize_response_success`
**Purpose**: Verify response serialization metrics for successful responses

**Setup**:
- Endpoint with response model

**Assertions**:
- `increment` called with `"serialize_response.count"`
- Tags include `response_type:TypeName` and `has_model:True`
- `histogram` called for both latency and size

#### Test 16: `test_serialize_response_validation_errors`
**Purpose**: Verify response validation errors are tracked

**Setup**:
- Endpoint returning invalid data for response model

**Assertions**:
- `increment` called with `"serialize_response.validation_errors"`
- Tags include `error_count:N`

#### Test 17: `test_serialize_response_size_tracking`
**Purpose**: Verify response size histogram

**Setup**:
- Endpoint returning various sized responses

**Assertions**:
- `histogram` called with `"serialize_response.size"` with value > 0

#### Test 18: `test_serialize_response_feature_flag`
**Purpose**: Verify METRICS_SERIALIZATION flag disables metrics

---

### Category E: Request Parsing Tests (6 tests)

Tests for `request_body_to_args()` and `request_params_to_args()` in `dependencies/utils.py`

#### Test 19: `test_request_body_to_args_json`
**Purpose**: Verify JSON body parsing metrics

**Setup**:
- POST endpoint with JSON body

**Assertions**:
- `increment` called with `"request_body_to_args.count"`
- Tags include `body_type:json` and `field_count:N`
- `histogram` called with `"request_body_to_args.body_size"`

#### Test 20: `test_request_body_to_args_form_data`
**Purpose**: Verify form data parsing metrics

**Setup**:
- POST endpoint with form data

**Assertions**:
- Tags include `body_type:form`

#### Test 21: `test_request_body_validation_errors`
**Purpose**: Verify body validation errors are tracked

**Setup**:
- Send invalid JSON body

**Assertions**:
- `increment` called with `"request_body_to_args.validation_errors"`

#### Test 22: `test_request_params_to_args_query`
**Purpose**: Verify query parameter extraction metrics

**Setup**:
- GET endpoint with query parameters

**Assertions**:
- `increment` called with `"request_params_to_args.count"`
- Tags include `param_location:query`

#### Test 23: `test_request_params_to_args_headers`
**Purpose**: Verify header parameter extraction metrics

**Setup**:
- Endpoint with header dependencies

**Assertions**:
- Tags include `param_location:headers`

#### Test 24: `test_request_params_validation_errors`
**Purpose**: Verify parameter validation errors are tracked

**Setup**:
- Invalid query parameter

**Assertions**:
- `increment` called with `"request_params_to_args.validation_errors"`

---

### Category F: Encoding Tests (3 tests)

Tests for `jsonable_encoder()` in `encoders.py:109`

#### Test 25: `test_jsonable_encoder_pydantic_model`
**Purpose**: Verify encoding metrics for Pydantic models

**Setup**:
- Call jsonable_encoder with Pydantic model

**Assertions**:
- `increment` called with `"jsonable_encoder.count"`
- Tags include `is_pydantic:True`
- `histogram` called for latency and output_size

#### Test 26: `test_jsonable_encoder_dataclass`
**Purpose**: Verify encoding metrics for dataclasses

**Setup**:
- Call jsonable_encoder with dataclass

**Assertions**:
- Tags include `is_dataclass:True`

#### Test 27: `test_jsonable_encoder_error_handling`
**Purpose**: Verify encoding errors are tracked

**Setup**:
- Pass unserializable object

**Assertions**:
- `increment` called with `"jsonable_encoder.errors"`
- Tags include `error_type:ValueError`

---

### Category G: Authentication Tests (5 tests)

Tests for OAuth2 and API Key authentication

#### Test 28: `test_oauth2_bearer_success`
**Purpose**: Verify successful OAuth2 authentication metrics

**Setup**:
- Endpoint with OAuth2PasswordBearer
- Send valid Authorization header

**Assertions**:
- `increment` called with `"auth.oauth2.count"`
- `increment` called with `"auth.oauth2.success"`
- Tags include `path:/endpoint` and `scheme:oauth2_bearer`
- `histogram` called with `"auth.oauth2.latency"`

#### Test 29: `test_oauth2_bearer_missing_token`
**Purpose**: Verify missing token failure metrics

**Setup**:
- Request without Authorization header

**Assertions**:
- `increment` called with `"auth.oauth2.failures"`
- Tags include `failure_reason:missing` and `auto_error:True`

#### Test 30: `test_oauth2_bearer_invalid_scheme`
**Purpose**: Verify invalid scheme (not "Bearer") failure metrics

**Setup**:
- Send Authorization header with "Basic" scheme

**Assertions**:
- Tags include `failure_reason:invalid_scheme`

#### Test 31: `test_api_key_header_success`
**Purpose**: Verify successful API key authentication metrics

**Setup**:
- Endpoint with APIKeyHeader
- Send valid key in header

**Assertions**:
- `increment` called with `"auth.api_key.count"`
- `increment` called with `"auth.api_key.success"`
- Tags include `header:x-api-key` and `scheme:api_key`

#### Test 32: `test_api_key_header_missing`
**Purpose**: Verify missing API key failure metrics

**Setup**:
- Request without API key header

**Assertions**:
- `increment` called with `"auth.api_key.failures"`
- Tags include `failure_reason:missing`

---

### Category H: Configuration and Infrastructure Tests (3 tests)

Tests for configuration and NoOp behavior

#### Test 33: `test_noop_statsd_when_disabled`
**Purpose**: Verify NoOpStatsd is used when DATADOG_ENABLED=false

**Setup**:
- Set environment variable `DATADOG_ENABLED=false`
- Import metrics module

**Assertions**:
- Verify `statsd` is instance of `NoOpStatsd`
- Verify metrics methods don't raise exceptions

#### Test 34: `test_metrics_disabled_by_feature_flags`
**Purpose**: Verify all feature flags work correctly

**Setup**:
- Set each feature flag to false
- Make requests that would trigger those metrics

**Assertions**:
- Verify corresponding metrics are not emitted

#### Test 35: `test_datadog_import_failure_fallback`
**Purpose**: Verify graceful fallback when datadog package not installed

**Setup**:
- Mock ImportError when importing datadog

**Assertions**:
- NoOpStatsd used as fallback
- No exceptions raised

---

## Test Implementation Details

### Mock Setup Pattern

```python
from unittest.mock import patch, MagicMock, call
import pytest
from fastapi import FastAPI, Depends, Query
from fastapi.testclient import TestClient
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from pydantic import BaseModel

# Common mock setup
@pytest.fixture
def mock_statsd():
    with patch("fastapi.observability.metrics.statsd") as mock:
        yield mock

@pytest.fixture
def mock_routing_statsd():
    with patch("fastapi.routing.statsd") as mock:
        yield mock

@pytest.fixture
def mock_dependencies_statsd():
    with patch("fastapi.dependencies.utils.statsd") as mock:
        yield mock
```

### Assertion Helpers

```python
def assert_metric_called(mock_statsd, method, metric_name, tags=None):
    """Helper to assert a metric was called with specific name and tags."""
    calls = getattr(mock_statsd, method).call_args_list
    for call_obj in calls:
        args, kwargs = call_obj
        if args[0] == metric_name:
            if tags is None:
                return True
            call_tags = kwargs.get("tags", [])
            if all(tag in call_tags for tag in tags):
                return True
    pytest.fail(f"Metric {method}('{metric_name}') not called with tags {tags}")
```

### Feature Flag Testing Pattern

```python
import os

@pytest.fixture
def disable_request_metrics():
    original = os.environ.get("METRICS_REQUEST_HANDLER")
    os.environ["METRICS_REQUEST_HANDLER"] = "false"

    # Force reload of config module to pick up env var
    import importlib
    import fastapi.observability.config
    importlib.reload(fastapi.observability.config)

    yield

    # Restore
    if original is not None:
        os.environ["METRICS_REQUEST_HANDLER"] = original
    else:
        del os.environ["METRICS_REQUEST_HANDLER"]
    importlib.reload(fastapi.observability.config)
```

---

## Running the Tests

### Execute All Tests
```bash
pytest tests/test_observability_metrics.py -v
```

### Execute Specific Category
```bash
# Request handler tests only
pytest tests/test_observability_metrics.py -k "request_handler" -v

# Authentication tests only
pytest tests/test_observability_metrics.py -k "auth" -v
```

### With Coverage
```bash
pytest tests/test_observability_metrics.py --cov=fastapi.observability --cov-report=html
```

---

## Success Criteria

The test suite should:

1. **Achieve >90% code coverage** for instrumented methods
2. **All tests pass** without flakiness
3. **Mock isolation** - No actual Datadog connection required
4. **Fast execution** - Complete suite runs in <10 seconds
5. **Clear failures** - Failed assertions clearly indicate which metric or tag is wrong

---

## Maintenance Notes

### When Adding New Instrumentation

1. Add corresponding test in appropriate category
2. Follow the test naming convention: `test_{method_name}_{scenario}`
3. Always test: success path, error path, and feature flag
4. Update this document with new test details

### Common Pitfalls to Avoid

1. **Mock scope** - Ensure mock is patched in the correct module (where it's imported, not where it's defined)
2. **Async context** - Use `TestClient` which handles async properly
3. **Tag ordering** - Don't assume tag order in assertions, check for tag presence
4. **Timing sensitivity** - Don't assert exact latency values, use ranges
5. **Feature flags** - Remember to reload config module after changing env vars

---

## Summary

This test plan provides comprehensive coverage for the Datadog metrics instrumentation:

- **35+ unit tests** covering all 10 instrumented methods
- **8 test categories** organized by functionality
- **Mock-based testing** with no external dependencies
- **Feature flag coverage** to ensure metrics can be toggled
- **Error scenarios** to verify error tracking works correctly

The tests verify that metrics are emitted with correct:
- Metric names (e.g., `fastapi.request_handler.count`)
- Metric types (increment, histogram, gauge)
- Tags (e.g., `path:/endpoint`, `method:GET`, `error_type:ValueError`)
- Values (latency, size, count)

This comprehensive test suite ensures the instrumentation is reliable, maintainable, and provides accurate observability data for FastAPI applications.
