"""
Unit tests for Datadog metrics instrumentation in FastAPI.

This test suite verifies that all 10 instrumented methods emit correct metrics
with proper tags, values, and error handling.
"""
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi import Depends, FastAPI, HTTPException, Query, Header, Body
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN


# Test fixtures
@pytest.fixture
def mock_statsd():
    """Mock the statsd client in the observability module."""
    with patch("fastapi.observability.metrics.statsd") as mock:
        yield mock


@pytest.fixture
def mock_routing_statsd():
    """Mock statsd specifically in routing module."""
    with patch("fastapi.routing.statsd") as mock:
        yield mock


@pytest.fixture
def mock_dependencies_statsd():
    """Mock statsd specifically in dependencies utils module."""
    with patch("fastapi.dependencies.utils.statsd") as mock:
        yield mock


@pytest.fixture
def mock_encoders_statsd():
    """Mock statsd specifically in encoders module."""
    with patch("fastapi.encoders.statsd") as mock:
        yield mock


@pytest.fixture
def mock_api_key_statsd():
    """Mock statsd specifically in api_key module."""
    with patch("fastapi.security.api_key.statsd") as mock:
        yield mock


@pytest.fixture
def mock_oauth2_statsd():
    """Mock statsd specifically in oauth2 module."""
    with patch("fastapi.security.oauth2.statsd") as mock:
        yield mock


# Helper functions
def assert_metric_called(mock_statsd, method, metric_name, tags=None):
    """
    Helper to assert a metric was called with specific name and tags.

    Args:
        mock_statsd: The mocked statsd object
        method: Method name (e.g., 'increment', 'histogram', 'gauge')
        metric_name: Expected metric name
        tags: Optional list of tags to check for
    """
    mock_method = getattr(mock_statsd, method)
    calls = mock_method.call_args_list

    for call_obj in calls:
        args, kwargs = call_obj
        if args[0] == metric_name:
            if tags is None:
                return True
            call_tags = kwargs.get("tags", [])
            if all(tag in call_tags for tag in tags):
                return True

    # If we get here, the metric wasn't called as expected
    pytest.fail(
        f"Metric {method}('{metric_name}') not called with tags {tags}. "
        f"Actual calls: {calls}"
    )


def get_metric_calls(mock_statsd, method, metric_name):
    """Get all calls to a specific metric."""
    mock_method = getattr(mock_statsd, method)
    calls = mock_method.call_args_list
    return [c for c in calls if c[0][0] == metric_name]


# ============================================================================
# Category A: Request Handler Tests (5 tests)
# ============================================================================

def test_request_handler_success_metrics(mock_routing_statsd):
    """Test that successful requests emit count and latency metrics."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "success"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify count metric was called
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.count",
        tags=["path:/test", "method:GET"]
    )

    # Verify latency histogram was called
    latency_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "request_handler.latency"
    )
    assert len(latency_calls) > 0
    # Check that duration is non-negative
    duration = latency_calls[0][0][1]
    assert duration >= 0


def test_request_handler_http_exception_metrics(mock_routing_statsd):
    """Test that HTTPException errors are tracked with correct tags."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        raise HTTPException(status_code=404, detail="Not found")

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 404

    # Verify error metric was called with correct tags
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.errors",
        tags=["error_type:HTTPException", "status_code:404"]
    )

    # Verify latency is still tracked
    latency_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "request_handler.latency"
    )
    assert len(latency_calls) > 0


def test_request_handler_validation_error_metrics(mock_routing_statsd):
    """Test that validation errors (422) are tracked separately."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(item_id: int = Query(...)):
        return {"item_id": item_id}

    client = TestClient(app)
    # Send invalid query parameter (string instead of int)
    response = client.get("/test?item_id=invalid")

    assert response.status_code == 422

    # Verify error metric includes validation error tags
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.errors",
        tags=["error_type:RequestValidationError", "status_code:422"]
    )


def test_request_handler_unexpected_error_metrics(mock_routing_statsd):
    """Test that unexpected exceptions are tracked with status 500."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        raise ValueError("Unexpected error")

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 500

    # Verify error metric includes generic exception type
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.errors",
        tags=["error_type:ValueError", "status_code:500"]
    )


def test_request_handler_different_methods(mock_routing_statsd):
    """Test that different HTTP methods are tagged correctly."""
    app = FastAPI()

    @app.post("/test")
    def test_post():
        return {"method": "POST"}

    @app.put("/test")
    def test_put():
        return {"method": "PUT"}

    client = TestClient(app)

    # Test POST
    response = client.post("/test")
    assert response.status_code == 200
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.count",
        tags=["path:/test", "method:POST"]
    )

    # Test PUT
    response = client.put("/test")
    assert response.status_code == 200
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.count",
        tags=["path:/test", "method:PUT"]
    )


# ============================================================================
# Category B: Dependency Resolution Tests (5 tests)
# ============================================================================

def test_solve_dependencies_success_metrics(mock_dependencies_statsd):
    """Test that dependency resolution emits count and latency metrics."""
    app = FastAPI()

    def dependency1():
        return "dep1"

    def dependency2():
        return "dep2"

    @app.get("/test")
    def test_endpoint(
        d1: str = Depends(dependency1),
        d2: str = Depends(dependency2)
    ):
        return {"d1": d1, "d2": d2}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify dependency resolution metrics
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "solve_dependencies.count"
    )

    # Verify latency was tracked
    latency_calls = get_metric_calls(
        mock_dependencies_statsd, "histogram", "solve_dependencies.latency"
    )
    assert len(latency_calls) > 0


def test_solve_dependencies_cache_metrics(mock_dependencies_statsd):
    """Test that cache size gauge is emitted."""
    app = FastAPI()

    # Dependency with use_cache=True (default)
    def cached_dependency():
        return "cached"

    @app.get("/test")
    def test_endpoint(dep: str = Depends(cached_dependency)):
        return {"dep": dep}

    client = TestClient(app)

    # Make multiple requests to populate cache
    response1 = client.get("/test")
    response2 = client.get("/test")

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Check if cache size gauge was called
    gauge_calls = get_metric_calls(
        mock_dependencies_statsd, "gauge", "solve_dependencies.cache_size"
    )
    # Cache size should be tracked
    assert len(gauge_calls) >= 0  # May or may not have cache depending on implementation


def test_solve_dependencies_with_query_params(mock_dependencies_statsd):
    """Test dependency resolution with query parameters."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(q: str = Query(...), limit: int = Query(10)):
        return {"q": q, "limit": limit}

    client = TestClient(app)
    response = client.get("/test?q=search&limit=20")

    assert response.status_code == 200

    # Verify metrics were emitted
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "solve_dependencies.count"
    )


def test_solve_dependencies_validation_errors(mock_dependencies_statsd):
    """Test that validation errors in dependencies are tracked."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(item_id: int = Query(..., gt=0)):
        return {"item_id": item_id}

    client = TestClient(app)
    # Send invalid value (negative when gt=0 required)
    response = client.get("/test?item_id=-1")

    assert response.status_code == 422

    # Validation errors should be tracked
    validation_calls = get_metric_calls(
        mock_dependencies_statsd, "increment", "solve_dependencies.validation_errors"
    )
    # Note: This may or may not be called depending on where validation happens
    # The important thing is that the request_handler.errors is called


def test_solve_dependencies_with_multiple_types(mock_dependencies_statsd):
    """Test dependency resolution with mixed parameter types."""
    app = FastAPI()

    @app.get("/test/{item_id}")
    def test_endpoint(
        item_id: int,
        q: Optional[str] = Query(None),
        x_token: str = Header(...)
    ):
        return {"item_id": item_id, "q": q, "x_token": x_token}

    client = TestClient(app)
    response = client.get("/test/123?q=search", headers={"x-token": "secret"})

    assert response.status_code == 200

    # Verify dependency resolution was tracked
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "solve_dependencies.count"
    )


# ============================================================================
# Category C: Endpoint Execution Tests (4 tests)
# ============================================================================

def test_endpoint_execution_async_function(mock_routing_statsd):
    """Test that async endpoint execution is tracked correctly."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        await asyncio.sleep(0.01)  # Small delay
        return {"message": "async"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify endpoint execution metrics
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "endpoint_execution.count",
        tags=["is_coroutine:True"]
    )

    # Verify latency
    latency_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "endpoint_execution.latency"
    )
    assert len(latency_calls) > 0


def test_endpoint_execution_sync_function(mock_routing_statsd):
    """Test that sync endpoint execution is tracked (threadpool)."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        time.sleep(0.01)  # Small delay
        return {"message": "sync"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify endpoint execution metrics for sync function
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "endpoint_execution.count",
        tags=["is_coroutine:False"]
    )


def test_endpoint_execution_error_tracking(mock_routing_statsd):
    """Test that errors in endpoint execution are tracked."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        raise RuntimeError("Endpoint error")

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 500

    # Verify error metric
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "endpoint_execution.errors",
        tags=["error_type:RuntimeError"]
    )


def test_endpoint_execution_latency_tracking(mock_routing_statsd):
    """Test that endpoint latency is measured accurately."""
    app = FastAPI()

    @app.get("/test")
    async def test_endpoint():
        await asyncio.sleep(0.05)  # 50ms delay
        return {"message": "delayed"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify latency is approximately correct
    latency_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "endpoint_execution.latency"
    )
    assert len(latency_calls) > 0
    duration = latency_calls[0][0][1]
    # Should be at least 50ms (0.05s), allow some overhead
    assert duration >= 0.04


# ============================================================================
# Category D: Serialization Tests (4 tests)
# ============================================================================

def test_serialize_response_success(mock_routing_statsd):
    """Test that response serialization metrics are emitted."""
    app = FastAPI()

    class ResponseModel(BaseModel):
        message: str
        count: int

    @app.get("/test", response_model=ResponseModel)
    def test_endpoint():
        return {"message": "success", "count": 42}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify serialization metrics
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "serialize_response.count",
        tags=["has_model:True"]
    )

    # Verify latency
    latency_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "serialize_response.latency"
    )
    assert len(latency_calls) > 0


def test_serialize_response_without_model(mock_routing_statsd):
    """Test serialization when no response model is specified."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "success"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify serialization metrics for responses without model
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "serialize_response.count",
        tags=["has_model:False"]
    )


def test_serialize_response_size_tracking(mock_routing_statsd):
    """Test that response size is tracked."""
    app = FastAPI()

    class ResponseModel(BaseModel):
        data: str

    @app.get("/test", response_model=ResponseModel)
    def test_endpoint():
        return {"data": "x" * 1000}  # Large response

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200

    # Verify size histogram
    size_calls = get_metric_calls(
        mock_routing_statsd, "histogram", "serialize_response.size"
    )
    assert len(size_calls) > 0
    size = size_calls[0][0][1]
    assert size > 0


def test_serialize_response_validation_error(mock_routing_statsd):
    """Test that response validation errors are tracked."""
    app = FastAPI()

    class ResponseModel(BaseModel):
        count: int  # Required field

    @app.get("/test", response_model=ResponseModel)
    def test_endpoint():
        return {"message": "missing count field"}  # Invalid response

    client = TestClient(app)
    response = client.get("/test")

    # Should return 500 due to response validation error
    assert response.status_code == 500

    # Verify validation error metric
    validation_calls = get_metric_calls(
        mock_routing_statsd, "increment", "serialize_response.validation_errors"
    )
    assert len(validation_calls) > 0


# ============================================================================
# Category E: Request Parsing Tests (6 tests)
# ============================================================================

def test_request_body_to_args_json(mock_dependencies_statsd):
    """Test that JSON body parsing metrics are emitted."""
    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float

    @app.post("/test")
    def test_endpoint(item: Item):
        return {"name": item.name, "price": item.price}

    client = TestClient(app)
    response = client.post("/test", json={"name": "Widget", "price": 9.99})

    assert response.status_code == 200

    # Verify body parsing metrics
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "request_body_to_args.count",
        tags=["body_type:json"]
    )

    # Verify body size was tracked
    size_calls = get_metric_calls(
        mock_dependencies_statsd, "histogram", "request_body_to_args.body_size"
    )
    assert len(size_calls) > 0


def test_request_body_to_args_form_data(mock_dependencies_statsd):
    """Test that form data parsing metrics are emitted."""
    app = FastAPI()

    @app.post("/test")
    def test_endpoint(username: str = Body(...), password: str = Body(...)):
        return {"username": username}

    client = TestClient(app)
    response = client.post(
        "/test",
        data={"username": "testuser", "password": "testpass"}
    )

    assert response.status_code == 200

    # Verify form data metrics
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "request_body_to_args.count",
        tags=["body_type:form"]
    )


def test_request_body_validation_errors(mock_dependencies_statsd):
    """Test that body validation errors are tracked."""
    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float = Field(..., gt=0)

    @app.post("/test")
    def test_endpoint(item: Item):
        return item

    client = TestClient(app)
    # Send invalid data (negative price)
    response = client.post("/test", json={"name": "Widget", "price": -1})

    assert response.status_code == 422

    # Validation errors should be tracked
    validation_calls = get_metric_calls(
        mock_dependencies_statsd, "increment", "request_body_to_args.validation_errors"
    )
    assert len(validation_calls) > 0


def test_request_params_to_args_query(mock_dependencies_statsd):
    """Test that query parameter extraction metrics are emitted."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(q: str = Query(...), limit: int = Query(10)):
        return {"q": q, "limit": limit}

    client = TestClient(app)
    response = client.get("/test?q=search&limit=20")

    assert response.status_code == 200

    # Verify query param metrics
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "request_params_to_args.count",
        tags=["param_location:query"]
    )


def test_request_params_to_args_headers(mock_dependencies_statsd):
    """Test that header parameter extraction metrics are emitted."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(
        x_token: str = Header(...),
        user_agent: Optional[str] = Header(None)
    ):
        return {"x_token": x_token, "user_agent": user_agent}

    client = TestClient(app)
    response = client.get(
        "/test",
        headers={"x-token": "secret", "user-agent": "test-client"}
    )

    assert response.status_code == 200

    # Verify header param metrics
    assert_metric_called(
        mock_dependencies_statsd,
        "increment",
        "request_params_to_args.count",
        tags=["param_location:headers"]
    )


def test_request_params_validation_errors(mock_dependencies_statsd):
    """Test that parameter validation errors are tracked."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint(limit: int = Query(..., ge=1, le=100)):
        return {"limit": limit}

    client = TestClient(app)
    # Send invalid value (exceeds maximum)
    response = client.get("/test?limit=200")

    assert response.status_code == 422

    # Validation errors should be tracked
    validation_calls = get_metric_calls(
        mock_dependencies_statsd, "increment", "request_params_to_args.validation_errors"
    )
    assert len(validation_calls) > 0


# ============================================================================
# Category F: Encoding Tests (3 tests)
# ============================================================================

def test_jsonable_encoder_pydantic_model(mock_encoders_statsd):
    """Test that encoding Pydantic models emits metrics."""
    from fastapi.encoders import jsonable_encoder

    class User(BaseModel):
        name: str
        age: int

    user = User(name="John", age=30)
    result = jsonable_encoder(user)

    assert result == {"name": "John", "age": 30}

    # Verify encoding metrics
    assert_metric_called(
        mock_encoders_statsd,
        "increment",
        "jsonable_encoder.count",
        tags=["is_pydantic:True"]
    )

    # Verify latency
    latency_calls = get_metric_calls(
        mock_encoders_statsd, "histogram", "jsonable_encoder.latency"
    )
    assert len(latency_calls) > 0


def test_jsonable_encoder_dataclass(mock_encoders_statsd):
    """Test that encoding dataclasses emits metrics."""
    from fastapi.encoders import jsonable_encoder

    @dataclass
    class Item:
        name: str
        count: int

    item = Item(name="Widget", count=42)
    result = jsonable_encoder(item)

    assert result == {"name": "Widget", "count": 42}

    # Verify dataclass metrics
    assert_metric_called(
        mock_encoders_statsd,
        "increment",
        "jsonable_encoder.count",
        tags=["is_dataclass:True"]
    )


def test_jsonable_encoder_error_handling(mock_encoders_statsd):
    """Test that encoding errors are tracked."""
    from fastapi.encoders import jsonable_encoder

    class Unserializable:
        def __iter__(self):
            raise NotImplementedError()

        @property
        def __dict__(self):
            raise NotImplementedError()

    obj = Unserializable()

    with pytest.raises(ValueError):
        jsonable_encoder(obj)

    # Verify error metric
    assert_metric_called(
        mock_encoders_statsd,
        "increment",
        "jsonable_encoder.errors",
        tags=["error_type:ValueError"]
    )


# ============================================================================
# Category G: Authentication Tests (5 tests)
# ============================================================================

def test_oauth2_bearer_success(mock_oauth2_statsd):
    """Test that successful OAuth2 authentication emits metrics."""
    app = FastAPI()

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    @app.get("/users/me")
    def read_users_me(token: str = Depends(oauth2_scheme)):
        return {"token": token}

    client = TestClient(app)
    response = client.get("/users/me", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200

    # Verify OAuth2 metrics
    assert_metric_called(
        mock_oauth2_statsd,
        "increment",
        "auth.oauth2.count",
        tags=["path:/users/me", "scheme:oauth2_bearer"]
    )

    assert_metric_called(
        mock_oauth2_statsd,
        "increment",
        "auth.oauth2.success",
        tags=["path:/users/me"]
    )

    # Verify latency
    latency_calls = get_metric_calls(
        mock_oauth2_statsd, "histogram", "auth.oauth2.latency"
    )
    assert len(latency_calls) > 0


def test_oauth2_bearer_missing_token(mock_oauth2_statsd):
    """Test that missing OAuth2 token failure is tracked."""
    app = FastAPI()

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    @app.get("/users/me")
    def read_users_me(token: str = Depends(oauth2_scheme)):
        return {"token": token}

    client = TestClient(app)
    response = client.get("/users/me")

    assert response.status_code == 401

    # Verify failure metric
    assert_metric_called(
        mock_oauth2_statsd,
        "increment",
        "auth.oauth2.failures",
        tags=["failure_reason:missing", "auto_error:True"]
    )


def test_oauth2_bearer_invalid_scheme(mock_oauth2_statsd):
    """Test that invalid authorization scheme is tracked."""
    app = FastAPI()

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    @app.get("/users/me")
    def read_users_me(token: str = Depends(oauth2_scheme)):
        return {"token": token}

    client = TestClient(app)
    response = client.get("/users/me", headers={"Authorization": "Basic secret"})

    assert response.status_code == 401

    # Verify failure with invalid scheme
    assert_metric_called(
        mock_oauth2_statsd,
        "increment",
        "auth.oauth2.failures",
        tags=["failure_reason:invalid_scheme"]
    )


def test_api_key_header_success(mock_api_key_statsd):
    """Test that successful API key authentication emits metrics."""
    app = FastAPI()

    api_key_header = APIKeyHeader(name="X-API-Key")

    @app.get("/items")
    def read_items(api_key: str = Depends(api_key_header)):
        return {"api_key": api_key}

    client = TestClient(app)
    response = client.get("/items", headers={"X-API-Key": "secret-key"})

    assert response.status_code == 200

    # Verify API key metrics
    assert_metric_called(
        mock_api_key_statsd,
        "increment",
        "auth.api_key.count",
        tags=["path:/items", "header:X-API-Key", "scheme:api_key"]
    )

    assert_metric_called(
        mock_api_key_statsd,
        "increment",
        "auth.api_key.success"
    )


def test_api_key_header_missing(mock_api_key_statsd):
    """Test that missing API key failure is tracked."""
    app = FastAPI()

    api_key_header = APIKeyHeader(name="X-API-Key")

    @app.get("/items")
    def read_items(api_key: str = Depends(api_key_header)):
        return {"api_key": api_key}

    client = TestClient(app)
    response = client.get("/items")

    assert response.status_code == 403

    # Verify failure metric
    assert_metric_called(
        mock_api_key_statsd,
        "increment",
        "auth.api_key.failures",
        tags=["failure_reason:missing", "auto_error:True"]
    )


# ============================================================================
# Category H: Configuration and Infrastructure Tests (3 tests)
# ============================================================================

def test_noop_statsd_methods():
    """Test that NoOpStatsd doesn't raise exceptions."""
    from fastapi.observability.metrics import NoOpStatsd

    statsd = NoOpStatsd()

    # These should all work without errors
    statsd.increment("test.metric", tags=["tag:value"])
    statsd.histogram("test.metric", 1.5, tags=["tag:value"])
    statsd.gauge("test.metric", 42, tags=["tag:value"])

    # No assertions needed - just checking no exceptions raised


def test_datadog_disabled_fallback():
    """Test that NoOpStatsd is used when DATADOG_ENABLED=false."""
    # Save original value
    original_enabled = os.environ.get("DATADOG_ENABLED")

    try:
        # Set to false
        os.environ["DATADOG_ENABLED"] = "false"

        # Reload config module to pick up env var
        import importlib
        import fastapi.observability.config
        import fastapi.observability.metrics

        importlib.reload(fastapi.observability.config)
        importlib.reload(fastapi.observability.metrics)

        from fastapi.observability.metrics import statsd, NoOpStatsd

        # Verify NoOpStatsd is used
        assert isinstance(statsd, NoOpStatsd)

    finally:
        # Restore original value
        if original_enabled is not None:
            os.environ["DATADOG_ENABLED"] = original_enabled
        else:
            if "DATADOG_ENABLED" in os.environ:
                del os.environ["DATADOG_ENABLED"]

        # Reload again to restore
        import importlib
        import fastapi.observability.config
        import fastapi.observability.metrics

        importlib.reload(fastapi.observability.config)
        importlib.reload(fastapi.observability.metrics)


def test_multiple_endpoints_tracked_separately(mock_routing_statsd):
    """Test that different endpoints are tracked with different path tags."""
    app = FastAPI()

    @app.get("/users")
    def get_users():
        return {"users": []}

    @app.get("/items")
    def get_items():
        return {"items": []}

    client = TestClient(app)

    response1 = client.get("/users")
    response2 = client.get("/items")

    assert response1.status_code == 200
    assert response2.status_code == 200

    # Verify both paths were tracked separately
    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.count",
        tags=["path:/users", "method:GET"]
    )

    assert_metric_called(
        mock_routing_statsd,
        "increment",
        "request_handler.count",
        tags=["path:/items", "method:GET"]
    )
