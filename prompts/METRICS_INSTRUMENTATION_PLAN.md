# FastAPI Datadog Metrics Instrumentation Plan

This document outlines a comprehensive plan to instrument the FastAPI codebase with Datadog metrics. For each of the 10 most critical methods, we will emit three metrics:
1. **Error count** - Track failures and exceptions
2. **Latency** - Measure execution time
3. **Request count** - Track invocation frequency

## Prerequisites

This instrumentation assumes the following Datadog setup:
```python
from datadog import DogStatsd
import time
from functools import wraps

# Initialize DogStatsd client (typically done at app startup)
statsd = DogStatsd(host="localhost", port=8125, namespace="fastapi")
```

---

## Section 1: Methods to Instrument

### 1. `get_request_handler().app()` (Main Request Handler)
**File**: `fastapi/routing.py:318`

**Why instrument**: This is the main entry point for all HTTP requests in FastAPI. It orchestrates body parsing, dependency resolution, endpoint execution, and response serialization. Instrumenting this method provides overall request-level metrics that capture the entire request lifecycle, making it the single most important method to monitor for understanding API health, performance, and error rates.

---

### 2. `solve_dependencies()` (Dependency Injection Resolution)
**File**: `fastapi/dependencies/utils.py:606`

**Why instrument**: This async function resolves all dependencies for a request, including path params, query params, headers, cookies, and body parameters. It's responsible for validation, caching, and recursive dependency resolution. Since dependency injection is a core FastAPI feature and can involve complex dependency chains, monitoring this method helps identify performance bottlenecks in dependency resolution and validation errors that might not surface as endpoint errors.

---

### 3. `run_endpoint_function()` (Endpoint Execution)
**File**: `fastapi/routing.py:280`

**Why instrument**: This method executes the actual user-defined endpoint function, either directly (if async) or via a threadpool (if sync). It represents the business logic execution time and is critical for identifying slow endpoints. Instrumenting this separately from the overall request handler allows us to distinguish between framework overhead and actual endpoint logic performance.

---

### 4. `serialize_response()` (Response Serialization)
**File**: `fastapi/routing.py:219`

**Why instrument**: This async function handles response validation against response models and serialization to JSON. It can be a significant source of latency for endpoints returning large or complex data structures. Tracking serialization errors and latency helps identify issues with response models, validation failures, and performance problems related to data transformation.

---

### 5. `request_body_to_args()` (Request Body Parsing)
**File**: `fastapi/dependencies/utils.py:938`

**Why instrument**: This async function processes request bodies, handling both JSON and form data (including file uploads). It performs validation of body parameters against Pydantic models. Body parsing is often a source of validation errors and can be slow for large payloads or file uploads, making it essential to monitor for both error rates and latency.

---

### 6. `request_params_to_args()` (Parameter Extraction)
**File**: `fastapi/dependencies/utils.py:774`

**Why instrument**: This function extracts and validates path parameters, query parameters, headers, and cookies. It handles multidict extraction and Pydantic field validation. Since this is where most 422 validation errors originate (invalid query params, missing headers, etc.), monitoring this method helps identify client-side issues and problematic API usage patterns.

---

### 7. `jsonable_encoder()` (JSON Encoding)
**File**: `fastapi/encoders.py:109`

**Why instrument**: This recursive function converts Python objects to JSON-serializable types, handling Pydantic models, dataclasses, enums, dates, UUIDs, and other complex types. It's used extensively in response serialization and can be computationally expensive for deeply nested objects. Monitoring encoding latency helps identify performance issues with complex response structures.

---

### 8. `OAuth2PasswordBearer.__call__()` (OAuth2 Authentication)
**File**: `fastapi/security/oauth2.py:488`

**Why instrument**: This method validates OAuth2 bearer tokens by extracting and verifying the Authorization header. Authentication is critical for security, and monitoring auth attempts, failures, and latency helps detect attacks (brute force, credential stuffing), identify auth service issues, and understand authentication patterns across your API.

---

### 9. `APIKeyHeader.__call__()` (API Key Authentication)
**File**: `fastapi/security/api_key.py:198`

**Why instrument**: This method validates API keys from request headers. Like OAuth2, tracking API key authentication is crucial for security monitoring. It helps identify invalid API key usage, potential abuse, and helps with API key lifecycle management (detecting keys that generate high error rates).

---

### 10. `get_dependant()` (Dependency Analysis)
**File**: `fastapi/dependencies/utils.py:277`

**Why instrument**: This function analyzes endpoint function signatures and builds the dependency graph. It extracts parameter annotations, processes Pydantic fields, and constructs the Dependant object that drives dependency resolution. While typically cached, understanding dependency analysis performance helps optimize application startup time and identify issues with complex dependency structures.

---

## Section 2: Implementation Code Snippets

### Helper Utilities

First, create a utility module for metric decorators:

```python
# File: fastapi/observability/metrics.py
"""
Datadog metrics instrumentation utilities for FastAPI.
"""
from datadog import DogStatsd
import time
from functools import wraps
from typing import Callable, Any, Optional
import asyncio

# Initialize DogStatsd client
statsd = DogStatsd(host="localhost", port=8125, namespace="fastapi")


def instrument_sync(
    metric_prefix: str,
    tags: Optional[list] = None
) -> Callable:
    """
    Decorator to instrument synchronous functions with Datadog metrics.
    Emits: request count, latency, and error count.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
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


def instrument_async(
    metric_prefix: str,
    tags: Optional[list] = None
) -> Callable:
    """
    Decorator to instrument asynchronous functions with Datadog metrics.
    Emits: request count, latency, and error count.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
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
```

---

### 1. Instrumenting `get_request_handler().app()` (Main Request Handler)

**Location**: `fastapi/routing.py` - inside `get_request_handler()` function

**Implementation**:

```python
def get_request_handler(
    dependant: Dependant,
    body_field: Optional[ModelField] = None,
    status_code: Optional[int] = None,
    response_class: Union[Type[Response], DefaultPlaceholder] = Default(JSONResponse),
    response_field: Optional[ModelField] = None,
    response_model_include: Optional[IncEx] = None,
    response_model_exclude: Optional[IncEx] = None,
    response_model_by_alias: bool = True,
    response_model_exclude_unset: bool = False,
    response_model_exclude_defaults: bool = False,
    response_model_exclude_none: bool = False,
    dependency_overrides_provider: Optional[Any] = None,
    embed_body_fields: bool = False,
) -> Callable[[Request], Coroutine[Any, Any, Response]]:
    # ... existing setup code ...

    async def app(request: Request) -> Response:
        # Import metrics utilities
        from fastapi.observability.metrics import statsd
        import time

        # Extract route information for tags
        route_path = request.scope.get("route", {}).get("path", "unknown")
        method = request.method
        tags = [f"path:{route_path}", f"method:{method}"]

        start_time = time.time()
        statsd.increment("request_handler.count", tags=tags)

        try:
            # ... existing request handling code ...

            # After response is created, add status code tag
            if response:
                status_tags = tags + [f"status_code:{response.status_code}"]
                statsd.increment("request_handler.count", tags=status_tags)

            return response

        except HTTPException as e:
            # Track HTTP exceptions
            error_tags = tags + [
                f"error_type:HTTPException",
                f"status_code:{e.status_code}"
            ]
            statsd.increment("request_handler.errors", tags=error_tags)
            raise

        except RequestValidationError as e:
            # Track validation errors
            error_tags = tags + [
                f"error_type:RequestValidationError",
                f"status_code:422"
            ]
            statsd.increment("request_handler.errors", tags=error_tags)
            raise

        except Exception as e:
            # Track unexpected errors
            error_tags = tags + [
                f"error_type:{type(e).__name__}",
                f"status_code:500"
            ]
            statsd.increment("request_handler.errors", tags=error_tags)
            raise

        finally:
            # Record request latency
            duration = time.time() - start_time
            statsd.histogram("request_handler.latency", duration, tags=tags)

    return app
```

**Metrics emitted**:
- `fastapi.request_handler.count` - tagged by path, method, status_code
- `fastapi.request_handler.errors` - tagged by path, method, error_type, status_code
- `fastapi.request_handler.latency` - histogram tagged by path, method

---

### 2. Instrumenting `solve_dependencies()` (Dependency Resolution)

**Location**: `fastapi/dependencies/utils.py:606`

**Implementation**:

```python
async def solve_dependencies(
    *,
    request: Union[Request, WebSocket],
    dependant: Dependant,
    body: Optional[Union[Dict[str, Any], FormData]] = None,
    background_tasks: Optional[StarletteBackgroundTasks] = None,
    response: Optional[Response] = None,
    dependency_overrides_provider: Optional[Any] = None,
    dependency_cache: Optional[Dict[Tuple[Callable[..., Any], Tuple[str]], Any]] = None,
    async_exit_stack: AsyncExitStack,
    embed_body_fields: bool,
) -> SolvedDependency:
    from fastapi.observability.metrics import statsd
    import time

    # Get route path for tagging
    route_path = "unknown"
    if hasattr(request, "scope"):
        route_path = request.scope.get("route", {}).get("path", "unknown")

    tags = [f"path:{route_path}"]
    start_time = time.time()

    # Track dependency count
    dep_count = len(dependant.dependencies)
    tags_with_count = tags + [f"dependency_count:{dep_count}"]
    statsd.increment("solve_dependencies.count", tags=tags_with_count)

    try:
        # ... existing dependency resolution code ...

        # Track cache hits
        if dependency_cache:
            cache_size = len(dependency_cache)
            statsd.gauge("solve_dependencies.cache_size", cache_size, tags=tags)

        # ... rest of existing code ...

        # Track validation errors
        if errors:
            error_count = len(errors)
            error_tags = tags + [f"error_count:{error_count}"]
            statsd.increment("solve_dependencies.validation_errors", tags=error_tags)

        return SolvedDependency(...)

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("solve_dependencies.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("solve_dependencies.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.solve_dependencies.count` - tagged by path, dependency_count
- `fastapi.solve_dependencies.errors` - tagged by path, error_type
- `fastapi.solve_dependencies.validation_errors` - tagged by path, error_count
- `fastapi.solve_dependencies.latency` - histogram tagged by path
- `fastapi.solve_dependencies.cache_size` - gauge tagged by path

---

### 3. Instrumenting `run_endpoint_function()` (Endpoint Execution)

**Location**: `fastapi/routing.py:280`

**Implementation**:

```python
async def run_endpoint_function(
    *, dependant: Dependant, values: Dict[str, Any], is_coroutine: bool
) -> Any:
    from fastapi.observability.metrics import statsd
    import time

    # Get endpoint name for tagging
    endpoint_name = "unknown"
    if dependant.call:
        endpoint_name = f"{dependant.call.__module__}.{dependant.call.__name__}"

    tags = [
        f"endpoint:{endpoint_name}",
        f"is_coroutine:{is_coroutine}"
    ]

    start_time = time.time()
    statsd.increment("endpoint_execution.count", tags=tags)

    try:
        assert dependant.call is not None, "dependant.call must be a function"

        if is_coroutine:
            result = await dependant.call(**values)
        else:
            result = await run_in_threadpool(dependant.call, **values)

        return result

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("endpoint_execution.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("endpoint_execution.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.endpoint_execution.count` - tagged by endpoint, is_coroutine
- `fastapi.endpoint_execution.errors` - tagged by endpoint, is_coroutine, error_type
- `fastapi.endpoint_execution.latency` - histogram tagged by endpoint, is_coroutine

---

### 4. Instrumenting `serialize_response()` (Response Serialization)

**Location**: `fastapi/routing.py:219`

**Implementation**:

```python
async def serialize_response(
    *,
    field: Optional[ModelField] = None,
    response_content: Any,
    include: Optional[IncEx] = None,
    exclude: Optional[IncEx] = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    is_coroutine: bool = True,
) -> Any:
    from fastapi.observability.metrics import statsd
    import time
    import sys

    # Determine response type for tagging
    response_type = type(response_content).__name__
    has_response_model = field is not None

    tags = [
        f"response_type:{response_type}",
        f"has_model:{has_response_model}"
    ]

    start_time = time.time()
    statsd.increment("serialize_response.count", tags=tags)

    try:
        if field:
            errors = []
            if not hasattr(field, "serialize"):
                response_content = _prepare_response_content(
                    response_content,
                    exclude_unset=exclude_unset,
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                )

            if is_coroutine:
                value, errors_ = field.validate(response_content, {}, loc=("response",))
            else:
                value, errors_ = await run_in_threadpool(
                    field.validate, response_content, {}, loc=("response",)
                )

            if isinstance(errors_, list):
                errors.extend(errors_)
            elif errors_:
                errors.append(errors_)

            if errors:
                # Track validation errors
                error_tags = tags + [f"error_count:{len(errors)}"]
                statsd.increment("serialize_response.validation_errors", tags=error_tags)
                raise ResponseValidationError(
                    errors=_normalize_errors(errors), body=response_content
                )

            if hasattr(field, "serialize"):
                result = field.serialize(
                    value,
                    include=include,
                    exclude=exclude,
                    by_alias=by_alias,
                    exclude_unset=exclude_unset,
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                )
            else:
                result = jsonable_encoder(
                    value,
                    include=include,
                    exclude=exclude,
                    by_alias=by_alias,
                    exclude_unset=exclude_unset,
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                )

            # Track response size
            response_size = sys.getsizeof(result)
            statsd.histogram("serialize_response.size", response_size, tags=tags)

            return result
        else:
            result = jsonable_encoder(response_content)
            response_size = sys.getsizeof(result)
            statsd.histogram("serialize_response.size", response_size, tags=tags)
            return result

    except ResponseValidationError:
        raise  # Already tracked above

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("serialize_response.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("serialize_response.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.serialize_response.count` - tagged by response_type, has_model
- `fastapi.serialize_response.errors` - tagged by response_type, has_model, error_type
- `fastapi.serialize_response.validation_errors` - tagged by response_type, has_model, error_count
- `fastapi.serialize_response.latency` - histogram tagged by response_type, has_model
- `fastapi.serialize_response.size` - histogram of response size in bytes

---

### 5. Instrumenting `request_body_to_args()` (Request Body Parsing)

**Location**: `fastapi/dependencies/utils.py:938`

**Implementation**:

```python
async def request_body_to_args(
    body_fields: List[ModelField],
    received_body: Optional[Union[Dict[str, Any], FormData]],
    embed_body_fields: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from fastapi.observability.metrics import statsd
    import time
    import sys

    # Determine body type
    body_type = "none"
    if received_body is not None:
        body_type = "form" if isinstance(received_body, FormData) else "json"

    field_count = len(body_fields)
    tags = [
        f"body_type:{body_type}",
        f"field_count:{field_count}",
        f"embedded:{embed_body_fields}"
    ]

    start_time = time.time()
    statsd.increment("request_body_to_args.count", tags=tags)

    # Track body size if available
    if received_body is not None:
        body_size = sys.getsizeof(received_body)
        statsd.histogram("request_body_to_args.body_size", body_size, tags=tags)

    try:
        values: Dict[str, Any] = {}
        errors: List[Dict[str, Any]] = []

        assert body_fields, "request_body_to_args() should be called with fields"

        # ... existing body parsing logic ...

        # Track validation errors
        if errors:
            error_tags = tags + [f"error_count:{len(errors)}"]
            statsd.increment("request_body_to_args.validation_errors", tags=error_tags)

        return values, errors

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("request_body_to_args.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("request_body_to_args.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.request_body_to_args.count` - tagged by body_type, field_count, embedded
- `fastapi.request_body_to_args.errors` - tagged by body_type, field_count, embedded, error_type
- `fastapi.request_body_to_args.validation_errors` - tagged by body_type, field_count, embedded, error_count
- `fastapi.request_body_to_args.latency` - histogram tagged by body_type, field_count, embedded
- `fastapi.request_body_to_args.body_size` - histogram of request body size

---

### 6. Instrumenting `request_params_to_args()` (Parameter Extraction)

**Location**: `fastapi/dependencies/utils.py:774`

**Implementation**:

```python
def request_params_to_args(
    fields: Sequence[ModelField],
    received_params: Union[Mapping[str, Any], QueryParams, Headers],
) -> Tuple[Dict[str, Any], List[Any]]:
    from fastapi.observability.metrics import statsd
    import time

    # Determine parameter location
    param_location = "query"
    if isinstance(received_params, Headers):
        param_location = "headers"
    elif hasattr(received_params, "__class__"):
        if "path" in received_params.__class__.__name__.lower():
            param_location = "path"

    field_count = len(fields)
    tags = [
        f"param_location:{param_location}",
        f"field_count:{field_count}"
    ]

    start_time = time.time()
    statsd.increment("request_params_to_args.count", tags=tags)

    try:
        values: Dict[str, Any] = {}
        errors: List[Dict[str, Any]] = []

        if not fields:
            return values, errors

        # ... existing parameter extraction logic ...

        # Track validation errors by location
        if errors:
            error_tags = tags + [f"error_count:{len(errors)}"]
            statsd.increment("request_params_to_args.validation_errors", tags=error_tags)

        return values, errors

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("request_params_to_args.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("request_params_to_args.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.request_params_to_args.count` - tagged by param_location, field_count
- `fastapi.request_params_to_args.errors` - tagged by param_location, field_count, error_type
- `fastapi.request_params_to_args.validation_errors` - tagged by param_location, field_count, error_count
- `fastapi.request_params_to_args.latency` - histogram tagged by param_location, field_count

---

### 7. Instrumenting `jsonable_encoder()` (JSON Encoding)

**Location**: `fastapi/encoders.py:109`

**Implementation**:

```python
def jsonable_encoder(
    obj: Any,
    include: Optional[IncEx] = None,
    exclude: Optional[IncEx] = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    custom_encoder: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    sqlalchemy_safe: bool = True,
) -> Any:
    from fastapi.observability.metrics import statsd
    import time
    import sys

    # Determine object type for tagging
    obj_type = type(obj).__name__
    is_pydantic = isinstance(obj, (BaseModel, v1.BaseModel))
    is_dataclass = dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    tags = [
        f"obj_type:{obj_type}",
        f"is_pydantic:{is_pydantic}",
        f"is_dataclass:{is_dataclass}"
    ]

    start_time = time.time()
    statsd.increment("jsonable_encoder.count", tags=tags)

    try:
        # ... existing encoding logic ...

        # For recursive calls, we only want to track at the top level
        # to avoid double-counting. Use a context variable or check call stack depth
        result = _jsonable_encoder_impl(
            obj, include, exclude, by_alias, exclude_unset,
            exclude_defaults, exclude_none, custom_encoder, sqlalchemy_safe
        )

        # Track encoded size
        encoded_size = sys.getsizeof(result)
        statsd.histogram("jsonable_encoder.output_size", encoded_size, tags=tags)

        return result

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("jsonable_encoder.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("jsonable_encoder.latency", duration, tags=tags)


# Rename original implementation
def _jsonable_encoder_impl(
    obj: Any,
    include: Optional[IncEx] = None,
    exclude: Optional[IncEx] = None,
    by_alias: bool = True,
    exclude_unset: bool = False,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    custom_encoder: Optional[Dict[Any, Callable[[Any], Any]]] = None,
    sqlalchemy_safe: bool = True,
) -> Any:
    """Original jsonable_encoder implementation without metrics."""
    # ... original implementation code ...
```

**Note**: For `jsonable_encoder`, since it's recursive, we should only instrument the top-level calls to avoid metric inflation. The implementation above shows wrapping the entry point while the recursive calls use an internal `_jsonable_encoder_impl` function.

**Metrics emitted**:
- `fastapi.jsonable_encoder.count` - tagged by obj_type, is_pydantic, is_dataclass
- `fastapi.jsonable_encoder.errors` - tagged by obj_type, is_pydantic, is_dataclass, error_type
- `fastapi.jsonable_encoder.latency` - histogram tagged by obj_type, is_pydantic, is_dataclass
- `fastapi.jsonable_encoder.output_size` - histogram of encoded output size

---

### 8. Instrumenting `OAuth2PasswordBearer.__call__()` (OAuth2 Authentication)

**Location**: `fastapi/security/oauth2.py:488`

**Implementation**:

```python
class OAuth2PasswordBearer(OAuth2):
    # ... existing __init__ and other methods ...

    async def __call__(self, request: Request) -> Optional[str]:
        from fastapi.observability.metrics import statsd
        import time

        # Get route information for tagging
        route_path = request.scope.get("route", {}).get("path", "unknown")
        tags = [
            f"path:{route_path}",
            f"scheme:oauth2_bearer"
        ]

        start_time = time.time()
        statsd.increment("auth.oauth2.count", tags=tags)

        try:
            authorization = request.headers.get("Authorization")
            scheme, param = get_authorization_scheme_param(authorization)

            if not authorization or scheme.lower() != "bearer":
                # Track authentication failures
                failure_reason = "missing" if not authorization else "invalid_scheme"
                failure_tags = tags + [
                    f"failure_reason:{failure_reason}",
                    f"auto_error:{self.auto_error}"
                ]
                statsd.increment("auth.oauth2.failures", tags=failure_tags)

                if self.auto_error:
                    raise HTTPException(
                        status_code=HTTP_401_UNAUTHORIZED,
                        detail="Not authenticated",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                else:
                    return None

            # Track successful token extraction
            statsd.increment("auth.oauth2.success", tags=tags)
            return param

        except HTTPException:
            raise  # Already tracked above

        except Exception as e:
            error_tags = tags + [f"error_type:{type(e).__name__}"]
            statsd.increment("auth.oauth2.errors", tags=error_tags)
            raise

        finally:
            duration = time.time() - start_time
            statsd.histogram("auth.oauth2.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.auth.oauth2.count` - tagged by path, scheme
- `fastapi.auth.oauth2.success` - tagged by path, scheme
- `fastapi.auth.oauth2.failures` - tagged by path, scheme, failure_reason, auto_error
- `fastapi.auth.oauth2.errors` - tagged by path, scheme, error_type
- `fastapi.auth.oauth2.latency` - histogram tagged by path, scheme

---

### 9. Instrumenting `APIKeyHeader.__call__()` (API Key Authentication)

**Location**: `fastapi/security/api_key.py:198`

**Implementation**:

```python
class APIKeyHeader(APIKeyBase):
    # ... existing __init__ ...

    async def __call__(self, request: Request) -> Optional[str]:
        from fastapi.observability.metrics import statsd
        import time

        # Get route and header name for tagging
        route_path = request.scope.get("route", {}).get("path", "unknown")
        header_name = self.model.name

        tags = [
            f"path:{route_path}",
            f"header:{header_name}",
            f"scheme:api_key"
        ]

        start_time = time.time()
        statsd.increment("auth.api_key.count", tags=tags)

        try:
            api_key = request.headers.get(self.model.name)

            if not api_key:
                # Track missing API key
                failure_tags = tags + [
                    f"failure_reason:missing",
                    f"auto_error:{self.auto_error}"
                ]
                statsd.increment("auth.api_key.failures", tags=failure_tags)

                if self.auto_error:
                    raise HTTPException(
                        status_code=HTTP_403_FORBIDDEN,
                        detail="Not authenticated"
                    )
                return None

            # Track successful API key extraction
            statsd.increment("auth.api_key.success", tags=tags)
            return api_key

        except HTTPException:
            raise  # Already tracked above

        except Exception as e:
            error_tags = tags + [f"error_type:{type(e).__name__}"]
            statsd.increment("auth.api_key.errors", tags=error_tags)
            raise

        finally:
            duration = time.time() - start_time
            statsd.histogram("auth.api_key.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.auth.api_key.count` - tagged by path, header, scheme
- `fastapi.auth.api_key.success` - tagged by path, header, scheme
- `fastapi.auth.api_key.failures` - tagged by path, header, scheme, failure_reason, auto_error
- `fastapi.auth.api_key.errors` - tagged by path, header, scheme, error_type
- `fastapi.auth.api_key.latency` - histogram tagged by path, header, scheme

---

### 10. Instrumenting `get_dependant()` (Dependency Analysis)

**Location**: `fastapi/dependencies/utils.py:277`

**Implementation**:

```python
def get_dependant(
    *,
    path: str,
    call: Callable[..., Any],
    name: Optional[str] = None,
    security_scopes: Optional[List[str]] = None,
    use_cache: bool = True,
) -> Dependant:
    from fastapi.observability.metrics import statsd
    import time

    # Get function info for tagging
    func_name = f"{call.__module__}.{call.__name__}" if hasattr(call, "__name__") else "unknown"

    tags = [
        f"function:{func_name}",
        f"use_cache:{use_cache}"
    ]

    start_time = time.time()
    statsd.increment("get_dependant.count", tags=tags)

    try:
        path_param_names = get_path_param_names(path)
        endpoint_signature = get_typed_signature(call)
        signature_params = endpoint_signature.parameters

        dependant = Dependant(
            call=call,
            name=name,
            path=path,
            security_scopes=security_scopes,
            use_cache=use_cache,
        )

        param_count = len(signature_params)
        dependency_count = 0

        for param_name, param in signature_params.items():
            is_path_param = param_name in path_param_names
            param_details = analyze_param(
                param_name=param_name,
                annotation=param.annotation,
                value=param.default,
                is_path_param=is_path_param,
            )

            if param_details.depends is not None:
                dependency_count += 1
                sub_dependant = get_param_sub_dependant(
                    param_name=param_name,
                    depends=param_details.depends,
                    path=path,
                    security_scopes=security_scopes,
                )
                dependant.dependencies.append(sub_dependant)
                continue

            if add_non_field_param_to_dependency(
                param_name=param_name,
                type_annotation=param_details.type_annotation,
                dependant=dependant,
            ):
                assert param_details.field is None, (
                    f"Cannot specify multiple FastAPI annotations for {param_name!r}"
                )
                continue

            assert param_details.field is not None
            if isinstance(
                param_details.field.field_info, (params.Body, temp_pydantic_v1_params.Body)
            ):
                dependant.body_params.append(param_details.field)
            else:
                add_param_to_fields(field=param_details.field, dependant=dependant)

        # Track complexity metrics
        complexity_tags = tags + [
            f"param_count:{param_count}",
            f"dependency_count:{dependency_count}",
            f"body_param_count:{len(dependant.body_params)}"
        ]
        statsd.gauge("get_dependant.complexity",
                    param_count + dependency_count,
                    tags=complexity_tags)

        return dependant

    except Exception as e:
        error_tags = tags + [f"error_type:{type(e).__name__}"]
        statsd.increment("get_dependant.errors", tags=error_tags)
        raise

    finally:
        duration = time.time() - start_time
        statsd.histogram("get_dependant.latency", duration, tags=tags)
```

**Metrics emitted**:
- `fastapi.get_dependant.count` - tagged by function, use_cache
- `fastapi.get_dependant.errors` - tagged by function, use_cache, error_type
- `fastapi.get_dependant.latency` - histogram tagged by function, use_cache
- `fastapi.get_dependant.complexity` - gauge of total params + dependencies

---

## Configuration and Deployment

### Environment Configuration

```python
# File: fastapi/observability/config.py
"""
Configuration for Datadog metrics.
"""
import os

DATADOG_ENABLED = os.getenv("DATADOG_ENABLED", "true").lower() == "true"
DATADOG_HOST = os.getenv("DATADOG_HOST", "localhost")
DATADOG_PORT = int(os.getenv("DATADOG_PORT", "8125"))
DATADOG_NAMESPACE = os.getenv("DATADOG_NAMESPACE", "fastapi")

# Feature flags for specific metric groups
METRICS_REQUEST_HANDLER = os.getenv("METRICS_REQUEST_HANDLER", "true").lower() == "true"
METRICS_DEPENDENCIES = os.getenv("METRICS_DEPENDENCIES", "true").lower() == "true"
METRICS_SERIALIZATION = os.getenv("METRICS_SERIALIZATION", "true").lower() == "true"
METRICS_AUTH = os.getenv("METRICS_AUTH", "true").lower() == "true"
```

### Datadog Dashboard Configuration

Create a Datadog dashboard with the following widgets:

1. **Request Overview**:
   - `fastapi.request_handler.count` by `path` and `status_code`
   - `fastapi.request_handler.latency` p50, p95, p99 by `path`
   - `fastapi.request_handler.errors` by `error_type`

2. **Dependency Performance**:
   - `fastapi.solve_dependencies.latency` by `path`
   - `fastapi.solve_dependencies.validation_errors` by `path`
   - `fastapi.solve_dependencies.cache_size` gauge

3. **Endpoint Performance**:
   - `fastapi.endpoint_execution.latency` by `endpoint`
   - `fastapi.endpoint_execution.errors` by `endpoint` and `error_type`

4. **Serialization Metrics**:
   - `fastapi.serialize_response.latency` by `response_type`
   - `fastapi.serialize_response.size` distribution
   - `fastapi.jsonable_encoder.latency` by `obj_type`

5. **Authentication Metrics**:
   - `fastapi.auth.oauth2.failures` by `failure_reason`
   - `fastapi.auth.api_key.success` vs `failures`
   - `fastapi.auth.*.latency` comparison

6. **Request Parsing**:
   - `fastapi.request_body_to_args.body_size` distribution
   - `fastapi.request_params_to_args.validation_errors` by `param_location`

### Alerting Recommendations

1. **High Error Rate**: Alert when `request_handler.errors` rate exceeds 5% of total requests
2. **High Latency**: Alert when p95 of `request_handler.latency` exceeds 1 second
3. **Authentication Failures**: Alert when `auth.*.failures` spike suddenly
4. **Validation Errors**: Alert when validation errors exceed expected baseline
5. **Dependency Issues**: Alert when `solve_dependencies.latency` p95 exceeds threshold

---

## Testing the Instrumentation

### Unit Test Example

```python
# File: tests/test_metrics.py
"""
Tests for Datadog metrics instrumentation.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_request_handler_metrics():
    """Test that request handler emits correct metrics."""
    app = FastAPI()

    @app.get("/test")
    def test_endpoint():
        return {"message": "success"}

    client = TestClient(app)

    with patch("fastapi.observability.metrics.statsd") as mock_statsd:
        response = client.get("/test")

        assert response.status_code == 200

        # Verify metrics were called
        mock_statsd.increment.assert_any_call(
            "request_handler.count",
            tags=["path:/test", "method:GET"]
        )
        mock_statsd.histogram.assert_called()


def test_authentication_failure_metrics():
    """Test that auth failures emit correct metrics."""
    from fastapi.security import OAuth2PasswordBearer

    app = FastAPI()
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    @app.get("/protected")
    def protected_endpoint(token: str = Depends(oauth2_scheme)):
        return {"token": token}

    client = TestClient(app)

    with patch("fastapi.observability.metrics.statsd") as mock_statsd:
        response = client.get("/protected")

        assert response.status_code == 401

        # Verify auth failure metrics
        mock_statsd.increment.assert_any_call(
            "auth.oauth2.failures",
            tags=["path:/protected", "scheme:oauth2_bearer",
                  "failure_reason:missing", "auto_error:True"]
        )
```

---

## Performance Considerations

1. **Metric Collection Overhead**: StatsD operations are UDP-based and non-blocking, adding minimal latency (typically <1ms)

2. **Tag Cardinality**: Be cautious with high-cardinality tags (like user IDs). The current implementation uses:
   - Path templates (not actual paths with IDs)
   - Function names
   - Error types
   - This keeps cardinality manageable

3. **Sampling**: For very high-traffic endpoints, consider sampling:
   ```python
   import random
   if random.random() < 0.1:  # 10% sampling
       statsd.histogram("metric", value, tags=tags)
   ```

4. **Async Operations**: All instrumentation uses time.time() which is safe for async contexts

---

## Summary

This instrumentation plan provides comprehensive observability for FastAPI applications:

- **Request-level metrics**: Track overall API health and performance
- **Dependency metrics**: Monitor FastAPI's dependency injection performance
- **Authentication metrics**: Security and auth flow monitoring
- **Serialization metrics**: Track data transformation overhead
- **Validation metrics**: Identify client-side issues and data quality

Each instrumented method emits three core metric types (count, errors, latency) with relevant tags for dimensionality, enabling deep insights into API behavior and performance characteristics.
