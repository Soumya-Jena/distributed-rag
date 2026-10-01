"""Small, dependency-free resilience primitives for the RAG service boundary."""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import BoundedSemaphore, Lock

from src.runtime_metrics import (
    BULKHEAD_REJECTIONS,
    CIRCUIT_STATE,
    RESILIENCE_EVENTS,
)


class ServiceMode(str, Enum):
    NORMAL = "normal"
    DEGRADED_CACHE = "degraded_cache"
    DEGRADED_QUERY = "degraded_query"
    DEGRADED_VECTOR_ONLY = "degraded_vector_only"
    DEGRADED_LEXICAL_ONLY = "degraded_lexical_only"
    DEGRADED_NO_RERANK = "degraded_no_rerank"
    DEGRADED_RAW_CONTEXT = "degraded_raw_context"
    UNAVAILABLE_RETRIEVAL = "unavailable_retrieval"
    UNAVAILABLE_GENERATION = "unavailable_generation"
    BLOCKED_SECURITY = "blocked_security"


class DependencyUnavailableError(RuntimeError):
    mode = ServiceMode.UNAVAILABLE_RETRIEVAL


class RetrievalUnavailableError(DependencyUnavailableError):
    pass


class GenerationUnavailableError(DependencyUnavailableError):
    mode = ServiceMode.UNAVAILABLE_GENERATION


class SecurityUnavailableError(DependencyUnavailableError):
    mode = ServiceMode.BLOCKED_SECURITY


class CircuitOpenError(DependencyUnavailableError):
    pass


class BulkheadFullError(DependencyUnavailableError):
    pass


class DeadlineExceededError(DependencyUnavailableError):
    def __init__(self, message, mode=ServiceMode.UNAVAILABLE_RETRIEVAL):
        super().__init__(message)
        self.mode = mode


class TransientDependencyError(RuntimeError):
    pass


@dataclass
class ResilienceState:
    modes: list[ServiceMode] = field(default_factory=list)
    degraded_components: list[str] = field(default_factory=list)

    def degrade(self, mode, component, *, record=True):
        if mode not in self.modes:
            self.modes.append(mode)
        if component not in self.degraded_components:
            self.degraded_components.append(component)
            if record:
                RESILIENCE_EVENTS.labels(mode=mode.value, component=component).inc()

    @property
    def status(self):
        return "degraded" if self.modes else "normal"


class Deadline:
    def __init__(self, seconds, clock=time.monotonic):
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.clock = clock
        self.expires_at = clock() + seconds

    def remaining(self):
        return max(0.0, self.expires_at - self.clock())

    def expired(self):
        return self.remaining() <= 0

    def require(self, stage="request"):
        if self.expired():
            mode = (
                ServiceMode.UNAVAILABLE_GENERATION
                if stage == "generation" else ServiceMode.UNAVAILABLE_RETRIEVAL
            )
            raise DeadlineExceededError(
                f"Request deadline exhausted before {stage}", mode=mode
            )


def retry_transient(operation, *, attempts=2, base_delay=0.05, sleeper=time.sleep, rng=None):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    rng = rng or random.Random()
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except TransientDependencyError as error:
            last_error = error
            if attempt == attempts - 1:
                break
            backoff = base_delay * (2**attempt)
            sleeper(backoff + rng.uniform(0, backoff))
    raise last_error


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold=5,
        recovery_seconds=10,
        clock=time.monotonic,
        name="dependency",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self.failures = 0
        self.opened_at = None
        self.state = "closed"
        self.name = name
        self._lock = Lock()
        CIRCUIT_STATE.labels(dependency=self.name).set(0)

    def _set_state(self, state):
        self.state = state
        CIRCUIT_STATE.labels(dependency=self.name).set({
            "closed": 0,
            "half_open": 0.5,
            "open": 1,
        }[state])

    def call(self, operation):
        with self._lock:
            if self.state == "open":
                if self.clock() - self.opened_at < self.recovery_seconds:
                    raise CircuitOpenError("Dependency circuit is open")
                self._set_state("half_open")
        try:
            value = operation()
        except Exception:
            with self._lock:
                self.failures += 1
                if self.state == "half_open" or self.failures >= self.failure_threshold:
                    self._set_state("open")
                    self.opened_at = self.clock()
            raise
        with self._lock:
            self.failures = 0
            self.opened_at = None
            self._set_state("closed")
        return value


class Bulkhead:
    def __init__(self, slots, acquire_timeout=1.0, name="dependency"):
        if slots < 1:
            raise ValueError("bulkhead slots must be positive")
        self.semaphore = BoundedSemaphore(slots)
        self.acquire_timeout = acquire_timeout
        self.name = name

    def call(self, operation):
        acquired = self.semaphore.acquire(timeout=self.acquire_timeout)
        if not acquired:
            BULKHEAD_REJECTIONS.labels(dependency=self.name).inc()
            raise BulkheadFullError(f"{self.name} bulkhead is full")
        try:
            return operation()
        finally:
            self.semaphore.release()
