# Timeout, retry, circuit, and bulkhead policy

- PostgreSQL connection timeout: 2 seconds normally; the experiment uses 1 second through the proxy.
- Retrieval statement timeout: 500 ms, transaction-local.
- Redis connect/read timeout: 250/500 ms.
- Overall request deadline: 120 seconds.
- Database retry: at most two attempts with exponential backoff and jitter, only for PostgreSQL connection-class (`08xxx`) failures. Query errors and non-idempotent generation are never retried.
- Circuit breaker: opens after five failures and permits a half-open probe after ten seconds. These values are experimental, not production SLOs.
- Bulkheads: generation and reranking each have an independent one-slot semaphore with a one-second acquisition timeout.

The deadline is checked before retrieval, query transformation, context optimization, and generation. A local in-process model call cannot be safely killed by a Python timeout: the deadline prevents starting work after the budget is exhausted, but it does not cancel CPU/GPU computation already running. Process or worker isolation is required for hard cancellation.
