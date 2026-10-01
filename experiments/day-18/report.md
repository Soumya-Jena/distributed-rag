# Resilience engineering report

## Result

All ten recorded failure scenarios passed. The deterministic matrix confirmed lexical-only, vector-only, no-rerank, hard retrieval failure, and circuit recovery behavior. The real Toxiproxy run measured:

| Scenario | Observed result | Elapsed |
|---|---|---:|
| Healthy PostgreSQL proxy | `SELECT 1`, normal | 41.264 ms |
| PostgreSQL +750 ms downstream latency | bounded connection timeout, unavailable retrieval | 2046.045 ms |
| PostgreSQL proxy disabled | controlled operational failure | 13.474 ms |
| Redis +750 ms downstream latency | cache timeout and fail-open degradation | 515.248 ms |
| Dependencies restored | `SELECT 1`, normal | 37.491 ms |

The complete machine-readable evidence is in `failure-matrix.csv`. The resilience-focused unit suite covers development-only fault guards, deadlines, transient-only retry, circuit opening/recovery, bulkhead rejection, embedding/vector/lexical fallbacks, healthy empty retrieval, loss of both retrieval branches, raw-context fallback, hard generation failure, fail-closed security, fail-open observability, and the HTTP 503 contract. The complete project suite passed: 76 tests in 12.99 seconds (one upstream Starlette/httpx deprecation warning).

## Observation

The hypothesis was supported for the tested boundaries. A single retrieval branch failure retained evidence-based service, while the service refused to generate when both branches were unavailable. Redis latency stayed close to its configured 500 ms read timeout and did not make the request depend on cache availability. PostgreSQL handshake traffic under a 750 ms downstream toxic accumulated enough delay to hit the bounded connection timeout; this is more severe than adding 750 ms to one SQL statement and illustrates why injected network latency must be measured rather than assumed.

The recovery checks matter as much as the failure checks: proxies were restored after every toxic, PostgreSQL returned to normal, and the circuit breaker moved from open through a successful half-open probe back to closed without a retry storm.

## Limitations

- The corpus and answer-quality baseline were not rerun because these changes alter failure handling, not ranking configuration.
- Toxiproxy exercises network behavior, not CPU starvation, disk exhaustion, or a killed local model process.
- A Python deadline cannot cancel generation already executing in-process.
- Circuit thresholds and bulkhead sizes are experimental values and require production traffic/SLO evidence before deployment.
- HTTP 503 behavior is verified at the API boundary; this lab does not yet include a production load balancer or multi-instance failover.
