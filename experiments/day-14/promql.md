# PromQL runbook

Request rate:

```promql
sum(rate(rag_requests_total[5m]))
```

Error rate:

```promql
sum(rate(rag_requests_total{status="error"}[5m])) / clamp_min(sum(rate(rag_requests_total[5m])), 0.001)
```

End-to-end P95 latency:

```promql
histogram_quantile(0.95, sum by (le) (rate(rag_request_duration_seconds_bucket[5m])))
```

P95 latency by bounded pipeline stage:

```promql
histogram_quantile(0.95, sum by (stage, le) (rate(rag_stage_duration_seconds_bucket[5m])))
```

Cache hit ratio by layer:

```promql
sum by (cache) (rate(rag_cache_operations_total{result="hit"}[5m])) / clamp_min(sum by (cache) (rate(rag_cache_operations_total{result=~"hit|miss"}[5m])), 0.001)
```

Evidence refusals:

```promql
rate(rag_refusals_total[5m])
```
