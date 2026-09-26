# Experimental SLO and safety policy

The initial capacity SLO is:

- P95 latency at most 10 seconds.
- Error rate below 1%.
- Non-empty response that passes the existing grounding/security path.

Maximum sustainable concurrency is the highest tested level satisfying every condition.

A sweep stops immediately for any of the following:

- error rate above 5%;
- P95 above three times the C1 baseline;
- out-of-memory behavior or process termination;
- sustained host instability;
- throughput flattening while latency grows sharply.

The genuine run stopped at C1 because the API process exited under 93% host-memory pressure and all measured requests failed. C2 and higher, spike, stampede, and soak runs were therefore unsafe and intentionally not attempted.
