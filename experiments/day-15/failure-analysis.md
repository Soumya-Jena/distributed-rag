# Failure analysis

## Genuine local RAG run

The guarded unique-query sweep stopped at concurrency 1. Locust observed 12 attempted requests, 12 failures, zero goodput, and a maximum/P95 observed failure latency of about 118 seconds. The first request entered the full pipeline, but the Uvicorn process exited before returning a valid response; later attempts received connection failures (`HTTP 0`).

Host memory reached 93%. The model startup message reported Qwen parameters offloaded to CPU and disk. PostgreSQL peaked at only 5.82% container CPU and about 38 MiB memory, while Redis used about 9 MiB. This does not prove a precise kernel OOM cause, but it strongly rules against PostgreSQL or Redis as the first observed bottleneck and is consistent with memory pressure plus local generation/offload.

The correct operational conclusion is not “C1 takes 118 seconds.” No successful response completed. The conclusion is that this deployment has **no sustainable concurrency under the stated SLO** on the tested hardware/configuration.

## Safety decision

C2+, repeated-cache, cold-stampede, spike, and soak tests were not run against Qwen because the safety policy requires stopping after process termination or more than 5% errors. Running them would measure crash behavior, not useful capacity.

## Next engineering experiment

Measure again after changing exactly one capacity variable: a smaller/quantized generator, more RAM/GPU VRAM, or a separate model-serving process with explicit queueing. Do not optimize pgvector based on this result.
