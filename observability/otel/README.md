# Trace export

Set `OTEL_EXPORTER=otlp` and `OTEL_ENDPOINT=http://localhost:4318/v1/traces` before starting the Python process. The compose-managed Jaeger instance accepts OTLP/HTTP on port 4318 and provides its trace UI on port 16686.

Only bounded operational attributes and counts are exported. Do not add raw questions, retrieved content, prompts, or generated answers to spans.
