# Hypothesis

Splitting generation from the public RAG API and packaging both services in one reproducible, non-root image will make deployment and failure ownership explicit without changing retrieval behavior. The API should remain live but become unready and refuse queries when the hard model dependency fails; Redis loss should remain fail-open. Named volumes should preserve the corpus and model cache across container recreation.

The controlled variables are the existing corpus, embedding model, retrieval configuration, prompt, security validation, and resilience policy. A deterministic fake model is used only for packaging, contract, and failure-injection tests so that model download and generation latency do not obscure deployment behavior.
