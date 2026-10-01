# Hypothesis

If every external dependency has a short timeout, a bounded failure policy, and an explicit fallback, then a single optional-stage failure should return a truthful degraded response rather than a generic error. If retrieval, generation, or security validation cannot safely complete, the service should fail closed with an explicit unavailable mode instead of asking the language model to improvise.

The controlled variables are the existing corpus, chunking, embedding model, hybrid retrieval settings, prompts, and security mode. The independent variable is the injected dependency failure. The measured outcomes are service mode, HTTP contract, elapsed time, recovery, and whether unsupported generation was prevented.
