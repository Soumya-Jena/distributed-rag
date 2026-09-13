# Indirect Prompt Injection and RAG Poisoning

## Research question

Can instructions hidden in retrieved documents alter answers, leak a synthetic marker, or corrupt citations despite trusted application instructions?

## Hypothesis

The selected strict grounding prompt resists some simple attacks but is not a complete security boundary. Explicit untrusted-data delimiting, lightweight detection, and output validation should reduce measured attack success without materially damaging clean-query answers.

## Controlled variables

The isolated corpus, 100/20 chunking, MiniLM embeddings, hybrid retrieval with RRF, no reranker (the measured retrieval winner), top-five context, Qwen2.5-1.5B-Instruct, greedy decoding, questions, and local hardware remain fixed. The only experimental mode variable is baseline, structured, or layered security handling. The synthetic canary is identically present in the trusted system prompt for all three modes.

## Safety boundary

All markers and canaries are synthetic. The normal `ragdb` database and `datasets/raw` corpus are not modified. Security ingestion is restricted to `ragdb_security`.
