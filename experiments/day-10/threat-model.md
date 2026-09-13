# Threat model

## Assets

The trusted system instructions, synthetic canary, answer integrity, citation integrity, and the user's original question are protected. The canary is an artificial test marker, not a secret or credential.

## Attacker control

The attacker can edit a document that enters the isolated retrieval corpus. They cannot change the application prompt, user question, model weights, retrieval code, or database configuration. Their text can be retrieved alongside legitimate technical documentation and may appear in the final five source chunks.

## Entry point and trust boundary

Retrieved document content is lower-trust data. It may provide facts, but cannot grant authority, change instructions, request hidden prompt disclosure, or dictate citation choices. The structured mode delimits every excerpt and names this boundary. The layered mode additionally flags suspicious text while retaining it for the experiment, then blocks outputs containing synthetic markers or the canary.

## Attack objectives

Instruction override, prompt/canary extraction, output replacement, citation hijacking, social-engineering claims of prior approval, and factual retrieval poisoning. The keyword-stuffed poison document tests whether malicious content can rank highly even without the model following a direct command.

## Limits

The application currently generates text only and has no shell, email, database-write, or cloud-action tools available to the LLM. That constrains impact. Future tool access would require per-user authorization, least-privilege credentials, allowlisted actions, and human confirmation for high-impact operations. A detector or output guard that catches synthetic markers does not prove safety against paraphrased attacks or unmarked factual manipulation. Documents marked `unreviewed` or even `trusted` in provenance metadata still remain untrusted as instructions.
