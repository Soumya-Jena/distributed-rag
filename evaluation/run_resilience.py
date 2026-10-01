"""Run deterministic resilience checks and optional real network faults."""

import argparse
import csv
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from src.fault_injection import FaultInjector
from src.resilience import CircuitBreaker, CircuitOpenError, RetrievalUnavailableError
from src.retrieval_pipeline import RetrievalPipeline

from evaluation.setup_toxiproxy import ToxiproxyClient, configure


OUTPUT = Path("experiments/day-18/failure-matrix.csv")


def chunk(identifier, vector=True):
    return SimpleNamespace(
        chunk_id=identifier,
        title=identifier,
        source_path=f"{identifier}.md",
        chunk_index=0,
        content=identifier,
        similarity=0.8 if vector else None,
        lexical_score=None if vector else 1.0,
    )


def pipeline(actions, use_reranker=False):
    vector = SimpleNamespace(search=lambda *_args, **_kwargs: [chunk("vector")])
    lexical = SimpleNamespace(
        search=lambda *_args, **_kwargs: [chunk("lexical", vector=False)]
    )
    hybrid = SimpleNamespace(vector_retriever=vector, lexical_retriever=lexical)
    reranker = SimpleNamespace(
        rerank=lambda _query, chunks, top_k: [
            SimpleNamespace(chunk=item) for item in chunks[:top_k]
        ]
    )
    return RetrievalPipeline(
        use_reranker=use_reranker,
        retrieval_mode="hybrid",
        retriever=vector,
        hybrid_retriever=hybrid,
        reranker=reranker,
        fault_injector=FaultInjector(
            enabled=True,
            environment="test",
            actions=actions,
        ),
    )


def record(rows, scenario, expected, operation):
    started = perf_counter()
    try:
        mode, note = operation()
        outcome = "pass"
    except Exception as error:
        mode = type(error).__name__
        note = str(error)
        outcome = "fail"
    rows.append({
        "scenario": scenario,
        "expected": expected,
        "outcome": outcome,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "mode": mode,
        "notes": note,
    })


def in_process_rows():
    rows = []

    def retrieve(actions):
        result = pipeline(actions).retrieve("resilience question")
        return result.service_modes[0].value, result.final_chunks[0].chunk_id

    record(rows, "vector branch unavailable", "lexical-only 200", lambda: retrieve({
        "vector_retrieval": "fail"
    }))
    record(rows, "lexical branch unavailable", "vector-only 200", lambda: retrieve({
        "lexical_retrieval": "fail"
    }))

    def both_down():
        try:
            pipeline({
                "vector_retrieval": "fail",
                "lexical_retrieval": "fail",
            }).retrieve("resilience question")
        except RetrievalUnavailableError:
            return "unavailable_retrieval", "controlled 503 contract"
        raise AssertionError("retrieval unexpectedly succeeded")

    record(rows, "both retrieval branches unavailable", "controlled 503", both_down)

    def reranker_down():
        result = pipeline({"reranker": "fail"}, use_reranker=True).retrieve(
            "resilience question"
        )
        return result.service_modes[0].value, "original ranking retained"

    record(rows, "reranker unavailable", "no-rerank degraded 200", reranker_down)

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=10, clock=clock)

    def circuit_cycle():
        for _ in range(2):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(OSError("down")))
            except OSError:
                pass
        try:
            breaker.call(lambda: "blocked")
        except CircuitOpenError:
            pass
        clock.now = 11
        if breaker.call(lambda: "recovered") != "recovered":
            raise AssertionError("half-open probe failed")
        return breaker.state, "opened, rejected, half-open probe, closed"

    record(rows, "dependency repeatedly fails then recovers", "breaker recovery", circuit_cycle)
    return rows


def network_rows(client):
    import psycopg
    import redis

    rows = []
    configure(client)

    def postgres_query():
        with psycopg.connect(
            "postgresql://rag:rag@127.0.0.1:15432/ragdb",
            connect_timeout=1,
        ) as connection:
            connection.execute("SET statement_timeout = 500")
            return connection.execute("SELECT 1").fetchone()[0]

    record(rows, "PostgreSQL proxy healthy", "SELECT succeeds", lambda: (
        "normal", f"SELECT returned {postgres_query()}"
    ))

    try:
        client.add_toxic(
            "postgres", "latency-750ms", "latency", {"latency": 750, "jitter": 0}
        )
        def postgres_latency():
            try:
                value = postgres_query()
                return "slow", f"SELECT returned {value}"
            except psycopg.Error as error:
                return "unavailable_retrieval", type(error).__name__

        record(
            rows,
            "PostgreSQL 750ms network latency",
            "bounded success or controlled timeout",
            postgres_latency,
        )
    finally:
        client.clear_toxics("postgres")

    try:
        client.set_enabled("postgres", False)

        def postgres_down():
            try:
                postgres_query()
            except psycopg.Error as error:
                return "unavailable_retrieval", type(error).__name__
            raise AssertionError("disabled PostgreSQL proxy accepted a query")

        record(rows, "PostgreSQL proxy disabled", "fast controlled failure", postgres_down)
    finally:
        client.set_enabled("postgres", True)

    try:
        client.add_toxic(
            "redis", "latency-750ms", "latency", {"latency": 750, "jitter": 0}
        )

        def redis_timeout():
            cache = redis.Redis.from_url(
                "redis://localhost:16379/0",
                decode_responses=True,
                socket_connect_timeout=0.25,
                socket_timeout=0.5,
            )
            try:
                cache.ping()
            except redis.RedisError as error:
                return "degraded_cache", type(error).__name__
            raise AssertionError("Redis latency did not exceed the timeout")

        record(rows, "Redis 750ms network latency", "fail-open cache timeout", redis_timeout)
    finally:
        client.clear_toxics("redis")
        client.set_enabled("redis", True)

    record(rows, "dependencies restored", "healthy after faults", lambda: (
        "normal", f"PostgreSQL SELECT returned {postgres_query()}"
    ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows = in_process_rows()
    if args.network:
        rows.extend(network_rows(ToxiproxyClient()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} resilience scenarios to {args.output}")


if __name__ == "__main__":
    main()
