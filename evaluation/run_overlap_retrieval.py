import os
import subprocess
import sys
import time


CONFIGURATIONS = [(200, 0), (200, 20), (200, 40), (200, 80)]


def run(module, *arguments, environment):
    subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        check=True,
        env=environment,
    )


def main():
    for chunk_size, overlap in CONFIGURATIONS:
        label = f"overlap-{chunk_size}-{overlap}"
        environment = os.environ.copy()
        environment["CHUNK_SIZE"] = str(chunk_size)
        environment["CHUNK_OVERLAP"] = str(overlap)

        print(f"\n=== {label} ===", flush=True)
        started = time.perf_counter()
        run("src.ingest", "datasets/raw", environment=environment)
        ingestion_seconds = time.perf_counter() - started
        run(
            "evaluation.corpus_stats",
            "--label",
            label,
            "--ingestion-seconds",
            ingestion_seconds,
            environment=environment,
        )
        run(
            "evaluation.evaluate_retrieval",
            "--label",
            label,
            environment=environment,
        )


if __name__ == "__main__":
    main()
