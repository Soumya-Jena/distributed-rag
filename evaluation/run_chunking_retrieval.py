import os
import subprocess
import sys

from time import perf_counter


CONFIGURATIONS = [
    (100, 20),
    (150, 30),
    (200, 40),
    (240, 40),
]


def run(command, environment):
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, env=environment, check=True)


def main():
    for chunk_size, overlap in CONFIGURATIONS:
        label = f"chunk-{chunk_size}-{overlap}"
        environment = os.environ.copy()
        environment["CHUNK_SIZE"] = str(chunk_size)
        environment["CHUNK_OVERLAP"] = str(overlap)

        started = perf_counter()
        run(
            [sys.executable, "-m", "src.ingest", "datasets/raw"],
            environment,
        )
        ingestion_seconds = perf_counter() - started

        run(
            [
                sys.executable,
                "-m",
                "evaluation.corpus_stats",
                "--label",
                label,
                "--ingestion-seconds",
                str(ingestion_seconds),
            ],
            environment,
        )
        run(
            [
                sys.executable,
                "-m",
                "evaluation.evaluate_retrieval",
                "--label",
                label,
            ],
            environment,
        )


if __name__ == "__main__":
    main()
