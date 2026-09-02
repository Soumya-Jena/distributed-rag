import subprocess
import sys

from time import perf_counter


def run(*arguments):
    print(f"\n> {sys.executable} {' '.join(arguments)}", flush=True)
    subprocess.run([sys.executable, *arguments], check=True)


def main():
    started = perf_counter()
    run("-m", "src.ingest", "datasets/raw")
    ingestion_seconds = perf_counter() - started
    run(
        "-m",
        "evaluation.corpus_stats",
        "--label",
        "final",
        "--ingestion-seconds",
        str(ingestion_seconds),
    )
    run("-m", "evaluation.evaluate_retrieval", "--label", "final")
    run("-m", "evaluation.evaluate_rag", "--label", "final")


if __name__ == "__main__":
    main()
