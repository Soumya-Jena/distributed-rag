import os
import subprocess
import sys


# Selected after the retrieval-only comparison. The 200/40 baseline was
# excluded because it had the lowest average top-1 similarity in the tied
# perfect-MRR group.
CONFIGURATIONS = [
    (100, 20),
    (150, 30),
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

        run(
            [sys.executable, "-m", "src.ingest", "datasets/raw"],
            environment,
        )
        run(
            [
                sys.executable,
                "-m",
                "evaluation.evaluate_rag",
                "--label",
                label,
            ],
            environment,
        )


if __name__ == "__main__":
    main()
