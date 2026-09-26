"""Run guarded Locust concurrency levels and stop after saturation signals."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def aggregate_row(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return next(row for row in rows if row.get("Name") == "Aggregated")


def number(row, *names):
    for name in names:
        if row.get(name) not in (None, ""):
            return float(row[name])
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--mode", choices=("unique", "repeat"), default="unique")
    parser.add_argument("--levels", type=int, nargs="+", default=(1, 2, 5, 10, 15, 25, 50))
    parser.add_argument("--duration", default="5m")
    parser.add_argument("--spawn-rate", type=float, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/day-15/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob(f"c*-{args.mode}_*.csv"):
        stale.unlink()
    baseline_p95 = None
    manifest = []
    for users in args.levels:
        prefix = args.output_dir / f"c{users}-{args.mode}"
        environment = os.environ.copy()
        environment["LOAD_MODE"] = args.mode
        command = [
            sys.executable, "-m", "locust", "-f", "load_tests/locustfile.py",
            "--headless", "--host", args.host, "-u", str(users),
            "-r", str(min(args.spawn_rate, users)), "-t", args.duration,
            "--csv", str(prefix), "--only-summary",
        ]
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run(command, env=environment, check=False)
        stats_path = Path(f"{prefix}_stats.csv")
        if not stats_path.exists():
            raise RuntimeError(f"Locust did not create {stats_path}")
        row = aggregate_row(stats_path)
        requests = int(number(row, "Request Count"))
        failures = int(number(row, "Failure Count"))
        p95 = number(row, "95%", "95%ile")
        rps = number(row, "Requests/s")
        error_percent = 100 * failures / requests if requests else 100.0
        baseline_p95 = p95 if baseline_p95 is None else baseline_p95
        reason = ""
        if completed.returncode != 0 or error_percent > 5:
            reason = "errors"
        elif users > args.levels[0] and p95 > 3 * baseline_p95:
            reason = "p95_over_3x_baseline"
        manifest.append({
            "mode": args.mode, "concurrency": users, "requests": requests,
            "failures": failures, "error_percent": error_percent,
            "rps": rps, "p95_ms": p95, "stop_reason": reason,
        })
        if reason:
            print(f"Safety stop after C={users}: {reason}")
            break
    manifest_path = args.output_dir.parent / f"{args.mode}-sweep-manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
