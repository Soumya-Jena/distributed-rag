"""Turn Locust CSV output into capacity and cache-amplification tables."""

import argparse
import csv
import re
from pathlib import Path


def value(row, *names):
    for name in names:
        if row.get(name) not in (None, ""):
            return float(row[name])
    return 0.0


def read_aggregate(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(row for row in csv.DictReader(handle) if row.get("Name") == "Aggregated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("experiments/day-15/raw"))
    parser.add_argument("--slo-p95-ms", type=float, default=10_000)
    parser.add_argument("--slo-error-percent", type=float, default=1.0)
    args = parser.parse_args()
    rows = []
    allowed = set()
    for manifest in args.directory.parent.glob("*-sweep-manifest.csv"):
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                allowed.add((item["mode"], int(item["concurrency"])))
    pattern = re.compile(r"c(\d+)-(unique|repeat)_stats\.csv$")
    for path in args.directory.glob("*_stats.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        concurrency, mode = int(match.group(1)), match.group(2)
        if allowed and (mode, concurrency) not in allowed:
            continue
        source = read_aggregate(path)
        requests = int(value(source, "Request Count"))
        failures = int(value(source, "Failure Count"))
        rps = value(source, "Requests/s")
        error_percent = 100 * failures / requests if requests else 100.0
        rows.append({
            "mode": mode, "concurrency": concurrency,
            "goodput_rps": rps * (1 - error_percent / 100),
            "p50_ms": value(source, "50%", "Median Response Time"),
            "p95_ms": value(source, "95%"), "p99_ms": value(source, "99%"),
            "error_percent": error_percent,
            "requests": requests, "failures": failures,
            "meets_slo": int(
                value(source, "95%") <= args.slo_p95_ms
                and error_percent < args.slo_error_percent
            ),
        })
    rows.sort(key=lambda item: (item["mode"], item["concurrency"]))
    if not rows:
        raise SystemExit("No Locust stats CSV files found")
    output = args.directory.parent / "saturation-summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    sustainable = {}
    for mode in {row["mode"] for row in rows}:
        valid = [row for row in rows if row["mode"] == mode and row["meets_slo"]]
        sustainable[mode] = max((row["concurrency"] for row in valid), default=0)
    print(f"Wrote {output}")
    print("Maximum sustainable concurrency:", sustainable)


if __name__ == "__main__":
    main()
