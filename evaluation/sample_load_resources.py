"""Sample host and Docker component resources during a load run."""

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def docker_stats():
    command = [
        "docker", "stats", "--no-stream", "--format",
        '{"name":"{{.Name}}","cpu":"{{.CPUPerc}}","memory":"{{.MemUsage}}"}',
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("experiments/day-15/resource-samples.csv"))
    parser.add_argument("--pid", type=int, help="Optional Uvicorn process ID")
    args = parser.parse_args()
    process = psutil.Process(args.pid) if args.pid else None
    rows = []
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        timestamp = datetime.now(timezone.utc).isoformat()
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        try:
            app_cpu = process.cpu_percent() if process else ""
            app_rss = round(process.memory_info().rss / 1024 / 1024, 2) if process else ""
        except psutil.NoSuchProcess:
            app_cpu, app_rss = "exited", "exited"
        containers = docker_stats() or [{"name": "host", "cpu": "", "memory": ""}]
        for container in containers:
            rows.append({
                "timestamp": timestamp, "host_cpu_percent": cpu,
                "host_memory_percent": memory, "component": container["name"],
                "component_cpu": container["cpu"], "component_memory": container["memory"],
                "app_cpu_percent": app_cpu, "app_rss_mb": app_rss,
            })
        time.sleep(args.interval)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
