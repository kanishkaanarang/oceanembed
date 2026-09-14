"""Run from the project root: python -m scripts.benchmark_mission_metrics."""
from time import perf_counter_ns
import json

import numpy as np

from src.models.mission_metrics import compute_mission_metrics, mackenzie_sound_speed


def main():
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps = [29, 29, 28.9, 28.7, 28.5, 28, 27, 25, 23, 21, 17, 12, 8, 6, 4]

    def calculate():
        speeds = mackenzie_sound_speed(temps, 35, depths)
        return compute_mission_metrics(depths, temps, speeds)

    for _ in range(100):
        calculate()
    timings = []
    for _ in range(5000):
        start = perf_counter_ns()
        calculate()
        timings.append((perf_counter_ns() - start) / 1e6)
    report = {
        "samples": len(depths), "iterations": len(timings),
        "scope": "Mackenzie + heat potential + descriptive profile diagnostics; excludes CNN, I/O and UI",
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "max_ms": float(np.max(timings)),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
