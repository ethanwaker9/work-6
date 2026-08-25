import csv
import gc
import os
import time
import tracemalloc

from thetasieve import count_visible, divisor_count4, reset_caches
from baselines import naive_count, legendre_count, sparse_count

GRIDS = {
    3: {
        "naive": [50, 80, 120, 180, 270, 400],
        "legendre": [100, 200, 400, 800, 1600],
        "sparse": [100, 200, 400, 800, 1600],
        "theta": [100, 200, 400, 800, 1600, 2000],
    },
    4: {
        "naive": [20, 30, 45, 65, 80],
        "legendre": [60, 100, 170, 280, 450],
        "sparse": [100, 200, 400, 700, 1000],
        "theta": [100, 200, 400, 800, 1600, 2000],
        "divisor": [1000, 3000, 10000, 30000, 100000, 300000],
    },
    5: {
        "naive": [10, 15, 22, 30],
        "legendre": [30, 50, 80, 110],
        "sparse": [100, 200, 400, 600, 800],
        "theta": [100, 200, 400, 800, 1600, 2000],
    },
}

METHODS = {
    "naive": naive_count,
    "legendre": legendre_count,
    "sparse": sparse_count,
    "theta": count_visible,
    "divisor": divisor_count4,
}

MEM_POINTS = {
    3: {"naive": [120, 270], "legendre": [400, 800], "sparse": [400, 800, 1600], "theta": [400, 800, 1600]},
    4: {
        "naive": [45, 65],
        "legendre": [170, 280],
        "sparse": [400, 700],
        "theta": [400, 800, 1600],
        "divisor": [3000, 30000, 300000],
    },
    5: {"naive": [15, 22], "legendre": [50, 80], "sparse": [400, 600], "theta": [400, 800, 1600]},
}


def run_time():
    rows = []
    for m, grids in GRIDS.items():
        for method, radii in grids.items():
            fn = METHODS[method]
            for R in radii:
                reset_caches()
                gc.collect()
                t0 = time.perf_counter()
                value = fn(m, R) if method != "divisor" else fn(R)
                t1 = time.perf_counter()
                rows.append({"m": m, "method": method, "R": R, "seconds": t1 - t0, "value": value})
                print(m, method, R, round(t1 - t0, 3), value, flush=True)
    with open("results/timings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["m", "method", "R", "seconds", "value"])
        w.writeheader()
        w.writerows(rows)


def run_memory():
    rows = []
    for m, grids in MEM_POINTS.items():
        for method, radii in grids.items():
            fn = METHODS[method]
            for R in radii:
                reset_caches()
                gc.collect()
                tracemalloc.start()
                value = fn(m, R) if method != "divisor" else fn(R)
                peak = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                rows.append({"m": m, "method": method, "R": R, "peak_bytes": peak, "value": value})
                print(m, method, R, peak, flush=True)
    with open("results/memory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["m", "method", "R", "peak_bytes", "value"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run_time()
    run_memory()
    print("benchmarks written to results/")
