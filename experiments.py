import csv
import gc
import math
import os
import time
import tracemalloc

import numpy as np

import exponents as ex
from pairsieve import (pair_probability_shell, pair_probability_ball,
                       pair_profile_dp, pair_profile_float, generic_probability,
                       pair_probability_mc, shell_counts_exact,
                       mask_ball_reduce, mask_ball_pairs)
from shellsieve import (schedule, generic_schedule, shell_sieve, gauss_sieve_zn)

RESULTS = "results"
SEED = 20260731

D_GRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
          ex.saturation_density(), 0.10, 0.15, 0.20, 0.30, 0.40, 0.60,
          0.80, 1.20, 1.60, 2.40, 3.20]

PAIR_GRID = [(30, 2), (40, 3), (50, 4), (64, 5), (80, 6), (100, 8),
             (128, 10), (160, 12), (200, 16), (240, 19), (280, 22)]

CROSSOVER_GRID = [(n, max(1, int(round(ex.saturation_density() * n))))
                  for n in (26, 40, 52, 64, 78, 90, 102, 116, 128, 154, 180,
                            206, 232, 256)]


def write(name, fields, rows):
    with open(os.path.join(RESULTS, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)", flush=True)


def run_exponents():
    rows = []
    for d in D_GRID:
        p = ex.phi(d)[0]
        g = ex.gamma_gh(d)
        L = ex.Lambda(d)[0]
        r = 2 * p - L
        rows.append({"d": d, "phi": p, "gamma": g, "defect": p - g,
                     "defect_asym": ex.defect_asymptotic(d), "Lambda": L,
                     "rho": r, "rho_c": ex.RHO_C, "gap": ex.RHO_C - r,
                     "gap_asym": ex.defect_asymptotic(d),
                     "rho_dual": ex.rho_dual(d)})
    write("exponents.csv",
          ["d", "phi", "gamma", "defect", "defect_asym", "Lambda", "rho",
           "rho_c", "gap", "gap_asym", "rho_dual"], rows)


def run_pair_probabilities():
    rows = []
    for n, k in PAIR_GRID:
        t0 = time.perf_counter()
        good, total, p = pair_probability_shell(n, k)
        dt = time.perf_counter() - t0
        pg = generic_probability(n, 1.0)
        rows.append({"n": n, "k": k, "d": k / n, "p_zn": p, "p_gen": pg,
                     "ratio": p / pg, "expo_zn": -math.log2(p) / n,
                     "expo_gen": -math.log2(pg) / n,
                     "rho_asym": ex.rho(k / n), "rho_c": ex.RHO_C,
                     "N_zn": math.ceil(2.0 / p), "N_gen": math.ceil(2.0 / pg),
                     "pairs": total, "reducing": good, "seconds": dt})
        print(rows[-1], flush=True)
    write("pairprob.csv",
          ["n", "k", "d", "p_zn", "p_gen", "ratio", "expo_zn", "expo_gen",
           "rho_asym", "rho_c", "N_zn", "N_gen", "pairs", "reducing",
           "seconds"], rows)


def run_crossover():
    rows = []
    d0 = ex.saturation_density()
    for n, k in CROSSOVER_GRID:
        _, _, p = pair_probability_shell(n, k)
        pg = generic_probability(n, 1.0)
        rows.append({"n": n, "k": k, "d": k / n, "expo_zn": -math.log2(p) / n,
                     "expo_gen": -math.log2(pg) / n,
                     "rho_asym": ex.rho(k / n), "rho_c": ex.RHO_C,
                     "ratio": p / pg})
        print(rows[-1], flush=True)
    del d0
    write("crossover.csv",
          ["n", "k", "d", "expo_zn", "expo_gen", "rho_asym", "rho_c", "ratio"],
          rows)


def run_mc_validation():
    rng = np.random.default_rng(SEED)
    rows = []
    for n, K, target, trials in [(20, 2, 2, 400000), (30, 3, 3, 400000),
                                 (30, 3, 2, 400000), (40, 3, 3, 400000),
                                 (40, 3, 2, 400000), (50, 4, 4, 400000),
                                 (50, 4, 3, 400000), (64, 5, 5, 400000)]:
        _, _, p = pair_probability_ball(n, K, target)
        t0 = time.perf_counter()
        pm = pair_probability_mc(n, K, target, trials, rng)
        dt = time.perf_counter() - t0
        rows.append({"n": n, "K": K, "target": target, "exact": p,
                     "monte_carlo": pm, "trials": trials,
                     "rel_error": abs(pm - p) / p, "mc_seconds": dt})
        print(rows[-1], flush=True)
    write("mccheck.csv",
          ["n", "K", "target", "exact", "monte_carlo", "trials", "rel_error",
           "mc_seconds"], rows)


def _peak(fn, *a, **kw):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return out, dt, peak


def run_engine_benchmark():
    rng = np.random.default_rng(SEED)
    rows = []
    grid = [(40, 3), (64, 5), (100, 8), (128, 10), (160, 12), (200, 16)]
    for n, K in grid:
        masks = [mask_ball_reduce(K, K - 1), mask_ball_pairs(K)]
        from pairsieve import pair_theta_sieve
        _, dt, peak = _peak(pair_theta_sieve, n, K, masks)
        rows.append({"n": n, "K": K, "method": "pair theta sieve",
                     "seconds": dt, "peak_bytes": peak})
        print(rows[-1], flush=True)
        if n <= 128:
            _, dt, peak = _peak(pair_profile_dp, n, K)
            rows.append({"n": n, "K": K, "method": "coordinate DP",
                         "seconds": dt, "peak_bytes": peak})
            print(rows[-1], flush=True)
        counts, pref = shell_counts_exact(n, K)
        _, _, p = pair_probability_ball(n, K, K - 1)
        t0 = time.perf_counter()
        pair_probability_mc(n, K, K - 1, 20000, rng)
        rate = 20000 / (time.perf_counter() - t0)
        need = 100.0 / p
        rows.append({"n": n, "K": K, "method": "Monte Carlo (10 percent)",
                     "seconds": need / rate, "peak_bytes": 0})
        print(rows[-1], flush=True)
        rows.append({"n": n, "K": K, "method": "shell enumeration",
                     "seconds": float(pref[K] - 1), "peak_bytes": 0})
        print(rows[-1], flush=True)
    write("enginebench.csv", ["n", "K", "method", "seconds", "peak_bytes"], rows)


def run_gauss_sieve():
    rng = np.random.default_rng(SEED)
    rows = []
    d0 = ex.saturation_density()
    for n in range(24, 51, 2):
        for s in (10.0,):
            reps = 2 if n <= 40 else 1
            for _ in range(reps):
                r = gauss_sieve_zn(n, s, rng, band_d=d0)
                rows.append({"n": n, "s": s, "samples": r["samples"],
                             "max_list": r["max_list"],
                             "max_band": r["max_band"], "band": r["band"],
                             "comparisons": r["comparisons"],
                             "shortest": r["shortest"],
                             "gh": 0.5 * (4.0 / 3.0) ** (n / 2.0)})
                print(rows[-1], flush=True)
    write("gausssieve.csv",
          ["n", "s", "samples", "max_list", "max_band", "band", "comparisons",
           "shortest", "gh"], rows)


def run_shell_sieve():
    rng = np.random.default_rng(SEED)
    rows = []
    for n, K0 in [(40, 4), (50, 4), (60, 5), (70, 5)]:
        lv = schedule(n, K0, 2.0)
        gv = generic_schedule(n, K0, 2.0)
        K, tr = shell_sieve(n, K0, rng, c=2.0, levels=lv, cap=9000)
        for t in tr:
            t["n"] = n
            t["N_generic"] = gv[t["K"]][1]
            t["p_generic"] = gv[t["K"]][0]
            t["attained"] = K
            rows.append(t)
            print(t, flush=True)
    write("shelltrace.csv",
          ["n", "K", "target", "p", "N", "produced", "needed", "seconds",
           "db_bytes", "ok", "N_generic", "p_generic", "attained"], rows)


def run_engine_fixedK():
    from pairsieve import pair_theta_sieve
    K = 12
    rows = []
    for n in (64, 128, 256, 512, 1024, 2048, 4096):
        masks = [mask_ball_reduce(K, K - 1), mask_ball_pairs(K)]
        _, dt, peak = _peak(pair_theta_sieve, n, K, masks)
        rows.append({"n": n, "K": K, "method": "pair theta sieve",
                     "seconds": dt, "peak_bytes": peak})
        print(rows[-1], flush=True)
        _, dt, peak = _peak(pair_profile_dp, n, K)
        rows.append({"n": n, "K": K, "method": "coordinate recursion",
                     "seconds": dt, "peak_bytes": peak})
        print(rows[-1], flush=True)
    write("enginefixedK.csv", ["n", "K", "method", "seconds", "peak_bytes"], rows)


def run_schedules():
    rows = []
    for n, K0 in [(40, 4), (50, 4), (60, 5), (70, 5), (80, 6), (100, 8),
                  (128, 10), (160, 12)]:
        lv = schedule(n, K0, 2.0)
        gv = generic_schedule(n, K0, 2.0)
        peak = max(v[1] for v in lv.values() if v[1] is not None)
        gpeak = max((v[1] for v in gv.values() if v[1] is not None), default=None)
        rows.append({"n": n, "K0": K0, "peak_shell": peak,
                     "peak_generic": gpeak,
                     "ratio": (gpeak / peak) if gpeak else None,
                     "log2_peak_shell": math.log2(peak),
                     "log2_peak_generic": math.log2(gpeak) if gpeak else None})
        print(rows[-1], flush=True)
    write("schedules.csv",
          ["n", "K0", "peak_shell", "peak_generic", "ratio",
           "log2_peak_shell", "log2_peak_generic"], rows)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    run_exponents()
    run_pair_probabilities()
    run_crossover()
    run_mc_validation()
    run_engine_benchmark()
    run_engine_fixedK()
    run_schedules()
    run_shell_sieve()
    run_gauss_sieve()


if __name__ == "__main__":
    main()
