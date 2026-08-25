import csv
import math
import os
import time

import numpy as np

from thetasieve import PrefixCounter, mobius_sieve, zeta, ball_volume, prefix4_divisor, error_series


def explicit_constants(mmax=8):
    log2 = math.log(2.0)
    C = {4: 142.0}
    for m in range(4, mmax):
        km = ball_volume(m)
        C[m + 1] = 3.0 * C[m] + (km * m * (m - 1) / 4.0 + 16.0 * 5.0 ** m) / log2
    B = {}
    for m in range(5, mmax + 1):
        km = ball_volume(m)
        B[m] = C[m] * zeta(m - 2) + (2.0 * 5.0 ** m + 2.0 * km + 1.0) / log2
    return C, B


def sphere_ratio(m, Rmax, counter=None):
    if m == 4:
        best = (0.0, 0)
        for R in range(2, Rmax + 1):
            P = float(prefix4_divisor(R * R)) - ball_volume(4) * R ** 4
            ratio = abs(P) / (R ** 2 * math.log(R))
            if ratio > best[0]:
                best = (ratio, R)
        return best
    best = (0.0, 0)
    km = ball_volume(m)
    for R in range(2, Rmax + 1):
        P = float(counter.prefix(R * R)) - km * R ** m
        ratio = abs(P) / (R ** (m - 2) * math.log(R))
        if ratio > best[0]:
            best = (ratio, R)
    return best


def prime_shell_excess(counter, qs):
    rows = []
    for q in qs:
        r5 = counter.prefix(q * q) - counter.prefix(q * q - 1)
        prim = r5 - 10
        rows.append((q, r5, prim, prim / q ** 3))
    return rows


def kfree_series(m, k, Rmax, counter):
    mu = mobius_sieve(int(round(Rmax ** (1.0 / k))) + 2)
    km = ball_volume(m)
    zkm = zeta(k * m)
    E = np.zeros(Rmax + 1)
    V = np.zeros(Rmax + 1, dtype=object)
    for R in range(1, Rmax + 1):
        X = R * R
        total = 0
        d = 1
        while d ** k <= R:
            if mu[d] != 0:
                total += int(mu[d]) * (counter.prefix(X // d ** (2 * k)) - 1)
            d += 1
        V[R] = total
        E[R] = float(total) - km * float(R) ** m / zkm
    return V, E


def stats(y):
    q = np.percentile(y, [25, 50, 75])
    return dict(mean=float(np.mean(y)), std=float(np.std(y)), minimum=float(np.min(y)),
                q25=float(q[0]), median=float(q[1]), q75=float(q[2]), maximum=float(np.max(y)))


def main():
    os.makedirs("results", exist_ok=True)
    out = {}
    C, B = explicit_constants()
    out["C"] = C
    out["B"] = B

    t0 = time.perf_counter()
    c5 = PrefixCounter(5, 1500 * 1500)
    build5 = time.perf_counter() - t0

    t0 = time.perf_counter()
    V5, E5 = error_series(5, 1500)
    series5 = time.perf_counter() - t0
    out["build5_seconds"] = build5
    out["series5_seconds"] = series5

    out["P4_ratio"] = sphere_ratio(4, 2000)
    out["P5_ratio"] = sphere_ratio(5, 1500, c5)
    c6 = PrefixCounter(6, 600 * 600)
    out["P6_ratio"] = sphere_ratio(6, 600, c6)

    out["shells"] = prime_shell_excess(c5, [101, 499, 997, 1499])

    Vk, Ek = kfree_series(5, 2, 1500, c5)
    with open("results/error_kfree_m5.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R", "V", "E"])
        for R in range(1, 1501):
            w.writerow([R, Vk[R], float(Ek[R])])
    R = np.arange(1501)
    mask = R >= 20
    out["kfree_stats"] = stats(Ek[mask] / R[mask].astype(float) ** 3)

    y5 = np.array([float(E5[i]) for i in range(1501)])
    out["E5_stats"] = stats(y5[mask] / R[mask].astype(float) ** 3)

    with open("results/appendix_summary.txt", "w") as f:
        for key, val in out.items():
            f.write(f"{key}: {val}\n")
    for key, val in out.items():
        print(key, val)


if __name__ == "__main__":
    main()
