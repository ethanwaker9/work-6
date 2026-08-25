import math
import numpy as np

from pairsieve import (pair_probability_ball, pair_probability_shell,
                       generic_probability, sample_ball, sample_shell,
                       shell_counts_exact)



def required_database(p, c=2.0):
    if p <= 0.0:
        return None
    return int(math.ceil(c / p))


def sieve_round(db, target, block=256):
    N = db.shape[0]
    big = db.astype(np.int64)
    norms = (big * big).sum(axis=1)
    found = {}
    for i0 in range(0, N, block):
        A = big[i0 : i0 + block]
        na = norms[i0 : i0 + block]
        G = A @ big.T
        for sgn in (1, -1):
            d2 = na[:, None] + norms[None, :] - 2 * sgn * G
            sel = np.argwhere((d2 <= target) & (d2 > 0))
            for a, b in sel:
                if b <= i0 + a:
                    continue
                w = A[a] - sgn * big[b]
                nz = np.flatnonzero(w)
                if w[nz[0]] < 0:
                    w = -w
                found[w.astype(np.int8).tobytes()] = int(d2[a, b])
    return found


def schedule(n, K0, c=2.0, kmin=1, margin=1.4):
    probs = {}
    for K in range(kmin + 1, K0 + 1):
        probs[K] = pair_probability_ball(n, K, K - 1)[2]
    levels = {kmin: (None, 1)}
    for K in range(kmin + 1, K0 + 1):
        p = probs[K]
        prev = levels[K - 1][1]
        need = max(math.ceil(c / p),
                   math.ceil(math.sqrt(2.0 * margin * prev / p)))
        levels[K] = (p, int(need))
    return levels


def generic_schedule(n, K0, c=2.0, kmin=1, margin=1.4):
    levels = {kmin: (None, 1)}
    for K in range(kmin + 1, K0 + 1):
        p = generic_probability(n, (K - 1) / K)
        prev = levels[K - 1][1]
        if p <= 0.0:
            levels[K] = (p, None)
            continue
        need = max(math.ceil(c / p),
                   math.ceil(math.sqrt(2.0 * margin * prev / p)))
        levels[K] = (p, int(need))
    return levels


def shell_sieve(n, K0, rng, c=2.0, kmin=1, cap=6000, levels=None):
    import time
    if levels is None:
        levels = schedule(n, K0, c, kmin)
    trace = []
    K = K0
    db = None
    while K > kmin:
        p, N = levels[K]
        if N is None or N > cap:
            break
        if db is None or db.shape[0] < N:
            db = sample_ball(n, K, N, rng)
        else:
            order = np.argsort((db.astype(np.int64) ** 2).sum(axis=1))
            db = db[order[:N]]
        t0 = time.perf_counter()
        found = sieve_round(db, K - 1)
        dt = time.perf_counter() - t0
        nxt = levels[K - 1][1]
        ok = len(found) >= nxt
        trace.append({"K": K, "target": K - 1, "p": p, "N": N,
                      "produced": len(found), "needed": nxt,
                      "seconds": dt, "db_bytes": int(db.nbytes), "ok": ok})
        if not ok:
            break
        arr = np.frombuffer(b"".join(found.keys()), dtype=np.int8).reshape(-1, n)
        db = arr.copy()
        K = K - 1
    return K, trace


def sample_discrete_gaussian(n, s, count, rng):
    bound = int(math.ceil(6.0 * s))
    xs = np.arange(-bound, bound + 1)
    w = np.exp(-math.pi * (xs.astype(np.float64) ** 2) / (s * s))
    w /= w.sum()
    return rng.choice(xs, size=(count, n), p=w).astype(np.int64)


def gauss_sieve_zn(n, s, rng, max_samples=4000000, band_d=None):
    lst = np.zeros((0, n), dtype=np.int64)
    lst_norm = np.zeros(0, dtype=np.int64)
    stack = []
    samples = 0
    comparisons = 0
    max_list = 0
    band = None if band_d is None else int(math.floor(band_d * n))
    max_band = 0
    while samples < max_samples:
        if stack:
            v = stack.pop()
        else:
            v = sample_discrete_gaussian(n, s, 1, rng)[0]
            samples += 1
        while lst.shape[0]:
            ip = lst @ v
            comparisons += lst.shape[0]
            gain = 2 * np.abs(ip) - lst_norm
            order = np.argsort(-gain)
            moved = False
            for j in order:
                if gain[j] <= 0:
                    break
                d = int(lst[j] @ v)
                if 2 * abs(d) > lst_norm[j]:
                    v = v - np.sign(d) * lst[j]
                    moved = True
            if not moved:
                break
        nv = int(v @ v)
        if nv == 0:
            break
        if lst.shape[0]:
            ip = lst @ v
            comparisons += lst.shape[0]
            red = 2 * np.abs(ip) - nv > 0
            if red.any():
                moved = lst[red] - np.sign(ip[red])[:, None] * v
                for row in moved:
                    stack.append(row.copy())
                lst = lst[~red]
                lst_norm = lst_norm[~red]
        lst = np.vstack([lst, v[None, :]])
        lst_norm = np.append(lst_norm, nv)
        if lst.shape[0] > max_list:
            max_list = lst.shape[0]
        if band is not None:
            b = int((lst_norm <= band).sum())
            if b > max_band:
                max_band = b
    order = np.argsort(lst_norm)
    return {"list_size": int(lst.shape[0]), "max_list": max_list,
            "samples": samples, "comparisons": comparisons,
            "shortest": int(lst_norm.min()) if lst_norm.size else None,
            "norms": lst_norm[order], "max_band": max_band, "band": band}


if __name__ == "__main__":
    rng = np.random.default_rng(20260731)
    for n in (30, 40, 50):
        r = gauss_sieve_zn(n, 100.0, rng, band_d=0.078)
        gh = 0.5 * (4.0 / 3.0) ** (n / 2.0)
        print(f"n={n} samples={r['samples']} max_list={r['max_list']} "
              f"gh={gh:.1f} band={r['band']} max_band={r['max_band']} "
              f"shortest={r['shortest']}")
    K, tr = shell_sieve(40, 4, rng, c=2.0)
    for t in tr:
        print(t)
    print("attained", K)
