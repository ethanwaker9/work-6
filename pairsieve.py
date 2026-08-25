import math
import numpy as np

from ntt import ntt_nd, primes_for_bound, crt_combine


def pad_shape(K):
    L12 = 1
    while L12 < 2 * K + 2:
        L12 <<= 1
    L3 = 1
    while L3 < 4 * K + 2:
        L3 <<= 1
    return (L12, L12, L3)


def single_coordinate_array(K, L):
    a = np.zeros(L, dtype=np.int64)
    t = math.isqrt(K)
    for x in range(-t, t + 1):
        for y in range(-t, t + 1):
            a[x * x, y * y, x * y + K] += 1
    return a


def truncate(a, K, L):
    out = np.zeros(L, dtype=np.int64)
    out[: K + 1, : K + 1, : 2 * K + 1] = a[: K + 1, : K + 1, K : 3 * K + 1]
    return out


def mul_trunc(a, b, p, g, K, L):
    fa = ntt_nd(a % p, p, g, False)
    if b is a:
        fa *= fa
    else:
        fb = ntt_nd(b % p, p, g, False)
        fa *= fb
        del fb
    fa %= p
    fa = ntt_nd(fa, p, g, True)
    out = truncate(fa, K, L)
    del fa
    return out


def ball_count(n, K):
    from thetasieve import PrefixCounter
    return PrefixCounter(n, K).prefix(K)


def pair_theta_sieve(n, K, masks):
    L = pad_shape(K)
    bound = 2 * ball_count(n, K) ** 2
    order_bits = max(max(L).bit_length() - 1, 1)
    primes = primes_for_bound(order_bits, bound)
    base = single_coordinate_array(K, L)
    residues = [[] for _ in masks]
    mods = []
    for p, g in primes:
        cur = None
        sq = base.copy()
        e = n
        while e > 0:
            if e & 1:
                cur = sq.copy() if cur is None else mul_trunc(cur, sq, p, g, K, L)
            e >>= 1
            if e:
                sq = mul_trunc(sq, sq, p, g, K, L)
        core = cur[: K + 1, : K + 1, : 2 * K + 1]
        for i, m in enumerate(masks):
            residues[i].append(int(core[m].sum() % p))
        mods.append(p)
    return [crt_combine([[r] for r in res], mods)[0] for res in residues]


def _grid(K):
    k1 = np.arange(K + 1)[:, None, None]
    k2 = np.arange(K + 1)[None, :, None]
    j = np.arange(-K, K + 1)[None, None, :]
    return k1, k2, j


def mask_ball_pairs(K, kmin=1):
    k1, k2, j = _grid(K)
    m = (k1 >= kmin) & (k2 >= kmin) & (j == j)
    return np.broadcast_to(m, (K + 1, K + 1, 2 * K + 1)).copy()


def mask_ball_reduce(K, target, kmin=1):
    k1, k2, j = _grid(K)
    diff = k1 + k2 - 2 * np.abs(j)
    return (k1 >= kmin) & (k2 >= kmin) & (diff > 0) & (diff <= target)


def mask_shell_pairs(K, k):
    k1, k2, j = _grid(K)
    m = (k1 == k) & (k2 == k) & (j == j)
    return np.broadcast_to(m, (K + 1, K + 1, 2 * K + 1)).copy()


def mask_shell_reduce(K, k, target):
    k1, k2, j = _grid(K)
    diff = k1 + k2 - 2 * np.abs(j)
    return (k1 == k) & (k2 == k) & (diff > 0) & (diff <= target)


def pair_probability_shell(n, k, target=None):
    if target is None:
        target = k
    good, total = pair_theta_sieve(n, k, [mask_shell_reduce(k, k, target),
                                          mask_shell_pairs(k, k)])
    return good, total, good / total


def pair_probability_ball(n, K, target):
    good, total = pair_theta_sieve(n, K, [mask_ball_reduce(K, target),
                                          mask_ball_pairs(K)])
    return good, total, good / total


def moves(K):
    from collections import Counter
    c = Counter()
    t = math.isqrt(K)
    for a in range(-t, t + 1):
        for b in range(-t, t + 1):
            c[(a * a, b * b, a * b)] += 1
    return sorted(c.items())


def pair_profile_dp(n, K):
    mv = moves(K)
    W = np.zeros((K + 1, K + 1, 2 * K + 1), dtype=object)
    W[0, 0, K] = 1
    for _ in range(n):
        NW = np.zeros_like(W)
        for (da, db, dj), m in mv:
            src = W[: K + 1 - da, : K + 1 - db, :]
            if dj >= 0:
                NW[da:, db:, dj:] += src[:, :, : 2 * K + 1 - dj] * m
            else:
                NW[da:, db:, :dj] += src[:, :, -dj:] * m
        W = NW
    return W


def pair_profile_float(n, K):
    mv = moves(K)
    W = np.zeros((K + 1, K + 1, 2 * K + 1))
    W[0, 0, K] = 1.0
    logscale = 0.0
    for _ in range(n):
        NW = np.zeros_like(W)
        for (da, db, dj), m in mv:
            src = W[: K + 1 - da, : K + 1 - db, :]
            if dj >= 0:
                NW[da:, db:, dj:] += src[:, :, : 2 * K + 1 - dj] * m
            else:
                NW[da:, db:, :dj] += src[:, :, -dj:] * m
        mx = NW.max()
        NW /= mx
        logscale += math.log2(mx)
        W = NW
    return W, logscale


def shell_counts_exact(n, K):
    from thetasieve import PrefixCounter
    c = PrefixCounter(n, K)
    pref = [c.prefix(x) for x in range(K + 1)]
    return [pref[0]] + [pref[i] - pref[i - 1] for i in range(1, K + 1)], pref


def square_shapes(k):
    t = math.isqrt(k)
    shapes = []

    def rec(a, rem, cur):
        if rem == 0:
            shapes.append(tuple(cur))
            return
        if a < 1:
            return
        sq = a * a
        c = 0
        while c * sq <= rem:
            rec(a - 1, rem - c * sq, cur + [(a, c)] if c else cur)
            c += 1

    rec(t, k, [])
    return [tuple(s for s in sh if s[1] > 0) for sh in shapes]


def shape_weights(n, k):
    shapes = square_shapes(k)
    ws = []
    for sh in shapes:
        used = sum(c for _, c in sh)
        if used > n:
            ws.append(0)
            continue
        w = 1
        rem = n
        for a, c in sh:
            w *= math.comb(rem, c) * (2 ** c)
            rem -= c
        ws.append(w)
    return shapes, ws


def sample_shell(n, k, count, rng):
    shapes, ws = shape_weights(n, k)
    tot = sum(ws)
    probs = np.array([w / tot for w in ws], dtype=np.float64)
    idx = rng.choice(len(shapes), size=count, p=probs)
    out = np.zeros((count, n), dtype=np.int8)
    for i in range(count):
        sh = shapes[idx[i]]
        used = sum(c for _, c in sh)
        pos = rng.choice(n, size=used, replace=False)
        o = 0
        for a, c in sh:
            signs = rng.integers(0, 2, size=c) * 2 - 1
            out[i, pos[o : o + c]] = (a * signs).astype(np.int8)
            o += c
    return out


def sample_ball(n, K, count, rng, kmin=1):
    counts, _ = shell_counts_exact(n, K)
    ws = [counts[t] if t >= kmin else 0 for t in range(K + 1)]
    tot = sum(ws)
    probs = np.array([w / tot for w in ws], dtype=np.float64)
    ks = rng.choice(np.arange(K + 1), size=count, p=probs)
    out = np.zeros((count, n), dtype=np.int8)
    for k in np.unique(ks):
        sel = np.flatnonzero(ks == k)
        out[sel] = sample_shell(n, int(k), len(sel), rng)
    return out


def pair_probability_mc(n, K, target, trials, rng, shell=False, k=None):
    hits = 0
    done = 0
    block = 20000
    while done < trials:
        b = min(block, trials - done)
        if shell:
            u = sample_shell(n, k, b, rng).astype(np.int32)
            v = sample_shell(n, k, b, rng).astype(np.int32)
        else:
            u = sample_ball(n, K, b, rng).astype(np.int32)
            v = sample_ball(n, K, b, rng).astype(np.int32)
        nu = (u * u).sum(axis=1)
        nv = (v * v).sum(axis=1)
        ip = np.abs((u * v).sum(axis=1))
        diff = nu + nv - 2 * ip
        hits += int(((diff > 0) & (diff <= target)).sum())
        done += b
    return hits / trials


def generic_probability(n, gamma2):
    from scipy.stats import beta
    c = 1.0 - gamma2 / 2.0
    if c >= 1.0:
        return 0.0
    if c <= 0.0:
        return 1.0
    return float(beta.sf(c * c, 0.5, (n - 1) / 2.0))


if __name__ == "__main__":
    rng = np.random.default_rng(20260731)
    for (n, K) in [(20, 2), (30, 3), (40, 3), (50, 4)]:
        for target in (K, K - 1):
            g, tot, p = pair_probability_ball(n, K, target)
            W = pair_profile_dp(n, K)
            m = mask_ball_reduce(K, target)
            m2 = mask_ball_pairs(K)
            g2 = int(W[m].sum())
            t2 = int(W[m2].sum())
            assert (g, tot) == (g2, t2), (n, K, target, g, g2, tot, t2)
            pm = pair_probability_mc(n, K, target, 200000, rng)
            print(f"n={n:3d} K={K} target={target} exact={p:.5e} mc={pm:.5e} "
                  f"generic={generic_probability(n, target / K):.5e}")
