import math
import itertools
import numpy as np

from thetasieve import mobius_sieve


def _isqrt_vec(a):
    t = np.floor(np.sqrt(a.astype(np.float64))).astype(np.int64)
    t[(t + 1) ** 2 <= a] += 1
    t[t ** 2 > a] -= 1
    return t


def naive_count(m, R):
    R2 = R * R
    rng = np.arange(-R, R + 1, dtype=np.int64)
    k = 0
    while (2 * R + 1) ** (m - k) > (1 << 22):
        k += 1
    mesh = np.meshgrid(*([rng] * (m - k)), indexing="ij")
    norm_mesh = sum(g * g for g in mesh)
    gcd_mesh = np.abs(mesh[0])
    for g in mesh[1:]:
        gcd_mesh = np.gcd(gcd_mesh, np.abs(g))
    if k == 0:
        return int(np.count_nonzero((norm_mesh <= R2) & (gcd_mesh == 1)))
    total = 0
    for prefix in itertools.product(*([rng] * k)):
        s = sum(int(v) * int(v) for v in prefix)
        if s > R2:
            continue
        g0 = 0
        for v in prefix:
            g0 = math.gcd(g0, abs(int(v)))
        g = np.gcd(gcd_mesh, g0)
        total += int(np.count_nonzero((norm_mesh <= R2 - s) & (g == 1)))
    return total


def ball_count_enum(m, R2):
    if R2 < 0:
        return 0
    t = math.isqrt(R2)
    if m == 1:
        return 2 * t + 1
    if m == 2:
        ys = np.arange(-t, t + 1, dtype=np.int64)
        rem = R2 - ys * ys
        tz = _isqrt_vec(rem)
        return int(np.sum(2 * tz + 1))
    rng = np.arange(-t, t + 1, dtype=np.int64)
    prefix_axes = [rng] * (m - 2)
    total = 0
    for prefix in itertools.product(*prefix_axes):
        s = sum(v * v for v in prefix)
        if s > R2:
            continue
        rem = R2 - s
        u = math.isqrt(rem)
        ys = np.arange(-u, u + 1, dtype=np.int64)
        rem2 = rem - ys * ys
        tz = _isqrt_vec(rem2)
        total += int(np.sum(2 * tz + 1))
    return total


def legendre_count(m, R):
    mu = mobius_sieve(R)
    R2 = R * R
    total = 0
    for d in range(1, R + 1):
        md = int(mu[d])
        if md == 0:
            continue
        total += md * (ball_count_enum(m, R2 // (d * d)) - 1)
    return total


def sparse_count(m, R):
    X = R * R
    if (2 * R + 2) ** m >= 2 ** 62:
        raise OverflowError("int64 capacity exceeded")
    sq = [j * j for j in range(1, math.isqrt(X) + 1)]
    cur = np.zeros(X + 1, dtype=np.int64)
    cur[0] = 1
    for s in sq:
        cur[s] = 2
    for _ in range(m - 1):
        new = cur.copy()
        for s in sq:
            new[s:] += 2 * cur[: X + 1 - s]
        cur = new
    pref = np.cumsum(cur)
    mu = mobius_sieve(R)
    total = 0
    for d in range(1, R + 1):
        md = int(mu[d])
        if md == 0:
            continue
        total += md * (int(pref[X // (d * d)]) - 1)
    return total
