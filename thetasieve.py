import math
import numpy as np

from ntt import bitrev as _bitrev, pow_table as _pow_table, twiddles as _twiddles
from ntt import primes_for_bound, reset_caches


def zeta(s, N=60):
    total = sum(n ** (-float(s)) for n in range(1, N))
    Nf = float(N)
    tail = Nf ** (1 - s) / (s - 1) + 0.5 * Nf ** (-s)
    tail += s * Nf ** (-s - 1) / 12.0
    tail -= s * (s + 1) * (s + 2) * Nf ** (-s - 3) / 720.0
    tail += s * (s + 1) * (s + 2) * (s + 3) * (s + 4) * Nf ** (-s - 5) / 30240.0
    return total + tail


def ball_volume(m):
    return math.pi ** (m / 2.0) / math.gamma(m / 2.0 + 1.0)


def mobius_sieve(n):
    mu = np.ones(n + 1, dtype=np.int64)
    primes = np.ones(n + 1, dtype=bool)
    primes[:2] = False
    for p in range(2, n + 1):
        if primes[p]:
            primes[2 * p::p] = False
            mu[p::p] *= -1
            sq = p * p
            if sq <= n:
                mu[sq::sq] = 0
    return mu


def _ntt(a, p, g, invert, rev):
    n = a.shape[0]
    a = a[rev].copy()
    wfull = _twiddles(p, g, n, invert)
    length = 2
    while length <= n:
        half = length >> 1
        w = wfull[:: n // length]
        A = a.reshape(-1, length)
        u = A[:, :half].copy()
        v = A[:, half:] * w % p
        A[:, :half] = (u + v) % p
        A[:, half:] = (u - v) % p
        length <<= 1
    if invert:
        ninv = pow(n, p - 2, p)
        a = a * ninv % p
    return a


def _mult_trunc(a, b, X, p, g, L, rev):
    fa = np.zeros(L, dtype=np.int64)
    fa[: a.shape[0]] = a
    fa = _ntt(fa, p, g, False, rev)
    if b is a:
        fb = fa
    else:
        fb = np.zeros(L, dtype=np.int64)
        fb[: b.shape[0]] = b
        fb = _ntt(fb, p, g, False, rev)
    fc = fa * fb % p
    del fa, fb
    c = _ntt(fc, p, g, True, rev)
    return c[: X + 1].copy()


def theta_prefix_mod(m, X):
    bound = 2 * (2 * math.isqrt(X) + 2) ** m
    L = 1
    while L < 2 * X + 2:
        L <<= 1
    primes = primes_for_bound(max(L.bit_length() - 1, 1), bound)
    rev = _bitrev(L)
    base = np.zeros(X + 1, dtype=np.int64)
    base[0] = 1
    for j in range(1, math.isqrt(X) + 1):
        base[j * j] = 2
    tables = []
    mods = []
    for p, g in primes:
        e = m
        result = None
        sq = base % p
        while e > 0:
            if e & 1:
                result = sq.copy() if result is None else _mult_trunc(result, sq, X, p, g, L, rev)
            e >>= 1
            if e:
                sq = _mult_trunc(sq, sq, X, p, g, L, rev)
        pref = np.cumsum(result) % p
        tables.append(pref)
        mods.append(p)
    return tables, mods


class PrefixCounter:
    def __init__(self, m, X):
        self.m = m
        self.X = X
        self.tables, self.mods = theta_prefix_mod(m, X)
        self._crt_coeffs = self._garner_setup()
        self._cache = {}

    def _garner_setup(self):
        coeffs = []
        prod = 1
        for p in self.mods:
            coeffs.append((prod, pow(prod % p, p - 2, p)))
            prod *= p
        self.modprod = prod
        return coeffs

    def prefix(self, x):
        if x in self._cache:
            return self._cache[x]
        val = int(self.tables[0][x])
        for i in range(1, len(self.mods)):
            p = self.mods[i]
            prod, inv = self._crt_coeffs[i]
            k = ((int(self.tables[i][x]) - val) * inv) % p
            val += k * prod
        self._cache[x] = val
        return val

    def ball_count(self, R):
        return self.prefix(min(R * R, self.X))


def count_visible(m, R, counter=None, mu=None):
    X = R * R
    if counter is None:
        counter = PrefixCounter(m, X)
    if mu is None:
        mu = mobius_sieve(R)
    total = 0
    for d in range(1, R + 1):
        md = int(mu[d])
        if md == 0:
            continue
        total += md * (counter.prefix(X // (d * d)) - 1)
    return total


def visible_series(m, Rmax):
    counter = PrefixCounter(m, Rmax * Rmax)
    mu = mobius_sieve(Rmax)
    sf = [d for d in range(1, Rmax + 1) if mu[d] != 0]
    out = np.zeros(Rmax + 1, dtype=object)
    for R in range(1, Rmax + 1):
        X = R * R
        total = 0
        for d in sf:
            if d > R:
                break
            total += int(mu[d]) * (counter.prefix(X // (d * d)) - 1)
        out[R] = total
    return out


def error_series(m, Rmax):
    V = visible_series(m, Rmax)
    km = ball_volume(m)
    zm = zeta(m)
    R = np.arange(Rmax + 1, dtype=np.float64)
    main = km * R ** m / zm
    E = np.array([float(V[i]) for i in range(Rmax + 1)]) - main
    return V, E


def error_series_half(m, Rmax):
    counter = PrefixCounter(m, Rmax * Rmax + Rmax)
    mu = mobius_sieve(Rmax)
    sf = [d for d in range(1, Rmax + 1) if mu[d] != 0]
    km = ball_volume(m)
    zm = zeta(m)
    V = np.zeros(Rmax + 1, dtype=object)
    E = np.zeros(Rmax + 1, dtype=np.float64)
    for R in range(1, Rmax + 1):
        X = R * R + R
        total = 0
        for d in sf:
            if d > R:
                break
            total += int(mu[d]) * (counter.prefix(X // (d * d)) - 1)
        V[R] = total
        E[R] = float(total) - km * (R + 0.5) ** m / zm
    return V, E


def _tsum(x):
    total = 0
    i = 1
    while i <= x:
        q = x // i
        j = x // q
        total += q * (j * (j + 1) // 2 - (i - 1) * i // 2)
        i = j + 1
    return total


def prefix4_divisor(x):
    return 1 + 8 * (_tsum(x) - 4 * _tsum(x // 4))


def divisor_count4(R):
    X = R * R
    mu = mobius_sieve(R)
    total = 0
    for d in range(1, R + 1):
        md = int(mu[d])
        if md == 0:
            continue
        x = X // (d * d)
        total += md * (8 * (_tsum(x) - 4 * _tsum(x // 4)))
    return total
