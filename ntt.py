import numpy as np

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _SMALL:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primitive_root(p):
    fac = []
    m = p - 1
    d = 2
    while d * d <= m:
        if m % d == 0:
            fac.append(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        fac.append(m)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac):
            return g
    raise ValueError("no primitive root")


_POOL = {}


def ntt_primes(order_bits, count):
    pool = _POOL.setdefault(order_bits, [])
    step = 1 << order_bits
    if not pool:
        _POOL[order_bits + 1000000] = (1 << 31) // step
    c = _POOL.get(order_bits + 1000000, (1 << 31) // step)
    while len(pool) < count:
        cand = c * step + 1
        c -= 1
        if cand < (1 << 29):
            raise ValueError("prime pool exhausted")
        if is_prime(cand):
            pool.append((cand, primitive_root(cand)))
    _POOL[order_bits + 1000000] = c
    return pool[:count]


def primes_for_bound(order_bits, bound):
    chosen = []
    prod = 1
    need = 1
    while True:
        pool = ntt_primes(order_bits, need)
        if len(pool) < need:
            raise ValueError("prime pool exhausted")
        chosen = []
        prod = 1
        for p, g in pool:
            chosen.append((p, g))
            prod *= p
            if prod > bound:
                return chosen
        need *= 2


_BITREV = {}


def bitrev(n):
    r = _BITREV.get(n)
    if r is None:
        bits = n.bit_length() - 1
        idx = np.arange(n, dtype=np.int64)
        r = np.zeros(n, dtype=np.int64)
        for b in range(bits):
            r |= ((idx >> b) & 1) << (bits - 1 - b)
        _BITREV[n] = r
    return r


def pow_table(base, count, p):
    w = np.ones(count, dtype=np.int64)
    idx = np.arange(count, dtype=np.int64)
    factor = base % p
    b = 0
    while (1 << b) < count:
        mask = ((idx >> b) & 1).astype(bool)
        w[mask] = w[mask] * factor % p
        factor = factor * factor % p
        b += 1
    return w


_TW = {}


def twiddles(p, g, L, invert):
    key = (p, L, invert)
    w = _TW.get(key)
    if w is None:
        root = pow(g, (p - 1) // L, p)
        if invert:
            root = pow(root, p - 2, p)
        w = pow_table(root, max(L >> 1, 1), p)
        _TW[key] = w
    return w


def reset_caches():
    _TW.clear()
    _BITREV.clear()


def ntt_last_axis(a, p, g, invert):
    L = a.shape[-1]
    shape = a.shape
    a = np.ascontiguousarray(a).reshape(-1, L)
    a = a[:, bitrev(L)]
    wfull = twiddles(p, g, L, invert)
    length = 2
    while length <= L:
        half = length >> 1
        w = wfull[:: L // length]
        A = a.reshape(a.shape[0], -1, length)
        u = A[:, :, :half].copy()
        v = A[:, :, half:] * w % p
        A[:, :, :half] = (u + v) % p
        A[:, :, half:] = (u - v) % p
        length <<= 1
    if invert:
        a = a * pow(L, p - 2, p) % p
    return a.reshape(shape)


def ntt_nd(a, p, g, invert):
    for axis in range(a.ndim):
        b = np.moveaxis(a, axis, -1)
        del a
        b = ntt_last_axis(b, p, g, invert)
        a = np.moveaxis(b, -1, axis)
        del b
    return a


def crt_combine(residues, mods):
    result = [int(v) for v in residues[0]]
    prod = mods[0]
    for i in range(1, len(mods)):
        p = mods[i]
        inv = pow(prod % p, p - 2, p)
        for j in range(len(result)):
            k = ((int(residues[i][j]) - result[j]) * inv) % p
            result[j] += k * prod
        prod *= p
    return result
