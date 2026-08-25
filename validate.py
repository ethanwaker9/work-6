import math

import numpy as np

from thetasieve import (PrefixCounter, count_visible, mobius_sieve, zeta,
                        ball_volume, prefix4_divisor, divisor_count4)
from baselines import naive_count, legendre_count, sparse_count, ball_count_enum
import exponents as ex
from pairsieve import (pair_probability_shell, pair_probability_ball,
                       pair_profile_dp, mask_ball_reduce, mask_ball_pairs,
                       mask_shell_reduce, mask_shell_pairs, pair_theta_sieve,
                       pair_probability_mc, shell_counts_exact, shape_weights,
                       sample_shell, generic_probability)
from shellsieve import sieve_round, schedule, generic_schedule


def sigma(n):
    return sum(d for d in range(1, n + 1) if n % d == 0)


def check_hand_values():
    assert count_visible(2, 1) == 4
    assert count_visible(2, 2) == 8
    assert count_visible(3, 1) == 6
    c = PrefixCounter(3, 4)
    assert c.prefix(4) == 33
    assert c.prefix(1) == 7
    print("hand values ok")


def check_ball_counts():
    for m in (2, 3, 4):
        for R2 in (1, 4, 10, 30):
            t = math.isqrt(R2)
            grid = np.arange(-t, t + 1, dtype=np.int64)
            mesh = np.meshgrid(*([grid] * m), indexing="ij")
            norm = sum(g * g for g in mesh)
            direct = int(np.count_nonzero(norm <= R2))
            assert ball_count_enum(m, R2) == direct, (m, R2)
    print("ball counts ok")


def check_jacobi():
    X = 2000
    c = PrefixCounter(4, X)
    r4 = [c.prefix(n) - (c.prefix(n - 1) if n else 0) for n in range(X + 1)]
    for n in range(1, 300, 2):
        assert r4[n] == 8 * sigma(n), n
    print("jacobi identity ok")


def check_cross_methods():
    cases = [(2, 5), (2, 12), (3, 5), (3, 11), (3, 20), (4, 5), (4, 9),
             (4, 14), (5, 4), (5, 8), (5, 12)]
    for m, R in cases:
        a = naive_count(m, R)
        b = legendre_count(m, R)
        d = sparse_count(m, R)
        e = count_visible(m, R)
        assert a == b == d == e, (m, R, a, b, d, e)
        if m == 4:
            assert divisor_count4(R) == a, (m, R)
    print("cross-method agreement ok")


def check_divisor_prefix():
    X = 2000
    c = PrefixCounter(4, X)
    for x in list(range(0, 50)) + [123, 777, 1234, 2000]:
        assert prefix4_divisor(x) == c.prefix(x), x
    assert divisor_count4(120) == count_visible(4, 120)
    print("divisor-sum shortcut ok")


def check_density():
    for m in (2, 3, 4, 5):
        R = 60
        v = count_visible(m, R)
        pred = ball_volume(m) * R ** m / zeta(m)
        assert abs(v / pred - 1.0) < 0.05, (m, v, pred)
    print("density sanity ok")


def check_two_dimensional_theta():
    for k in range(1, 400):
        direct2 = 0
        directA = 0
        t = int(math.ceil(2.0 * math.sqrt(k / 3.0))) + 1
        for a in range(-t, t + 1):
            for b in range(-t, t + 1):
                if a * a + b * b == k:
                    direct2 += 1
                if a * a - a * b + b * b == k:
                    directA += 1
        assert ex.rep_Z2(k) == direct2, (k, ex.rep_Z2(k), direct2)
        assert ex.rep_A2(k) == directA, (k, ex.rep_A2(k), directA)
    print("two-dimensional representation formulas ok")


def check_hexagonal_symmetry():
    for u in (0.3, 1.0, 2.5, 7.0):
        s = 0.0
        t = 0.0
        for a in range(-60, 61):
            for b in range(-60, 61):
                q = a * a - a * b + b * b
                w = math.exp(-u * q)
                s += a * a * w
                t += a * b * w
        assert abs(2 * t - s) < 1e-9 * max(1.0, abs(s)), (u, s, t)
    print("hexagonal correlation identity ok")


def check_theta_monotone():
    for u in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        assert ex.theta_A2(u) > ex.theta_Z(u) ** 2, u
    print("theta_A2 dominates theta_Z2 ok")


def check_dual_identity():
    for d in [0.01, 0.05, ex.saturation_density(), 0.2, 0.5, 1.0]:
        a = ex.rho(d)
        b = ex.rho_dual(d)
        assert abs(a - b) < 1e-6, (d, a, b)
        assert a < ex.RHO_C, (d, a)
        assert ex.phi(d)[0] > ex.gamma_gh(d), d
    for d in [1.6, 2.0, 3.0]:
        assert ex.rho(d) <= ex.RHO_C + 1e-12, d
        assert ex.phi(d)[0] >= ex.gamma_gh(d) - 1e-12, d
    print("duality and inequalities ok")


def check_pair_engine():
    for (n, K) in [(20, 2), (30, 3), (40, 3), (25, 4)]:
        W = pair_profile_dp(n, K)
        for target in (K, K - 1):
            g, tot = pair_theta_sieve(n, K, [mask_ball_reduce(K, target),
                                             mask_ball_pairs(K)])
            assert g == int(W[mask_ball_reduce(K, target)].sum()), (n, K, target)
            assert tot == int(W[mask_ball_pairs(K)].sum()), (n, K, target)
        g, tot = pair_theta_sieve(n, K, [mask_shell_reduce(K, K, K),
                                         mask_shell_pairs(K, K)])
        assert g == int(W[mask_shell_reduce(K, K, K)].sum())
        assert tot == int(W[mask_shell_pairs(K, K)].sum())
    print("pair theta sieve matches coordinate DP ok")


def check_shell_counts():
    for n in (20, 40, 60):
        counts, _ = shell_counts_exact(n, 8)
        for k in range(1, 9):
            _, ws = shape_weights(n, k)
            assert counts[k] == sum(ws), (n, k, counts[k], sum(ws))
    print("shell counts match shape enumeration ok")


def check_brute_force_pairs():
    n, K = 6, 3
    pts = []
    t = math.isqrt(K)
    import itertools
    for x in itertools.product(range(-t, t + 1), repeat=n):
        s = sum(v * v for v in x)
        if 1 <= s <= K:
            pts.append(x)
    arr = np.array(pts, dtype=np.int64)
    norms = (arr * arr).sum(axis=1)
    G = arr @ arr.T
    for target in (K, K - 1):
        d2 = norms[:, None] + norms[None, :] - 2 * G
        d2m = np.minimum(d2, norms[:, None] + norms[None, :] + 2 * G)
        good = int(((d2m > 0) & (d2m <= target)).sum())
        g, tot = pair_theta_sieve(n, K, [mask_ball_reduce(K, target),
                                         mask_ball_pairs(K)])
        assert tot == arr.shape[0] ** 2, (tot, arr.shape[0] ** 2)
        assert g == good, (target, g, good)
    print("brute-force pair enumeration ok")


def check_monte_carlo():
    rng = np.random.default_rng(4242)
    for (n, K, target) in [(30, 3, 3), (40, 3, 2), (50, 4, 4)]:
        _, _, p = pair_probability_ball(n, K, target)
        pm = pair_probability_mc(n, K, target, 300000, rng)
        assert abs(pm - p) / p < 0.08, (n, K, target, p, pm)
    print("monte carlo agrees with exact counts ok")


def check_sieve_round():
    rng = np.random.default_rng(99)
    db = sample_shell(30, 3, 400, rng)
    found = sieve_round(db, 3)
    big = db.astype(np.int64)
    norms = (big * big).sum(axis=1)
    ref = set()
    for i in range(db.shape[0]):
        for j in range(i + 1, db.shape[0]):
            for sgn in (1, -1):
                w = big[i] - sgn * big[j]
                s = int(w @ w)
                if 0 < s <= 3:
                    nz = np.flatnonzero(w)
                    ww = w if w[nz[0]] > 0 else -w
                    ref.add(ww.astype(np.int8).tobytes())
    assert set(found.keys()) == ref, (len(found), len(ref))
    del norms
    print("sieve round matches direct pair search ok")


def check_generic_probability():
    rng = np.random.default_rng(5)
    for n in (20, 30, 40):
        v = rng.standard_normal((200000, n))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        w = rng.standard_normal((200000, n))
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        emp = float((np.abs((v * w).sum(axis=1)) >= 0.5).mean())
        p = generic_probability(n, 1.0)
        assert abs(emp - p) <= 4.0 * math.sqrt(p * (1 - p) / 200000) + 1e-6, (n, emp, p)
    print("generic pair probability ok")


def check_schedules():
    lv = schedule(40, 4, 2.0)
    gv = generic_schedule(40, 4, 2.0)
    for K in range(2, 5):
        assert lv[K][1] < gv[K][1], (K, lv[K], gv[K])
    print("shell schedule below generic schedule ok")


if __name__ == "__main__":
    check_hand_values()
    check_ball_counts()
    check_jacobi()
    check_cross_methods()
    check_divisor_prefix()
    check_density()
    check_two_dimensional_theta()
    check_hexagonal_symmetry()
    check_theta_monotone()
    check_dual_identity()
    check_pair_engine()
    check_shell_counts()
    check_brute_force_pairs()
    check_monte_carlo()
    check_sieve_round()
    check_generic_probability()
    check_schedules()
    print("all checks passed")
