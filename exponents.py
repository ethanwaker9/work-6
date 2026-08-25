import math
from math import log, log2, exp, pi, e, sqrt

TAIL = 1e-18
RHO_C = 0.5 * log2(4.0 / 3.0)


def chi_m4(d):
    r = d % 4
    if r == 1:
        return 1
    if r == 3:
        return -1
    return 0


def chi_m3(d):
    r = d % 3
    if r == 1:
        return 1
    if r == 2:
        return -1
    return 0


def divisor_character_sum(k, chi):
    total = 0
    i = 1
    while i * i <= k:
        if k % i == 0:
            total += chi(i)
            j = k // i
            if j != i:
                total += chi(j)
        i += 1
    return total


def rep_Z2(k):
    if k == 0:
        return 1
    return 4 * divisor_character_sum(k, chi_m4)


def rep_A2(k):
    if k == 0:
        return 1
    return 6 * divisor_character_sum(k, chi_m3)


def theta_Z(u, jmax=200):
    total = 1.0
    for j in range(1, jmax + 1):
        t = 2.0 * exp(-u * j * j)
        total += t
        if t < TAIL * total:
            break
    return total


def theta_Z2_direct(u, kmax=4000):
    total = 1.0
    for k in range(1, kmax + 1):
        r = rep_Z2(k)
        if r:
            t = r * exp(-u * k)
            total += t
            if t < TAIL * total and k > 8:
                break
    return total


def theta_A2(u, kmax=4000):
    total = 1.0
    for k in range(1, kmax + 1):
        r = rep_A2(k)
        if r:
            t = r * exp(-u * k)
            total += t
            if t < TAIL * total and k > 8:
                break
    return total


def _golden_min(f, lo, hi, tol=1e-14, iters=400):
    gr = (sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if b - a < tol * (abs(a) + abs(b) + 1.0):
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return f(x), x


def phi(d):
    f = lambda s: log(theta_Z(s)) + d * s
    val, s = _golden_min(f, 1e-8, 2000.0 / max(d, 1e-6) if d < 1 else 200.0)
    return val / log(2.0), s


def gamma_gh(d):
    return 0.5 * log2(2.0 * pi * e * d)


def Lambda(d):
    f = lambda s: log(theta_A2(s)) + 1.5 * d * s
    val, s = _golden_min(f, 1e-8, 2000.0 / max(d, 1e-6) if d < 1 else 200.0)
    return val / log(2.0), s


def rho(d):
    return 2.0 * phi(d)[0] - Lambda(d)[0]


def gh_defect(d):
    return phi(d)[0] - gamma_gh(d)


def sieve_gain(d):
    return RHO_C - rho(d)


def defect_asymptotic(d):
    return 2.0 * exp(-2.0 * pi * pi * d) / log(2.0)


def phi_dual(d):
    f = lambda u: 0.5 * log(u / pi) + log(theta_Z(u)) + pi * pi * d / u
    val, u = _golden_min(f, 1e-8, max(200.0, 40.0 * pi * pi * d))
    return val / log(2.0), u


def rho_dual(d):
    g = lambda u: log(u) + 2.0 * pi * pi * d / u
    a, _ = _golden_min(lambda u: g(u) + log(theta_Z(u) ** 2), 1e-8,
                       max(200.0, 40.0 * pi * pi * d))
    b, _ = _golden_min(lambda u: g(u) + log(theta_A2(u)), 1e-8,
                       max(200.0, 40.0 * pi * pi * d))
    return RHO_C + (a - b) / log(2.0)


def gh_density(alpha):
    return alpha * alpha


def saturation_density():
    return 4.0 / (3.0 * 2.0 * pi * e)


def zn_saturation_density(target=None):
    if target is None:
        target = 0.5 * log2(4.0 / 3.0)
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid)[0] < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def log2_generic_pair_probability(n):
    from math import lgamma
    total = 0.0
    term_log = 0.0
    m = n - 1
    s = 0.0
    k = 0
    while True:
        lt = (lgamma(0.5 * (m + 1)) - lgamma(0.5 * m) - lgamma(0.5)
              + lgamma(k + 0.5) - lgamma(k + 1)
              + (0.5 * m + k) * log(0.75) - log(0.5 * m + k))
        t = exp(lt)
        s += t
        k += 1
        if t < 1e-18 * s or k > 200000:
            break
    del total, term_log
    return log2(s)


if __name__ == "__main__":
    print(f"{'d':>9} {'phi':>10} {'gamma':>10} {'defect':>10} {'asym':>10} "
          f"{'Lambda':>10} {'rho':>10} {'rho_dual':>10} {'rho_c-rho':>10}")
    for d in [0.005, 0.01, 0.02, 0.04, saturation_density(), 0.12, 0.2, 0.4, 0.8, 1.6]:
        p = phi(d)[0]
        L = Lambda(d)[0]
        r = 2 * p - L
        print(f"{d:9.5f} {p:10.5f} {gamma_gh(d):10.5f} {p-gamma_gh(d):10.5f} "
              f"{defect_asymptotic(d):10.5f} {L:10.5f} {r:10.5f} {rho_dual(d):10.5f} "
              f"{RHO_C-r:10.6f}")
    print()
    print("rho_c =", RHO_C)
    print("generic saturation density =", saturation_density())
    print("Z^n saturation density     =", zn_saturation_density())
