import csv
import math
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import exponents as ex

OUT = os.path.join("..", "final_paper", "figures")
RES = "results"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 6.4,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "lines.markersize": 2.8,
    "figure.autolayout": True,
})

WIDE = (4.85, 1.95)
HALF = (2.35, 1.85)
FULL = (4.85, 1.7)


def top_legend(ax, ncol=2):
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol,
              frameon=False, borderaxespad=0.0, handlelength=1.6,
              columnspacing=1.0, handletextpad=0.4)

MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*"]

LABELS = {
    "naive": "enumeration",
    "legendre": "Legendre",
    "sparse": "shift-add",
    "theta": "NTT",
    "divisor": "divisor",
    "coordinate DP": "recursion",
    "Monte Carlo (10 percent)": "Monte Carlo",
    "pair theta sieve": "pair theta sieve",
    "shell enumeration": "enumeration",
}


def label(name):
    return LABELS.get(name, name)


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".eps"), format="eps")
    plt.close(fig)
    print("wrote", name + ".eps", flush=True)


def plot_defect():
    d = np.geomspace(0.004, 4.0, 220)
    phi = np.array([ex.phi(x)[0] for x in d])
    gam = np.array([ex.gamma_gh(x) for x in d])
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    ax[0].plot(d, phi, "-", label=r"$\phi(d)$, exact count")
    ax[0].plot(d, gam, "--", label=r"$\gamma(d)$, ball volume")
    ax[0].axvline(ex.saturation_density(), color="0.6", lw=0.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$d$ (squared radius / $n$)")
    ax[0].set_ylabel(r"bits per dimension")
    top_legend(ax[0])
    ax[1].plot(d, phi - gam, "-", label=r"$\phi(d)-\gamma(d)$")
    ax[1].plot(d, [ex.defect_asymptotic(x) for x in d], "--",
               label=r"$2e^{-2\pi^2d}/\ln 2$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-9, 1e1)
    ax[1].axvline(ex.saturation_density(), color="0.6", lw=0.8, ls=":")
    ax[1].set_xlabel(r"$d$ (squared radius / $n$)")
    ax[1].set_ylabel(r"heuristic defect")
    top_legend(ax[1])
    save(fig, "defect")


def plot_rho():
    d = np.geomspace(0.004, 4.0, 220)
    rho = np.array([ex.rho(x) for x in d])
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    ax[0].plot(d, rho, "-", label=r"$\rho(d)$, integer lattice")
    ax[0].axhline(ex.RHO_C, color="k", ls="--", lw=1.0,
                  label=r"$\rho_*=\frac{1}{2}\log_2\frac{4}{3}$")
    ax[0].axvline(ex.saturation_density(), color="0.6", lw=0.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$d$ (squared radius / $n$)")
    ax[0].set_ylabel(r"database exponent")
    top_legend(ax[0])
    ax[1].plot(d, ex.RHO_C - rho, "-", label=r"$\rho_*-\rho(d)$")
    ax[1].plot(d, [ex.defect_asymptotic(x) for x in d], "--",
               label=r"$2e^{-2\pi^2d}/\ln 2$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-9, 1e1)
    ax[1].axvline(ex.saturation_density(), color="0.6", lw=0.8, ls=":")
    ax[1].set_xlabel(r"$d$ (squared radius / $n$)")
    ax[1].set_ylabel(r"sieve gain exponent")
    top_legend(ax[1])
    save(fig, "rho")


def plot_crossover():
    rows = load("crossover.csv")
    if rows is None:
        return
    n = np.array([float(r["n"]) for r in rows])
    k = np.array([float(r["k"]) for r in rows])
    ez = np.array([float(r["expo_zn"]) for r in rows])
    eg = np.array([float(r["expo_gen"]) for r in rows])
    ratio = np.array([float(r["ratio"]) for r in rows])
    even = (k % 2 == 0)
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    ax[0].plot(n[even], ez[even], "o-", label=r"$\mathbb{Z}^n$, $k$ even")
    ax[0].plot(n[~even], ez[~even], "s--", label=r"$\mathbb{Z}^n$, $k$ odd")
    ax[0].plot(n, eg, "^-", color="0.35", label="random lattice")
    ax[0].axhline(ex.RHO_C, color="k", ls="--", lw=1.0, label=r"$\rho_*$")
    ax[0].axhline(ex.rho(ex.saturation_density()), color="0.5", ls=":", lw=1.0,
                  label=r"$\rho(d_*)$")
    ax[0].set_xlabel(r"dimension $n$")
    ax[0].set_ylabel(r"$-\frac{1}{n}\log_2 p_n$")
    top_legend(ax[0], ncol=3)
    ax[1].semilogy(n, ratio, "o-", label=r"$p^{\mathbb{Z}}_n/p^{\mathrm{gen}}_n$")
    ax[1].semilogy(n, 2.0 ** ((ex.RHO_C - ex.rho(ex.saturation_density())) * n),
                   "--", color="0.35", label=r"$2^{(\rho_*-\rho(d_*))n}$")
    ax[1].set_xlabel(r"dimension $n$")
    ax[1].set_ylabel("probability ratio")
    top_legend(ax[1])
    save(fig, "crossover")


def plot_engine():
    rows = load("enginebench.csv")
    if rows is None:
        return
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    for i, m in enumerate(methods):
        sel = [r for r in rows if r["method"] == m]
        n = [float(r["n"]) for r in sel]
        t = [float(r["seconds"]) for r in sel]
        ax[0].loglog(n, t, MARKERS[i % len(MARKERS)] + "-", label=label(m))
    ax[0].set_xlabel(r"dimension $n$, with $K=\lfloor d_* n\rfloor$")
    ax[0].set_ylabel("time (s)")
    top_legend(ax[0])
    rows2 = load("enginefixedK.csv")
    if rows2 is not None:
        for i, m in enumerate(["pair theta sieve", "coordinate recursion"]):
            sel = [r for r in rows2 if r["method"] == m]
            n = [float(r["n"]) for r in sel]
            t = [float(r["seconds"]) for r in sel]
            ax[1].loglog(n, t, MARKERS[i] + "-", label=("sieve" if i == 0 else "recursion") + " time")
        for i, m in enumerate(["pair theta sieve", "coordinate recursion"]):
            sel = [r for r in rows2 if r["method"] == m]
            n = [float(r["n"]) for r in sel]
            mem = [float(r["peak_bytes"]) / 1e6 for r in sel]
            ax[1].loglog(n, mem, MARKERS[i] + "--", color="0.45",
                         label=("sieve" if i == 0 else "recursion") + " memory")
        ax[1].set_xlabel(r"dimension $n$, fixed $K=12$")
        ax[1].set_ylabel("seconds and megabytes")
        top_legend(ax[1])
    save(fig, "engine")


def plot_schedules():
    rows = load("schedules.csv")
    if rows is None:
        return
    n = np.array([float(r["n"]) for r in rows])
    a = np.array([float(r["log2_peak_shell"]) for r in rows])
    b = np.array([float(r["log2_peak_generic"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    ax[0].plot(n, a, "o-", label="shell sieve")
    ax[0].plot(n, b, "s-", color="0.35", label="Gaussian-heuristic sizing")
    ax[0].plot(n, ex.rho(ex.saturation_density()) * n, "--", color="0.6",
               label=r"$\rho(d_*)\,n$")
    ax[0].plot(n, ex.RHO_C * n, ":", color="0.6", label=r"$\rho_* n$")
    ax[0].set_xlabel(r"dimension $n$")
    ax[0].set_ylabel(r"$\log_2$ peak database size")
    top_legend(ax[0])
    ax[1].plot(n, b - a, "o-")
    ax[1].set_xlabel(r"dimension $n$")
    ax[1].set_ylabel(r"bits saved")
    save(fig, "schedules")


def plot_gauss():
    rows = load("gausssieve.csv")
    if rows is None:
        return
    n = np.array([float(r["n"]) for r in rows])
    samples = np.array([float(r["samples"]) for r in rows])
    mx = np.array([float(r["max_list"]) for r in rows])
    band = np.array([float(r["max_band"]) for r in rows])
    gh = np.array([float(r["gh"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=WIDE)
    ax[0].semilogy(n, mx, "s", ms=4.2, mfc="none", label="peak list")
    ax[0].semilogy(n, samples, "o", label="sampled")
    ax[0].semilogy(n, band, "^", label=r"inner band")
    ax[0].semilogy(n, gh, "--", color="0.35", label=r"$\frac{1}{2}(4/3)^{n/2}$")
    ax[0].set_xlabel(r"dimension $n$")
    ax[0].set_ylabel("vectors")
    top_legend(ax[0])
    c = np.polyfit(n, np.log2(samples), 1)
    ax[1].semilogy(n, samples, "o", label="sampled vectors")
    ax[1].semilogy(n, 2.0 ** (c[0] * n + c[1]), "-",
                   label=r"fit $%.2f\cdot%.4f^{n}$" % (2.0 ** c[1], 2.0 ** c[0]))
    ax[1].semilogy(n, 6.4 * 1.15 ** n, "--", color="0.35",
                   label=r"earlier fit $6.4\cdot1.15^{n}$")
    ax[1].set_xlabel(r"dimension $n$")
    ax[1].set_ylabel("vectors")
    top_legend(ax[1])
    save(fig, "gauss")
    return c


def _series(rows, key, xkey="R"):
    return (np.array([float(r[xkey]) for r in rows]),
            np.array([float(r[key]) for r in rows]))


def plot_math_timing():
    rows = load("timings.csv")
    if rows is None:
        return
    for m in (3, 4, 5):
        fig, ax = plt.subplots(figsize=HALF)
        methods = []
        for r in rows:
            if int(r["m"]) == m and r["method"] not in methods:
                methods.append(r["method"])
        for i, meth in enumerate(methods):
            sel = [r for r in rows if int(r["m"]) == m and r["method"] == meth]
            R = [float(r["R"]) for r in sel]
            t = [float(r["seconds"]) for r in sel]
            ax.loglog(R, t, MARKERS[i % len(MARKERS)] + "-", label=label(meth))
        ax.set_xlabel(r"$R$")
        ax.set_ylabel("time (s)")
        top_legend(ax)
        save(fig, f"time_m{m}")
    rows = load("memory.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=HALF)
    methods = []
    for r in rows:
        if int(r["m"]) == 4 and r["method"] not in methods:
            methods.append(r["method"])
    for i, meth in enumerate(methods):
        sel = [r for r in rows if int(r["m"]) == 4 and r["method"] == meth]
        R = [float(r["R"]) for r in sel]
        t = [float(r["peak_bytes"]) / 2 ** 20 for r in sel]
        ax.loglog(R, t, MARKERS[i % len(MARKERS)] + "-", label=label(meth))
    ax.set_xlabel(r"$R$")
    ax.set_ylabel("peak memory (MiB)")
    top_legend(ax)
    save(fig, "mem_m4")


def plot_math_errors():
    for m, power, lab in ((5, 3, r"$E_5(R)/R^3$"), (4, 2, r"$E_4(R)/R^2$"),
                          (3, 21 / 16, r"$E_3(R)/R^{21/16}$")):
        rows = load(f"error_m{m}.csv")
        if rows is None:
            continue
        R, E = _series(rows, "E")
        sel = R >= 10
        fig, ax = plt.subplots(figsize=FULL)
        ax.plot(R[sel], E[sel] / R[sel] ** power, ".", ms=1.4, color="0.15")
        if m == 4:
            x = R[sel]
            env = 16.0 / math.pi ** 2 * np.log(np.log(np.maximum(x, 3.0)))
            ax.plot(x, env, "--", color="0.4", lw=1.0)
            ax.plot(x, -env, "--", color="0.4", lw=1.0)
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(lab)
        save(fig, f"err_m{m}")
    rows = load("error_half_m5.csv")
    if rows is not None:
        R, E = _series(rows, "E")
        sel = R >= 10
        fig, ax = plt.subplots(figsize=FULL)
        ax.plot(R[sel], E[sel] / R[sel] ** 3, ".", ms=1.4, color="0.15")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$E_5(R+\frac{1}{2})/R^3$")
        save(fig, "err_half_m5")
    rows = load("error_kfree_m5.csv")
    if rows is not None:
        R, E = _series(rows, "E")
        sel = R >= 10
        fig, ax = plt.subplots(figsize=FULL)
        ax.plot(R[sel], E[sel] / R[sel] ** 3, ".", ms=1.4, color="0.15")
        ax.set_xlabel(r"$R$")
        ax.set_ylabel(r"$E^{(2)}_5(R)/R^3$")
        save(fig, "err_kfree_m5")
    rows = load("error_m5.csv")
    if rows is not None:
        R, E = _series(rows, "E")
        sel = R >= 20
        fig, ax = plt.subplots(figsize=HALF)
        ax.hist(E[sel] / R[sel] ** 3, bins=45, color="0.55", edgecolor="0.2",
                linewidth=0.4)
        ax.set_xlabel(r"$E_5(R)/R^3$")
        ax.set_ylabel("count")
        save(fig, "hist_m5")


def convert_all():
    for name in sorted(os.listdir(OUT)):
        if not name.endswith(".eps"):
            continue
        src = os.path.join(OUT, name)
        dst = src[:-4] + ".pdf"
        subprocess.run(["epstopdf", "--outfile=" + dst, src], check=True)
        print("converted", name, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    plot_defect()
    plot_rho()
    plot_crossover()
    plot_engine()
    plot_schedules()
    plot_gauss()
    plot_math_timing()
    plot_math_errors()
    convert_all()


if __name__ == "__main__":
    main()
