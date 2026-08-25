import csv
import os

from thetasieve import error_series, error_series_half

RMAX = {3: 1500, 4: 1500, 5: 1500}


def main():
    os.makedirs("results", exist_ok=True)
    for m, Rmax in RMAX.items():
        V, E = error_series(m, Rmax)
        with open(f"results/error_m{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["R", "V", "E"])
            for R in range(1, Rmax + 1):
                w.writerow([R, V[R], float(E[R])])
        print(f"m={m} done up to R={Rmax}", flush=True)
    V, E = error_series_half(5, RMAX[5])
    with open("results/error_half_m5.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R", "V", "E"])
        for R in range(1, RMAX[5] + 1):
            w.writerow([R, V[R], float(E[R])])
    print("m=5 half-integer radii done", flush=True)


if __name__ == "__main__":
    main()
