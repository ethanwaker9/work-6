# Lattice sieving by precise shell counting

This repository counts, exactly, the lattice points and the primitive lattice points of the integer lattice `Z^m` inside a Euclidean ball of radius `R`, far beyond the range that enumeration reaches. It computes, exactly, the joint distribution of the two norms and the inner product of a pair of lattice vectors of `Z^n`, and therefore the exact probability that two sieve database vectors reduce each other. These probabilities are exponentially small, so no sampling method can reach them. It runs the *shell sieve*, a two-sieve whose database size and radius schedule come from those exact probabilities instead of from the Gaussian heuristic, and it also evaluates the Gauss sieve experiment on `Z^n` presented by [Bennett et el.](https://link.springer.com/chapter/10.1007/978-3-031-30589-4_9) for comparison.


## Files and Content

The experiments have been implemented in Python 3.10 via NumPy, SciPy (for the spherical cap probability of a random lattice), and Matplotlib.

`ntt.py` is for number theoretic transform for prime generation, batched one dimensional transform, multi dimensional transform, Chinese remainder reconstruction. `thetasieve.py` for the theta sieve as Möbius dilation sieve over prefix sums of the `m`-fold theta convolution with an exact transform engine, the divisor sum shortcut for `m = 4`, and the discrepancy series. `baselines.py` is for counting baselines as direct enumeration with primitivity tests, the Legendre–Möbius enumeration, and the shift add sparse convolution engine. `exponents.py` if for the counting exponent `phi`, the volume exponent `gamma`, the pair exponent `Lambda`, the reduction exponent `rho`, the generic constant `rho_star`, the two dimensional representation formulas for `Z^2` and `A_2`, and the dual form of `rho` obtained by Poisson summation. `pairsieve.py` is for the pair theta sieve as exact pair profiles of `Z^n` by truncated three variable theta convolution powers, together with its baselines, namely the coordinate recursion, uniform shell and ball samplers, a Monte Carlo estimator, and the spherical cap probability of a random lattice. `shellsieve.py` is for the shell sieve with its exact schedule, the Gaussian heuristic schedule it is compared against, and a Gauss sieve on `Z^n` driven by discrete Gaussian samples. `validate.py` is for all correctness checks, including agreement of every engine with an independent method and with brute force enumeration. `experiments.py` produces tables and figures input as exponents, exact pair probabilities, the crossover study, Monte Carlo validation, engine benchmarks, database schedules, shell sieve traces, and the Gauss sieve reproduction. `benchmark.py` is for wall time and peak memory of the five counting methods. `errorterms.py` is for exact values of `V_m(R)` and of the discrepancy `E_m(R)` for all `R` up to 1500 and `m` in {3, 4, 5}, plus the half-integer series for `m = 5`. 

```
python3 validate.py
python3 experiments.py
python3 benchmark.py
python3 errorterms.py
python3 appendixdata.py
python3 makeplots.py
```

`validate.py` takes about a minute, `experiments.py` should take less than forty minutes, and the three remaining data scripts about one hour in total.

