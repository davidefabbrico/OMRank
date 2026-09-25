"""Wall-clock comparison between the legacy implementation (legacy/simulation_server.py)
and the vectorised one, on identical workloads (same swarm size, same number of
Monte Carlo profiles per evaluation, single BLAS thread).

    VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_legacy.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy"))

import simulation_server as legacy  # noqa: E402
from omrank.objectives import TieObjective, dimension  # noqa: E402
from omrank.pso import SimplexPSO  # noqa: E402

PARTICLES, MC, ITERS = 30, 100, 10


def per_iteration_legacy(setting, k, n, eps, processor):
    func = legacy.dynamic_choice if setting == "choice" else legacy.sparse_preference
    objective = lambda v: func(v, processor=processor, n=n, MCiterations=MC, eps=eps)  # noqa: E731
    np.random.seed(0)
    pso = legacy.SimplexPSO(PARTICLES, dimension(setting, k), objective, {}, ITERS, fixed_first=0.0)
    start = time.perf_counter()
    pso.optimize()
    return (time.perf_counter() - start) / ITERS


def per_iteration_new(setting, k, n, eps):
    objective = TieObjective(setting, k, n, eps, MC, np.random.default_rng(0))
    start = time.perf_counter()
    SimplexPSO(objective, dimension(setting, k) - 1, n_particles=PARTICLES, max_iter=ITERS, rng=0).optimize()
    return (time.perf_counter() - start) / ITERS


def main():
    print(f"{'setting':<11}{'k':>3}{'n':>5} | {'setup old':>10}{'setup new':>10} | "
          f"{'iter old':>10}{'iter new':>10}{'speed-up':>10}")
    rows = []
    for setting, k, n in [("choice", 5, 200), ("choice", 20, 500),
                          ("preference", 4, 200), ("preference", 5, 200), ("preference", 6, 200)]:
        start = time.perf_counter()
        if setting == "preference":
            processor = legacy.EfficientPermutationProcessor(k)
        else:
            # legacy run_choice built the full k! x k! table although dynamic_choice only
            # reads processor.k; that is infeasible for k >= 8, so it is skipped here
            processor = type("Processor", (), {"k": k})()
        setup_old = time.perf_counter() - start
        start = time.perf_counter()
        dimension(setting, k)            # builds (and caches) the relative-position table
        setup_new = time.perf_counter() - start

        old = per_iteration_legacy(setting, k, n, 0.03, processor)
        new = per_iteration_new(setting, k, n, 0.03)
        print(f"{setting:<11}{k:>3}{n:>5} | {setup_old:>9.2f}s{setup_new:>9.3f}s | "
              f"{old:>9.3f}s{new:>9.4f}s{old / new:>9.0f}x", flush=True)
        rows.append({"setting": setting, "k": k, "n": n, "setup_old": setup_old, "setup_new": setup_new,
                     "iter_old": old, "iter_new": new})
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "benchmark.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
