"""One replica of the MCPSO experiment: optimise on training profiles, then
evaluate the optimum and the classical rules on independent test profiles."""
import time

import numpy as np

from .objectives import TieObjective, dimension, estimate
from .pso import TRANSFORMS, SimplexPSO
from .rules import RULES


def run_single(setting, k, n, eps, seed, n_particles=30, max_iter=1000, patience=None,
               mc_train=100, mc_test=10_000, omega=0.9, c1=2.0, c2=2.0, costs="free",
               warm_start=False):
    """``costs="monotone"`` (Choice only) restricts to non-decreasing positional costs;
    ``warm_start`` adds the classical rules to the initial swarm."""
    rng_train, rng_pso, rng_test = [np.random.default_rng(s)
                                    for s in np.random.SeedSequence(seed).spawn(3)]
    dim = dimension(setting, k)

    rules = RULES[setting](k)
    init = None
    if warm_start:
        w = np.vstack(list(rules.values()))
        # simplex coordinates of the classical costs (inverse of the transform)
        init = w[:, 1:] if costs == "free" else np.diff(w, axis=1) * (k - np.arange(1, k))

    start = time.perf_counter()
    objective = TieObjective(setting, k, n, eps, mc_train, rng_train)
    pso = SimplexPSO(objective, dim - 1, n_particles=n_particles, max_iter=max_iter,
                     omega=omega, c1=c1, c2=c2, transform=TRANSFORMS[costs],
                     patience=patience, init_points=init, rng=rng_pso)
    result = pso.optimize()
    elapsed = time.perf_counter() - start

    names = ["MCPSO"] + list(rules)
    weights = np.vstack([result.best] + list(rules.values()))
    mean, se = estimate(setting, k, n, eps, weights, mc_test, rng_test)

    row = {
        "setting": setting, "costs": costs, "warm_start": warm_start, "k": k, "n": n, "eps": eps, "seed": seed,
        "train_score": result.score, "n_iter": result.n_iter, "time": elapsed,
        "best": result.best.tolist(), "history": result.history,
    }
    for name, m, s in zip(names, mean, se):
        row[f"test_{name}"] = float(m)
        row[f"se_{name}"] = float(s)
    return row
