"""Run the simulation study of Section 5 (Choice and Preference settings).

Example
-------
    python run_experiments.py                      # full study, all cores
    python run_experiments.py --setting choice --costs monotone   # monotone positional costs
    python run_experiments.py --setting choice --ks 3 5 --reps 2 --iters 200   # quick test
"""
import argparse
import json
import os
import time
import zlib

import pandas as pd
from joblib import Parallel, delayed

from omrank import run_single

DEFAULT_KS = {"choice": [3, 5, 10, 20], "preference": [3, 4, 5, 6]}


def job_seed(base, setting, k, n, eps, rep):
    """Reproducible seed that does not depend on the order in which jobs run."""
    return [base, zlib.crc32(setting.encode()), k, n, int(round(eps * 1e6)), rep]


def mc_train_for(setting, k, mc_train):
    # the k = 6 preference design has 720 x 720 entries per profile: cap memory
    return min(mc_train, 200) if setting == "preference" and k >= 6 else mc_train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["choice", "preference", "both"], default="both")
    parser.add_argument("--ks", type=int, nargs="+", help="numbers of alternatives (default depends on setting)")
    parser.add_argument("--ns", type=int, nargs="+", default=[50, 200, 500], help="numbers of voters")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.03, 0.10], help="relative tolerances")
    parser.add_argument("--costs", choices=["free", "monotone"], default="free",
                        help="monotone: non-decreasing positional costs (Choice only)")
    parser.add_argument("--warm-start", action="store_true", help="add the classical rules to the initial swarm")
    parser.add_argument("--reps", type=int, default=10, help="independent replicas per scenario")
    parser.add_argument("--particles", type=int, default=30)
    parser.add_argument("--iters", type=int, default=1000, help="maximum PSO iterations")
    parser.add_argument("--patience", type=int, default=None, help="stop after this many iterations without improvement")
    parser.add_argument("--mc-train", type=int, default=1000, help="profiles used inside the optimisation")
    parser.add_argument("--mc-test", type=int, default=10_000, help="independent profiles used for evaluation")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    settings = ["choice", "preference"] if args.setting == "both" else [args.setting]
    if args.costs == "monotone":
        settings = [s for s in settings if s == "choice"]
    os.makedirs(args.out, exist_ok=True)

    for setting in settings:
        ks = args.ks or DEFAULT_KS[setting]
        jobs = [(k, n, eps, rep) for k in ks for n in args.ns for eps in args.eps for rep in range(args.reps)]
        # largest problems first, so that the parallel pool stays busy until the end
        jobs.sort(key=lambda j: (-j[0], -j[1]))
        print(f"[{setting}] {len(jobs)} runs on {args.jobs} workers", flush=True)

        start = time.perf_counter()
        rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(run_single)(setting, k, n, eps, seed=job_seed(args.seed, setting, k, n, eps, rep),
                                n_particles=args.particles, max_iter=args.iters, patience=args.patience,
                                mc_train=mc_train_for(setting, k, args.mc_train), mc_test=args.mc_test,
                                costs=args.costs, warm_start=args.warm_start)
            for k, n, eps, rep in jobs)

        df = pd.DataFrame(rows)
        df["rep"] = [rep for *_, rep in jobs]
        df["best"] = df["best"].map(json.dumps)
        df["history"] = df["history"].map(json.dumps)
        df = df.sort_values(["k", "n", "eps", "rep"])
        suffix = ("" if args.costs == "free" else f"_{args.costs}") + ("_warm" if args.warm_start else "")
        path = os.path.join(args.out, f"{setting}{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"[{setting}] done in {time.perf_counter() - start:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
