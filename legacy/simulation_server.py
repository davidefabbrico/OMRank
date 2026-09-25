import numpy as np
import pandas as pd
import time
import math
import itertools
from functools import lru_cache
from sympy.combinatorics import Permutation
from joblib import Parallel, delayed


# ─────────────────────────────────────────────
# CLASSES AND FUNCTIONS
# ─────────────────────────────────────────────

class EfficientPermutationProcessor:
    def __init__(self, k):
        self.k = k
        self.rank_dict = {}
        self.perm_list = []

        for idx, perm in enumerate(itertools.permutations(range(k))):
            self.perm_list.append(perm)
            self.rank_dict[perm] = idx

        self._build_mult_table()

    def _build_mult_table(self):
        fact = self.fact
        self.mult_table = np.zeros((fact, fact), dtype=np.int32)
        for i in range(fact):
            for j in range(fact):
                self.mult_table[i, j] = self.perm_product(i, j)

    @property
    def fact(self):
        return math.factorial(self.k)

    @lru_cache(maxsize=None)
    def perm_product(self, i, j):
        p1 = Permutation(list(self.perm_list[i]))
        p2 = Permutation(list(self.perm_list[j]))
        product = p1 * p2
        return self.rank_dict[tuple(product.array_form)]


class SimplexPSO:
    def __init__(self, n_particles, dim, objective_func, objective_params,
                 max_iter=100, fixed_first=None):
        self.n_particles = n_particles
        self.dim = dim
        self.objective_func = objective_func
        self.objective_params = objective_params
        self.max_iter = max_iter
        self.fixed_first = fixed_first

        self.omega = 0.9
        self.c1 = 2.0
        self.c2 = 2.0

        if fixed_first is not None:
            free_dim = dim - 1
            remaining = 1.0 - fixed_first
            self.positions = np.random.dirichlet(np.ones(free_dim),
                                                  size=n_particles) * remaining
        else:
            self.positions = np.random.dirichlet(np.ones(dim), size=n_particles)

        self.velocities = np.zeros_like(self.positions)
        self.initialize_best_positions()

    def _build_full_vector(self, x):
        if self.fixed_first is not None:
            return np.concatenate([[self.fixed_first], x])
        return x

    def initialize_best_positions(self):
        self.pbest = self.positions.copy()
        self.pbest_scores = np.array([
            self.objective_func(self._build_full_vector(p), **self.objective_params)
            for p in self.positions
        ])
        best_idx = np.argmin(self.pbest_scores)
        self.gbest = self.pbest[best_idx]
        self.gbest_score = self.pbest_scores[best_idx]

    def project_simplex(self, x, target_sum=1.0):
        x_scaled = x / target_sum
        x_sorted = np.sort(x_scaled)[::-1]
        cumsum = np.cumsum(x_sorted) - 1.0
        indices = np.arange(1, len(x_scaled) + 1)
        rho = np.where(x_sorted - cumsum / indices > 0)[0][-1]
        lambda_ = cumsum[rho] / (rho + 1)
        return np.maximum(x_scaled - lambda_, 0) * target_sum

    def update_particle(self, idx):
        r1, r2 = np.random.rand(2)
        self.velocities[idx] = (self.omega * self.velocities[idx] +
                                self.c1 * r1 * (self.pbest[idx] - self.positions[idx]) +
                                self.c2 * r2 * (self.gbest - self.positions[idx]))

        target_sum = 1.0 - self.fixed_first if self.fixed_first is not None else 1.0
        new_pos = self.project_simplex(self.positions[idx] + self.velocities[idx],
                                        target_sum)
        self.positions[idx] = new_pos
        score = self.objective_func(self._build_full_vector(new_pos),
                                     **self.objective_params)

        if score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = score
            if score < self.gbest_score:
                self.gbest = new_pos
                self.gbest_score = score

    def optimize(self):
        convergence = []
        for _ in range(self.max_iter):  # tqdm rimosso per il server
            for i in range(self.n_particles):
                self.update_particle(i)
            self.update_inertia()
            convergence.append(self.gbest_score)
        return self._build_full_vector(self.gbest), self.gbest_score, convergence

    def update_inertia(self):
        self.omega *= 0.99


def sparse_preference(vector, processor, n=500, MCiterations=100, eps=0.05):
    fact = processor.fact
    vector = np.asarray(vector)
    mult_table = processor.mult_table
    counts = []

    for _ in range(MCiterations):
        samples = np.random.choice(fact, n)
        all_sums = vector[mult_table[:, samples]].sum(axis=1)
        current_min = all_sums.min()
        eps_value = eps * abs(current_min)
        counts.append(np.sum(
            (all_sums >= current_min - eps_value) &
            (all_sums <= current_min + eps_value)
        ))

    return np.mean(counts)


def dynamic_choice(vector, processor, n=500, MCiterations=100, eps=0.05):
    k = processor.k
    vector = np.asarray(vector)
    counts = []

    for _ in range(MCiterations):
        perms = np.argsort(np.random.rand(n, k), axis=1)
        sums = vector[perms].sum(axis=0)
        min_sum = sums.min()
        eps_value = eps * abs(min_sum)
        counts.append(np.sum(
            (sums >= min_sum - eps_value) &
            (sums <= min_sum + eps_value)
        ))

    return np.mean(counts)


# ─────────────────────────────────────────────
# SINGLE JOB FUNCTIONS
# ─────────────────────────────────────────────

def run_choice(k, n, eps, repeat, n_particles, max_iter):
    """Esegue una singola replica del setting Choice."""
    processor = EfficientPermutationProcessor(k)
    objective = lambda v: dynamic_choice(v, processor=processor, n=n, eps=eps)

    start = time.time()
    pso = SimplexPSO(n_particles, k, objective, {}, max_iter, fixed_first=0.0)
    best, score, _ = pso.optimize()
    elapsed = time.time() - start

    print(f"[Choice]     k={k}, n={n}, eps={eps}, repeat={repeat} | score={score:.4f}")

    return {
        'setting': 'choice',
        'k': k,
        'n': n,
        'eps': eps,
        'repeat': repeat,
        'best': best.tolist(),
        'score': score,
        'time': elapsed,
    }


def run_preference(k, n, eps, repeat, n_particles, max_iter):
    """Esegue una singola replica del setting Preference."""
    processor = EfficientPermutationProcessor(k)
    objective = lambda v: sparse_preference(v, processor=processor, n=n, eps=eps)
    dim_pref = math.factorial(k)

    start = time.time()
    pso = SimplexPSO(n_particles, dim_pref, objective, {}, max_iter, fixed_first=0.0)
    best, score, _ = pso.optimize()
    elapsed = time.time() - start

    print(f"[Preference] k={k}, n={n}, eps={eps}, repeat={repeat} | score={score:.4f}")

    return {
        'setting': 'preference',
        'k': k,
        'n': n,
        'eps': eps,
        'repeat': repeat,
        'best': best.tolist(),
        'score': score,
        'time': elapsed,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Parametri simulazione
    ks_choice = [3, 5, 8, 10, 15, 20]   # Choice scala linearmente
    ks_pref   = [3, 4, 5, 6]            # Preference scala factorialmente

    ns = [10, 100, 1000]
    epsilons = [0.1, 0.3, 0.5]
    n_particles = 50
    max_iter = 50
    n_repeats = 5
    n_jobs = -1    # -1 = tutti i core, -2 = tutti tranne uno

    output_path = "results_simulations_pso.csv"

    # ── Job list Choice ──
    jobs_choice = [
        (k, n, eps, repeat)
        for k in ks_choice
        for n, eps in itertools.product(ns, epsilons)
        for repeat in range(1, n_repeats + 1)
    ]

    # ── Job list Preference ──
    jobs_pref = [
        (k, n, eps, repeat)
        for k in ks_pref
        for n, eps in itertools.product(ns, epsilons)
        for repeat in range(1, n_repeats + 1)
    ]

    print(f"Job Choice:     {len(jobs_choice)}")
    print(f"Job Preference: {len(jobs_pref)}")
    print(f"Totale job:     {len(jobs_choice) + len(jobs_pref)} | n_jobs={n_jobs}\n")

    # ── Parallelizza Choice ──
    results_choice = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_choice)(k, n, eps, repeat, n_particles, max_iter)
        for k, n, eps, repeat in jobs_choice
    )

    # ── Parallelizza Preference ──
    results_pref = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(run_preference)(k, n, eps, repeat, n_particles, max_iter)
        for k, n, eps, repeat in jobs_pref
    )

    # ── Salvataggio in due CSV separati ──
    df_choice = pd.DataFrame(results_choice)
    df_pref   = pd.DataFrame(results_pref)

    df_choice.to_csv("results_choice.csv", index=False)
    df_pref.to_csv("results_preference.csv", index=False)

    print("\nSimulazione completata.")
    print("  → results_choice.csv")
    print("  → results_preference.csv")
