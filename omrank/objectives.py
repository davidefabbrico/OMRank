"""Monte Carlo estimators of the expected number of epsilon-ties.

For a fixed electoral profile p the aggregate scores are *linear* in the weight
vector:

* Choice (Algorithm 2): the score of alternative a is
  s_a = sum_i w[pos_{r_i}(a)] = (N_p w)_a, where N_p[a, j] counts the voters
  that place a in position j;
* Preference (Algorithm 3): the score of ranking r is
  s_r = sum_i v(r_i^{-1} o r) = (A_p v)_r, where A_p[r, t] counts the voters
  r_i with r_i^{-1} o r = t.

Stacking the matrices of M profiles gives a single "design" matrix, so the
objective of the whole swarm is one (sparse) matrix product. The profiles are
drawn once and kept fixed during an optimisation run (common random numbers),
which makes the objective deterministic; an independent set of profiles is
then used to estimate the performance of the optimum without selection bias.
"""
from functools import lru_cache

import numpy as np
from scipy import sparse

from .permutations import random_permutations, relative_position_table

DENSE_THRESHOLD = 0.2   # use a dense design when more than 20% of entries are non-zero
_MAX_BLOCK = 4_000_000  # cap on the number of temporary entries built at once


class LinearDesign:
    """Stack of M score matrices, one (K x d) block per profile."""

    def __init__(self, matrix, n_profiles, n_outcomes):
        self.matrix = matrix
        self.n_profiles = n_profiles
        self.n_outcomes = n_outcomes

    def scores(self, weights):
        """Scores for a batch of weight vectors (P, d) -> array (M, K, P)."""
        # numpy + Apple Accelerate can emit spurious FP warnings in matmul
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            s = self.matrix @ np.asarray(weights, dtype=float).T
        return np.asarray(s).reshape(self.n_profiles, self.n_outcomes, -1)


def choice_design(k, n, n_profiles, rng):
    """Position-count matrices of ``n_profiles`` impartial-culture profiles."""
    counts = np.empty((n_profiles, k, k))
    block = max(1, _MAX_BLOCK // (n * k))
    positions = np.arange(k)
    for start in range(0, n_profiles, block):
        b = min(block, n_profiles - start)
        # sigma[m, i, j] = alternative that voter i of profile m puts in position j
        sigma = random_permutations(rng, (b, n), k)
        flat = (np.arange(b)[:, None, None] * k + sigma) * k + positions
        counts[start:start + b] = np.bincount(flat.ravel(), minlength=b * k * k).reshape(b, k, k)
    return LinearDesign(counts.reshape(n_profiles * k, k), n_profiles, k)


@lru_cache(maxsize=None)
def _table(k):
    return relative_position_table(k)


def preference_design(k, n, n_profiles, rng):
    """Matrices A_p of ``n_profiles`` impartial-culture profiles (k! x k! each)."""
    table = _table(k)
    f = table.shape[0]
    samples = rng.integers(0, f, size=(n_profiles, n))
    flat = (np.arange(n_profiles)[:, None] * f + samples).ravel()
    hist = np.bincount(flat, minlength=n_profiles * f).reshape(n_profiles, f).astype(float)
    prof, perm = np.nonzero(hist)   # (profile, sampled ranking) pairs, sorted by profile
    mult = hist[prof, perm]
    density = len(prof) / (n_profiles * f)

    if density > DENSE_THRESHOLD:
        dense = np.zeros(n_profiles * f * f)
        block = max(1, _MAX_BLOCK // f)
        for start in range(0, len(prof), block):
            sl = slice(start, start + block)
            rows = prof[sl, None] * f + np.arange(f)          # row (m, r)
            flat = rows * f + table[perm[sl]]                 # column s^{-1} o r
            dense += np.bincount(flat.ravel(), weights=np.repeat(mult[sl], f),
                                 minlength=dense.size)
        matrix = dense.reshape(n_profiles * f, f)
    else:
        # for a fixed (m, r) the columns s^{-1} o r are distinct, so no duplicates
        rows = (prof[:, None] * f + np.arange(f)).ravel()
        cols = table[perm].ravel()
        data = np.repeat(mult, f)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_profiles * f, f))
    return LinearDesign(matrix, n_profiles, f)


DESIGNS = {"choice": choice_design, "preference": preference_design}


def dimension(setting, k):
    return k if setting == "choice" else len(_table(k))


def count_near_ties(scores, eps, atol):
    """Number of outcomes with s - s_min <= eps * |s_min| (+ atol), per profile.

    ``scores`` has shape (M, K, P); the result has shape (M, P). ``atol`` only
    absorbs floating point round-off, so that exact ties are always detected.
    """
    s_min = scores.min(axis=1, keepdims=True)
    tol = eps * np.abs(s_min) + atol
    return (scores <= s_min + tol).sum(axis=1)


class TieObjective:
    """Deterministic (sample average) objective on a fixed set of profiles."""

    def __init__(self, setting, k, n, eps, n_profiles, rng):
        self.design = DESIGNS[setting](k, n, n_profiles, rng)
        self.eps = eps
        self.atol = 1e-9 * n

    def per_profile(self, weights):
        return count_near_ties(self.design.scores(weights), self.eps, self.atol)

    def __call__(self, weights):
        return self.per_profile(weights).mean(axis=0)


def estimate(setting, k, n, eps, weights, n_profiles, rng, block_profiles=None):
    """Monte Carlo estimate (mean, standard error) for several weight vectors.

    All vectors are evaluated on the same fresh profiles (paired comparison).
    Profiles are generated in blocks to bound memory.
    """
    weights = np.atleast_2d(weights)
    if block_profiles is None:
        d = weights.shape[1]
        block_profiles = max(1, min(n_profiles, 20_000_000 // (d * d if setting == "preference" else k * k)))
    counts = []
    for start in range(0, n_profiles, block_profiles):
        b = min(block_profiles, n_profiles - start)
        design = DESIGNS[setting](k, n, b, rng)
        counts.append(count_near_ties(design.scores(weights), eps, 1e-9 * n))
    counts = np.concatenate(counts, axis=0)
    return counts.mean(axis=0), counts.std(axis=0, ddof=1) / np.sqrt(len(counts))
