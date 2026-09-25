"""Correctness checks. Run with ``python -m pytest tests`` or ``python tests/test_omrank.py``."""
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omrank.objectives import choice_design, count_near_ties, preference_design  # noqa: E402
from omrank.permutations import (all_permutations, inverse, lex_index,  # noqa: E402
                                 random_permutations, relative_position_table)
from omrank.pso import SimplexPSO, monotone_positional, project_rows_to_simplex  # noqa: E402
from omrank.rules import choice_rules, preference_rules  # noqa: E402


def test_lex_index_and_table():
    for k in range(1, 6):
        perms = all_permutations(k)
        assert np.array_equal(lex_index(perms), np.arange(math.factorial(k)))
        assert np.array_equal(perms[0], np.arange(k))
    k = 4
    perms = [tuple(p) for p in itertools.permutations(range(k))]
    index = {p: i for i, p in enumerate(perms)}
    table = relative_position_table(k)
    for s, ps in enumerate(perms):
        inv = np.argsort(ps)
        for r, pr in enumerate(perms):
            assert table[s, r] == index[tuple(inv[list(pr)])]


def test_choice_design_matches_naive():
    k, n, m = 5, 17, 6
    w = np.random.default_rng(0).dirichlet(np.ones(k))
    design = choice_design(k, n, m, np.random.default_rng(1))
    sigma = random_permutations(np.random.default_rng(1), (m, n), k)
    naive = np.zeros((m, k))
    for p in range(m):
        for i in range(n):
            for j in range(k):
                naive[p, sigma[p, i, j]] += w[j]
    assert np.allclose(design.scores(w[None])[..., 0], naive)


def _naive_preference(k, samples, v):
    perms = all_permutations(k)
    out = np.zeros((len(samples), len(perms)))
    for p, prof in enumerate(samples):
        for r in range(len(perms)):
            for s in prof:
                out[p, r] += v[lex_index(inverse(perms[s])[perms[r]])]
    return out


def test_preference_design_dense_and_sparse():
    k, m = 4, 3
    v = np.random.default_rng(0).dirichlet(np.ones(24))
    for n in (2, 40):   # sparse and dense code paths
        design = preference_design(k, n, m, np.random.default_rng(2))
        samples = np.random.default_rng(2).integers(0, 24, size=(m, n))
        assert np.allclose(design.scores(v[None])[..., 0], _naive_preference(k, samples, v))


def test_kendall_gives_kemeny():
    k, n = 4, 9
    perms = all_permutations(k)
    v = preference_rules(k)["Kemeny"]
    design = preference_design(k, n, 5, np.random.default_rng(3))
    samples = np.random.default_rng(3).integers(0, 24, size=(5, n))
    scores = design.scores(v[None])[..., 0]
    for p in range(5):
        pos = inverse(perms[samples[p]])           # pos[i, a] = position of a for voter i
        kemeny = np.array([sum(np.sum((pos[:, a] < pos[:, b]) != (np.argsort(r)[a] < np.argsort(r)[b]))
                               for a, b in itertools.combinations(range(k), 2)) for r in perms])
        assert set(np.flatnonzero(scores[p] <= scores[p].min() + 1e-12)) == \
            set(np.flatnonzero(kemeny == kemeny.min()))


def test_plurality_and_borda():
    k, n = 5, 11
    design = choice_design(k, n, 20, np.random.default_rng(4))
    counts = design.matrix.reshape(20, k, k)   # counts[m, a, j]
    rules = choice_rules(k)
    for name, expected in (("Plurality", -counts[:, :, 0]), ("Borda", counts @ np.arange(k))):
        s = design.scores(rules[name][None])[..., 0]
        for p in range(20):
            assert np.array_equal(np.flatnonzero(np.isclose(s[p], s[p].min())),
                                  np.flatnonzero(expected[p] == expected[p].min()))


def test_neutrality():
    """Psi(rho o r_1, ..., rho o r_n) = rho o Psi(r_1, ..., r_n) for a random cost."""
    k, n = 4, 7
    perms = all_permutations(k)
    v = np.random.default_rng(5).random(24)
    v[0] = 0
    table = relative_position_table(k)
    rng = np.random.default_rng(6)
    for _ in range(20):
        prof = rng.integers(0, 24, n)
        rho = rng.permutation(k)
        relabelled = lex_index(rho[perms[prof]])
        s = v[table[prof]].sum(axis=0)
        s_rel = v[table[relabelled]].sum(axis=0)
        win = {tuple(rho[perms[r]]) for r in np.flatnonzero(s == s.min())}
        win_rel = {tuple(perms[r]) for r in np.flatnonzero(s_rel == s_rel.min())}
        assert win == win_rel


def test_tie_count_and_projection():
    s = np.array([[10.0, 10.05, 10.2, 12.0]])[..., None]
    assert count_near_ties(s, 0.01, 0.0)[0, 0] == 2
    assert count_near_ties(s, 0.0, 1e-12)[0, 0] == 1
    rng = np.random.default_rng(7)
    y = rng.normal(size=(50, 6)) * 3
    x = project_rows_to_simplex(y)
    assert np.allclose(x.sum(axis=1), 1) and (x >= 0).all()
    z = rng.dirichlet(np.ones(6), size=(50, 2000))      # the projection is the closest point
    assert (((z - y[:, None]) ** 2).sum(-1) >= ((x - y) ** 2).sum(-1)[:, None] - 1e-12).all()
    assert np.allclose(project_rows_to_simplex(z[:, 0]), z[:, 0])


def test_pso_finds_simple_minimum():
    target = np.array([0.0, 0.2, 0.5, 0.3])
    res = SimplexPSO(lambda w: ((w - target) ** 2).sum(axis=1), 3, max_iter=300, rng=0).optimize()
    assert np.allclose(res.best, target, atol=1e-3)
    assert all(a >= b for a, b in zip(res.history, res.history[1:]))


def test_monotone_transform():
    k = 6
    u = np.random.default_rng(8).dirichlet(np.ones(k - 1), size=100)
    w = monotone_positional(u)
    assert np.allclose(w.sum(axis=1), 1) and (w[:, 0] == 0).all() and (np.diff(w, axis=1) >= 0).all()
    vertices = monotone_positional(np.eye(k - 1))
    rules = choice_rules(k)
    assert np.allclose(vertices[0], rules["Plurality"]) and np.allclose(vertices[-1], rules["Antiplurality"])
    # Borda is attained in the interior
    borda = rules["Borda"]
    u_borda = np.diff(borda) * (k - np.arange(1, k))
    assert np.allclose(u_borda.sum(), 1) and np.allclose(monotone_positional(u_borda[None])[0], borda)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
