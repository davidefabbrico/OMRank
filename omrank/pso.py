"""Particle Swarm Optimisation on the probability simplex (Algorithm 1)."""
from dataclasses import dataclass, field

import numpy as np


def project_rows_to_simplex(y):
    """Euclidean projection of every row of ``y`` onto the probability simplex.

    Sort-based algorithm (Held et al. 1974; Duchi et al. 2008), O(d log d) per row.
    """
    y = np.atleast_2d(y)
    d = y.shape[1]
    u = -np.sort(-y, axis=1)
    css = np.cumsum(u, axis=1) - 1.0
    positive = u - css / np.arange(1, d + 1) > 0
    rho = d - 1 - np.argmax(positive[:, ::-1], axis=1)   # last index where positive
    theta = css[np.arange(len(y)), rho] / (rho + 1)
    return np.maximum(y - theta[:, None], 0.0)


def prepend_zero(x):
    """Simplex coordinates -> cost vector with first entry pinned to 0."""
    return np.hstack([np.zeros((len(x), 1)), x])


def monotone_positional(x):
    """Simplex coordinates u -> non-decreasing positional cost w, w_1 = 0, sum(w) = 1.

    w_j = sum_{i<j} u_i / (k - i): the vertices of the simplex are mapped onto the
    t-approval rules (Plurality for u = e_1, ..., Antiplurality for u = e_{k-1}).
    """
    k = x.shape[1] + 1
    increments = x / (k - np.arange(1, k))
    return np.hstack([np.zeros((len(x), 1)), np.cumsum(increments, axis=1)])


@dataclass
class PSOResult:
    best: np.ndarray          # full weight vector (first coordinate included)
    score: float              # objective value of ``best``
    history: list = field(default_factory=list)   # global best after each iteration
    n_iter: int = 0


class SimplexPSO:
    """Synchronous, vectorised PSO on the probability simplex of dimension ``n_free``.

    Particles live on the simplex; ``transform`` maps a (P, n_free) array of
    simplex points to the (P, d) cost vectors passed to ``objective``, which
    returns P scores. The default ``prepend_zero`` pins the first coordinate
    (cost of the top position / of the identity) to 0, fixing the affine
    normalisation of the cost; ``transform=None`` optimises on the simplex itself.
    ``init_points`` (simplex coordinates) replace the first Dirichlet particles (warm start).
    """

    def __init__(self, objective, n_free, n_particles=30, max_iter=1000, omega=0.9,
                 c1=2.0, c2=2.0, inertia_decay=0.99, transform=prepend_zero,
                 patience=None, init_points=None, rng=None):
        self.objective = objective
        self.n_free = n_free
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.inertia_decay = inertia_decay
        self.transform = transform
        self.patience = patience
        self.init_points = init_points
        self.rng = np.random.default_rng(rng)

    def _full(self, x):
        return x if self.transform is None else self.transform(x)

    def _evaluate(self, x):
        return np.asarray(self.objective(self._full(x)), dtype=float)

    def optimize(self):
        rng = self.rng
        x = rng.dirichlet(np.ones(self.n_free), size=self.n_particles)
        if self.init_points is not None:
            init = np.atleast_2d(self.init_points)[:self.n_particles]
            x[:len(init)] = init
        v = np.zeros_like(x)
        f = self._evaluate(x)
        pbest, pscore = x.copy(), f.copy()
        g = np.argmin(pscore)
        gbest, gscore = pbest[g].copy(), pscore[g]

        omega = self.omega
        history = [gscore]
        stall = 0
        t = 0
        for t in range(1, self.max_iter + 1):
            r1 = rng.random((self.n_particles, 1))
            r2 = rng.random((self.n_particles, 1))
            v = omega * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = project_rows_to_simplex(x + v)
            f = self._evaluate(x)

            improved = f < pscore
            pbest[improved] = x[improved]
            pscore[improved] = f[improved]
            g = np.argmin(pscore)
            if pscore[g] < gscore:
                gbest, gscore = pbest[g].copy(), pscore[g]
                stall = 0
            else:
                stall += 1

            omega *= self.inertia_decay
            history.append(gscore)
            if self.patience is not None and stall >= self.patience:
                break

        return PSOResult(self._full(gbest[None])[0], float(gscore), history, t)


TRANSFORMS = {"free": prepend_zero, "monotone": monotone_positional}
