"""Vectorised utilities on the symmetric group S_k.

Conventions
-----------
A ranking r in L([k]) is stored as a permutation ``sigma`` with
``sigma[p] = alternative placed at position p``, position 0 being the top
(``max r``). Permutations are enumerated in lexicographic order, so index 0 is
always the identity. Composition is ``(a o b)[p] = a[b[p]]``.

With these conventions a cost c : L(A) x L(A) -> R is neutral (invariant under
relabelling the alternatives, sigma -> rho o sigma) if and only if

    c(sigma, tau) = v(sigma^{-1} o tau)

for some v : S_k -> R, where (sigma^{-1} o tau)[p] is the position in sigma of
the alternative that tau places at position p. Kendall's distance corresponds to
v(pi) = number of inversions of pi.
"""
import itertools
import math

import numpy as np


def all_permutations(k):
    """All k! permutations of range(k) in lexicographic order, shape (k!, k)."""
    return np.array(list(itertools.permutations(range(k))), dtype=np.int64)


def inverse(perms):
    """Row-wise inverse of an array of permutations (..., k)."""
    return np.argsort(perms, axis=-1)


def lex_index(perms):
    """Lexicographic rank (Lehmer code) of each permutation in ``perms`` (..., k)."""
    perms = np.asarray(perms)
    k = perms.shape[-1]
    # smaller_right[..., i] = #{j > i : perms[j] < perms[i]}
    later = np.triu(np.ones((k, k), dtype=bool), 1)
    smaller = perms[..., None, :] < perms[..., :, None]
    smaller_right = (smaller & later).sum(axis=-1)
    radix = np.array([math.factorial(k - 1 - i) for i in range(k)], dtype=np.int64)
    return smaller_right @ radix


def relative_position_table(k, chunk=512):
    """Table T with T[s, r] = index of s^{-1} o r, shape (k!, k!).

    Row s lists, for every candidate ranking r, the index of the permutation at
    which the neutral cost c(s, r) = v(s^{-1} o r) must be evaluated.
    """
    perms = all_permutations(k)
    inv = inverse(perms)
    f = len(perms)
    dtype = np.int32 if f < 2**31 else np.int64
    table = np.empty((f, f), dtype=dtype)
    for start in range(0, f, chunk):
        stop = min(start + chunk, f)
        # composed[s, r, p] = inv[s][perms[r, p]]
        composed = inv[start:stop][:, perms]
        table[start:stop] = lex_index(composed)
    return table


def random_permutations(rng, size, k):
    """Uniform random permutations of range(k), shape (*size, k)."""
    if isinstance(size, int):
        size = (size,)
    return np.argsort(rng.random((*size, k)), axis=-1)
