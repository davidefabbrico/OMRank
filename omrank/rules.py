"""Classical rules written as neutral costs (baselines).

Every vector is normalised as in the optimisation: first entry 0 (top position /
identity permutation) and total sum 1. Both the argmin in (1.7)-(1.8) and the
relative epsilon-tie criterion are invariant under this positive rescaling.
"""
import numpy as np

from .permutations import all_permutations


def _normalise(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def choice_rules(k):
    """Positional costs c(r, a) = w[position of a in r], position 0 = top (cf. (1.9))."""
    pos = np.arange(k)
    return {
        "Plurality": _normalise(pos > 0),
        "Borda": _normalise(pos),
        "Antiplurality": _normalise(pos == k - 1),
    }


def _cycles(p):
    seen = np.zeros(len(p), dtype=bool)
    count = 0
    for i in range(len(p)):
        if not seen[i]:
            count += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return count


def _lis(p):
    tails = []
    for x in p:
        i = np.searchsorted(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def preference_rules(k):
    """Costs c(s, r) = v(s^{-1} o r) of classical metrics on rankings.

    With v = Kendall's distance the induced SPC (1.7) is Kemeny's rule.
    """
    perms = all_permutations(k)
    ident = np.arange(k)
    i, j = np.triu_indices(k, 1)
    kendall = (perms[:, i] > perms[:, j]).sum(axis=1)
    footrule = np.abs(perms - ident).sum(axis=1)
    spearman = ((perms - ident) ** 2).sum(axis=1)
    hamming = (perms != ident).sum(axis=1)
    cayley = np.array([k - _cycles(p) for p in perms])
    ulam = np.array([k - _lis(p) for p in perms])
    return {
        "Kemeny": _normalise(kendall),
        "Footrule": _normalise(footrule),
        "Spearman": _normalise(spearman),
        "Cayley": _normalise(cayley),
        "Hamming": _normalise(hamming),
        "Ulam": _normalise(ulam),
    }


RULES = {"choice": choice_rules, "preference": preference_rules}
