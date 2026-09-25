"""OMRank: optimal metrics on the space of rankings (tie-avoiding social functions)."""
from .experiment import run_single
from .objectives import TieObjective, choice_design, estimate, preference_design
from .pso import SimplexPSO, project_rows_to_simplex
from .rules import choice_rules, preference_rules

__all__ = [
    "run_single", "TieObjective", "choice_design", "preference_design", "estimate",
    "SimplexPSO", "project_rows_to_simplex", "choice_rules", "preference_rules",
]
