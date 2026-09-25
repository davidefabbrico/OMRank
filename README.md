# OMRank

Optimal metrics on the space of rankings for Social Choice Theory: costs on rankings whose induced
social choice / preference correspondences minimise the expected number of near-ties
(ε-winners), found by Monte Carlo + Particle Swarm Optimisation on the simplex (MCPSO).

## Layout

| Path | Content |
|------|---------|
| `omrank/permutations.py` | vectorised operations on S_k (lexicographic index, table of s⁻¹∘r) |
| `omrank/objectives.py` | Choice / Preference objectives (Algorithms 2–3), linear "design" matrices, common random numbers |
| `omrank/pso.py` | synchronous vectorised PSO on the simplex (Algorithm 1), free and monotone parametrisations |
| `omrank/rules.py` | classical costs: Plurality, Borda, Antiplurality; Kendall (Kemeny), footrule, Spearman, Cayley, Hamming, Ulam (used for the warm start and as reference values in the result files) |
| `omrank/experiment.py` | one replica: optimise on training profiles, evaluate on independent test profiles |
| `run_experiments.py` | full simulation study (parallel, reproducible seeds) → `results/*.csv` |
| `make_report.py` | tables (LaTeX) and figures (PDF) of Section 5, written to the manuscript folder given with `--out` |
| `benchmark_legacy.py` | speed comparison with the previous implementation → `results/benchmark.json` |
| `tests/test_omrank.py` | correctness checks against brute-force computations |
| `results/` | simulation results (`choice.csv`, `choice_monotone.csv`, `choice_monotone_warm.csv`, `preference.csv`, `benchmark.json`) |
| `legacy/` | previous implementation (`simulation_server.py` and the two notebooks), kept for reference |

## Usage

```bash
pip install -r requirements.txt
python tests/test_omrank.py                       # correctness checks

python run_experiments.py                         # Choice + Preference (≈35 min on 10 cores)
python run_experiments.py --setting choice --costs monotone               # monotone costs
python run_experiments.py --setting choice --costs monotone --warm-start  # + classical costs in the initial swarm
python make_report.py --out ../Social_Choice      # tables and figures of the manuscript
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark_legacy.py     # optional timing comparison
```

Quick run: `python run_experiments.py --setting choice --ks 3 5 --reps 2 --iters 200`.

## Conventions

A ranking is a permutation σ with σ(j) = alternative in position j (position 1 = top). Positional costs
are vectors w with w₁ = 0 and Σw = 1; neutral costs on rankings are c(r, s) = v(σ_r⁻¹∘σ_s) with
v(id) = 0 and Σv = 1. An ε-winner is an outcome whose score s satisfies s − s_min ≤ ε·|s_min|.
