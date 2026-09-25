"""Build the LaTeX tables and PDF figures of Section 5 from the CSV files in results/.

The output goes to the manuscript folder, outside the package:

    python make_report.py --out ../Social_Choice      # writes <out>/tables and <out>/figures
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# categorical slots (fixed order) + text/grid tokens of the reference palette
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def load(path):
    df = pd.read_csv(path)
    df["best"] = df["best"].map(lambda s: np.array(json.loads(s)))
    df["history"] = df["history"].map(json.loads)
    return df


def cv(w):
    """Coefficient of variation of a cost vector summing to one."""
    return np.sqrt(len(w) * np.sum(np.square(w)) - 1)


def fmt_sd(x):
    """Compact standard deviation, e.g. 3e-3."""
    if x < 5e-4:
        return "$<$1e-3" if x > 0 else "0"
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}e{int(exp)}"


# ─────────────────────────────── tables ───────────────────────────────

def results_table(df, caption, label, monotone=None):
    """Rows (k, n), columns eps; optimal value on test profiles, mean (sd) over replicas."""
    ks, ns, epss = sorted(df.k.unique()), sorted(df.n.unique()), sorted(df.eps.unique())
    groups = [df.groupby(["k", "n", "eps"])]
    if monotone is not None:
        groups.append(monotone.groupby(["k", "n", "eps"]))
    per = len(groups)
    head = " & ".join(["Free", "Monotone"][:per]) if per == 2 else "Optimum"

    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{rr" + "c" * per * len(epss) + "}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{{per}}}{{c}}{{$\varepsilon={e * 100:g}\%$}}" for e in epss) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + per * i}-{2 + per * (i + 1)}}}" for i in range(len(epss))),
             r"$k$ & $n$ & " + " & ".join([head] * len(epss)) + r" \\", r"\midrule"]
    for k in ks:
        for j, n in enumerate(ns):
            cells = [str(k) if j == 0 else "", str(n)]
            for eps in epss:
                for grp in groups:
                    g = grp.get_group((k, n, eps)).test_MCPSO
                    cells.append(f"{fmt_value(g.mean())} ({fmt_sd(g.std())})")
            lines.append(" & ".join(cells) + r" \\")
        if k != ks[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def fmt_value(v):
    return f"{v:.3f}" if v < 10 else f"{v:.1f}"


# ─────────────────────────────── figures ──────────────────────────────

def fig_convergence(choice, pref, path, n=200, eps=0.03):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for ax, df, title in ((axes[0], choice, "Choice"), (axes[1], pref, "Preference")):
        sub = df[(df.n == n) & np.isclose(df.eps, eps)]
        for i, k in enumerate(sorted(sub.k.unique())):
            hist = np.array(sub[sub.k == k].history.tolist())
            med = np.median(hist, axis=0)
            q25, q75 = np.percentile(hist, [25, 75], axis=0)
            it = np.arange(1, len(med) + 1)
            ax.fill_between(it, q25, q75, color=SERIES[i], alpha=0.18, lw=0, zorder=1)
            ax.plot(it, med, color=SERIES[i], ls=LINESTYLES[i], label=f"$k={k}$", zorder=2)
            marks = np.array([1, 3, 10, 30, 100, 300, 1000])
            marks = marks[marks <= len(med)] - 1
            ax.plot(it[marks], med[marks], ls="none", marker=MARKERS[i], ms=4.5, color=SERIES[i],
                    mec="white", mew=0.8)
        ax.set_xscale("log")
        ax.set_title(title, color=INK)
        ax.set_xlabel("PSO iteration")
    axes[0].set_ylabel(r"$\hat f$ on training profiles")
    handles = [plt.Line2D([], [], color=SERIES[i], ls=LINESTYLES[i], marker=MARKERS[i], ms=4.5, mec="white")
               for i in range(4)]
    for ax, df in ((axes[0], choice), (axes[1], pref)):
        ax.legend(handles, [f"$k={k}$" for k in sorted(df.k.unique())], loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def gaussian_ties(k, x, reps=200_000, seed=0):
    """G_k(x) = E #{a : X_a - min X <= x}, X exchangeable Gaussian with corr -1/(k-1)."""
    z = np.random.default_rng(seed).standard_normal((reps, k))
    gaps = np.sqrt(k / (k - 1)) * (z - z.mean(axis=1, keepdims=True))
    gaps = np.sort(gaps - gaps.min(axis=1, keepdims=True), axis=1)
    return np.array([(gaps <= t).sum(axis=1).mean() for t in x])


def fig_gaussian(choice, monotone, path):
    """Test tie counts versus x = eps sqrt(n) / CV(w) with the Gaussian prediction G_k(x)."""
    ks = sorted(choice.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(6.6, 2.1), sharey=False)
    for ax, k in zip(axes, ks):
        points = []   # (x, y, is_monotone)
        for (n, eps), g in choice[choice.k == k].groupby(["n", "eps"]):
            points.append(((eps * np.sqrt(n) / g.best.map(cv)).mean(), g.test_MCPSO.mean(), False))
            if monotone is not None:
                gm = monotone[(monotone.k == k) & (monotone.n == n) & np.isclose(monotone.eps, eps)]
                points.append(((eps * np.sqrt(n) / gm.best.map(cv)).mean(), gm.test_MCPSO.mean(), True))
        px, py = np.array([p[0] for p in points]), np.array([p[1] for p in points])
        xs = np.logspace(np.log10(px.min() / 2), np.log10(px.max() * 2), 60)
        ax.plot(xs, gaussian_ties(k, xs, reps=50_000), color=INK_2, lw=1.0, zorder=1)
        for x, y, mono in points:
            if mono:
                ax.scatter(x, y, s=22, marker="s", facecolor="none", edgecolor=SERIES[1], lw=1.1, zorder=4)
            else:
                ax.scatter(x, y, s=18, color=SERIES[0], edgecolor="white", lw=0.6, zorder=3)
        ax.set_ylim(0.95, py.max() * 1.12)
        ax.set_xscale("log")
        ax.set_title(f"$k={k}$", color=INK)
        ax.set_xlabel(r"$\varepsilon\sqrt{n}\,/\,\mathrm{CV}(w)$")
    axes[0].set_ylabel(r"$\mathbb{E}\,[\#\,\varepsilon$-winners$]$")
    handles = [plt.Line2D([], [], color=INK_2, lw=1.0),
               plt.Line2D([], [], ls="none", marker="o", color=SERIES[0], mec="white"),
               plt.Line2D([], [], ls="none", marker="s", mfc="none", mec=SERIES[1], mew=1.1)]
    labels = [r"Gaussian approximation $G_k$", "optimal cost", "optimal monotone cost"]
    if monotone is None:
        del handles[2], labels[2]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_monotone_weights(monotone, path, k=10, eps=0.03):
    ns = sorted(monotone.n.unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(6.6, 2.0), sharey=True)
    pos = np.arange(1, k + 1)
    for ax, n in zip(axes, ns):
        g = monotone[(monotone.k == k) & (monotone.n == n) & np.isclose(monotone.eps, eps)]
        w = np.array(g.best.tolist())
        ax.bar(pos, w.mean(axis=0), width=0.7, color=SERIES[1], zorder=2)
        ax.errorbar(pos, w.mean(axis=0), yerr=w.std(axis=0, ddof=1), fmt="none", ecolor=INK_2, lw=0.8, zorder=3)
        ax.set_title(f"$n={n}$", color=INK)
        ax.set_xticks(pos)
        ax.set_xlabel("position $j$")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("$w_j$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", required=True, help="manuscript folder (tables/ and figures/ are created inside)")
    args = parser.parse_args()
    tables, figures = os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    choice = load(os.path.join(args.results, "choice.csv"))
    pref = load(os.path.join(args.results, "preference.csv"))
    # monotone costs: the warm-started runs are reported (see Section 5.2.1)
    mono_path = os.path.join(args.results, "choice_monotone_warm.csv")
    monotone = load(mono_path) if os.path.exists(mono_path) else None
    reps = choice.groupby(["k", "n", "eps"]).size().max()

    with open(os.path.join(tables, "choice.tex"), "w") as fh:
        fh.write(results_table(
            choice,
            "Choice setting: optimal expected number of $\\varepsilon$-winners, estimated on $10^4$ independent "
            f"test profiles; mean over {reps} replicas, standard deviation in parentheses. \\emph{{Free}}: "
            "optimisation over $\\mathcal{W}_k$; \\emph{Monotone}: optimisation over $\\mathcal{W}_k^{\\uparrow}$.",
            "tab:choice", monotone))
    with open(os.path.join(tables, "preference.tex"), "w") as fh:
        fh.write(results_table(
            pref,
            "Preference setting: optimal expected number of $\\varepsilon$-winning rankings, estimated on $10^4$ "
            f"independent test profiles; mean over {reps} replicas, standard deviation in parentheses.",
            "tab:preference"))

    fig_convergence(choice, pref, os.path.join(figures, "convergence.pdf"))
    fig_gaussian(choice, monotone, os.path.join(figures, "gaussian_prediction.pdf"))
    if monotone is not None:
        fig_monotone_weights(monotone, os.path.join(figures, "monotone_weights.pdf"))
    print("tables ->", tables, "| figures ->", figures)


if __name__ == "__main__":
    main()
