"""Turn benchmark CSVs into the paper's figures and tables.

    python experiments/make_figures.py [--results results] [--paper paper]

Writes PNG/PDF figures to <paper>/figures and LaTeX tables to <paper>/tables,
plus results/summary.md for the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geolabelqa.synthetic import CLASSES, inject_noise, make_landcover  # noqa: E402

# Categorical slots in fixed order; ink and grid are recessive greys.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, ERROR = "#0b0b0b", "#52514e", "#e4e3df", "#d03b3b"
BLUES = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]

DETECTORS = [  # (detector, cv, legend label) - order fixes colour
    ("neighbour_vote", "-", "Neighbour vote"),
    ("self_confidence", "random", "Self-confidence, random CV"),
    ("self_confidence", "spatial", "Self-confidence, spatial CV"),
    ("smoothed_sc", "spatial", "Smoothed SC, spatial CV (ours)"),
]
NOISE_LABEL = {"uniform": "Uniform", "pair": "Class-pair", "clustered": "Spatially clustered"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, out: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def select(det: pd.DataFrame, features: str, detector: str, cv: str) -> pd.DataFrame:
    feat = "-" if detector == "neighbour_vote" else features
    return det[(det.features == feat) & (det.detector == detector) & (det.cv == cv)]


def fig_study_area(out: Path):
    ds = make_landcover(seed=0)
    noisy, flip = inject_noise(ds.y, ds.coords, "clustered", 0.2, seed=1000)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    cmap = ListedColormap(SERIES)
    for ax, title in zip(axes, ("True land cover and sample points", "Spatially clustered label errors (20%)")):
        ax.imshow(ds.class_map, origin="lower", extent=(0, ds.extent, 0, ds.extent), cmap=cmap, vmin=-0.5, vmax=5.5, alpha=0.55)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Easting (km)")
        ax.grid(False)
    axes[0].set_ylabel("Northing (km)")
    axes[0].scatter(*ds.coords.T, s=2, c=INK, linewidths=0)
    axes[1].scatter(*ds.coords[~flip].T, s=2, c=INK2, linewidths=0, alpha=0.5)
    axes[1].scatter(*ds.coords[flip].T, s=9, marker="x", c=ERROR, linewidths=0.8, label="corrupted label")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.55) for c in SERIES]
    fig.legend(handles + [axes[1].collections[-1]], list(CLASSES) + ["corrupted label"],
               loc="lower center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.06), handlelength=1.2)
    save(fig, out, "fig1_study_area")


def fig_ap_vs_rate(det: pd.DataFrame, out: Path, features: str, name: str):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, noise in zip(axes, ("uniform", "pair", "clustered")):
        for colour, (detector, cv, label) in zip(SERIES, DETECTORS):
            g = select(det[det.noise == noise], features, detector, cv).groupby("rate")["ap"]
            m, s = g.mean(), g.std()
            ax.plot(m.index, m.values, color=colour, lw=2, marker="o", ms=5, label=label)
            ax.fill_between(m.index, m - s, m + s, color=colour, alpha=0.12, linewidth=0)
        ax.set_title(NOISE_LABEL[noise], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Noise rate")
        ax.set_xticks([0.1, 0.2, 0.3])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Average precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    save(fig, out, name)


def fig_smoothing(abl: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=False)
    sub = abl[(abl.features == "spectral+xy") & (abl.cv == "spatial")]
    for ax, noise in zip(axes, ("uniform", "pair", "clustered")):
        g = sub[sub.noise == noise].groupby(["k", "self_weight"])["ap"].mean().unstack(0)
        for colour, k in zip(BLUES, g.columns):
            ax.plot(g.index, g[k], color=colour, lw=2, marker="o", ms=4, label=f"k = {k}")
        ax.set_title(NOISE_LABEL[noise], fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Self weight $w$")
    axes[0].set_ylabel("Average precision")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    save(fig, out, "fig3_smoothing_ablation")


def fmt(m, s):
    return f"{m:.3f} $\\pm$ {s:.3f}"


def table_detection(det: pd.DataFrame, rate: float) -> tuple[str, str]:
    """LaTeX and Markdown tables of AP / P@N at one noise rate."""
    rows_tex, rows_md = [], []
    sub = det[det.rate == rate]
    configs = [("-", "neighbour_vote", "-", "Neighbour vote")]
    for fs in ("spectral", "spectral+xy"):
        for detector, dname in (("self_confidence", "Self-confidence"), ("smoothed_sc", "Smoothed SC (ours)")):
            for cv in ("random", "spatial"):
                configs.append((fs, detector, cv, dname))
    best = {}
    for noise in ("uniform", "pair", "clustered"):
        means = [sub[(sub.noise == noise) & (sub.features == fs) & (sub.detector == dt) & (sub.cv == cv)]["ap"].mean()
                 for fs, dt, cv, _ in configs]
        best[noise] = int(np.nanargmax(means))
    for i, (fs, dt, cv, dname) in enumerate(configs):
        cells_tex, cells_md = [], []
        for noise in ("uniform", "pair", "clustered"):
            r = sub[(sub.noise == noise) & (sub.features == fs) & (sub.detector == dt) & (sub.cv == cv)]
            ap = fmt(r.ap.mean(), r.ap.std())
            cells_tex.append(f"\\textbf{{{ap}}}" if best[noise] == i else ap)
            cells_md.append(f"{'**' if best[noise] == i else ''}{r.ap.mean():.3f}{'**' if best[noise] == i else ''}")
        fs_name = {"-": "--", "spectral": "spectral", "spectral+xy": "spectral + x,y"}[fs]
        rows_tex.append(f"{dname} & {fs_name} & {cv.replace('-', '--')} & " + " & ".join(cells_tex) + r" \\")
        rows_md.append(f"| {dname} | {fs_name.replace('--', '–')} | {cv.replace('-', '–')} | " + " | ".join(cells_md) + " |")
    tex = "\n".join([
        r"\begin{tabular}{lll ccc}", r"\toprule",
        r"Detector & Features & CV & Uniform & Class-pair & Clustered \\", r"\midrule",
        *rows_tex, r"\bottomrule", r"\end{tabular}",
    ])
    md = "\n".join(["| Detector | Features | CV | Uniform | Class-pair | Clustered |", "|---|---|---|---|---|---|", *rows_md])
    return tex, md


def table_cl_flags(det: pd.DataFrame, rate: float) -> str:
    sub = det[(det.rate == rate) & (det.features == "spectral+xy") & (det.cv == "spatial")]
    rows = []
    for detector, dname in (("self_confidence", "Raw probabilities"), ("smoothed_sc", "Smoothed probabilities (ours)")):
        cells = []
        for noise in ("uniform", "pair", "clustered"):
            r = sub[(sub.noise == noise) & (sub.detector == detector)]
            cells.append(f"{r.cl_precision.mean():.2f} / {r.cl_recall.mean():.2f} / {r.cl_f1.mean():.2f}")
        rows.append(f"{dname} & " + " & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{tabular}{l ccc}", r"\toprule",
        r"Confident-learning flags on & Uniform & Class-pair & Clustered \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
    ])


def table_downstream(down: pd.DataFrame, rate: float) -> tuple[str, str]:
    sub = down[(down.rate == rate) & (down.features == "spectral+xy")]
    methods = [
        ("no_cleaning", "No cleaning"),
        ("remove:neighbour_vote|-", "Remove top-$n$: neighbour vote"),
        ("remove:self_confidence|random", "Remove top-$n$: SC, random CV"),
        ("remove:self_confidence|spatial", "Remove top-$n$: SC, spatial CV"),
        ("remove:smoothed_sc|spatial", "Remove top-$n$: smoothed SC, spatial CV (ours)"),
        ("oracle_remove_noisy", "Oracle: remove all corrupted"),
        ("oracle_clean_labels", "Oracle: all labels correct"),
    ]
    rows_tex, rows_md = [], []
    for key, name in methods:
        cells = []
        for noise in ("uniform", "pair", "clustered"):
            r = sub[(sub.noise == noise) & (sub.method == key)]
            cells.append((r.macro_f1.mean(), r.macro_f1.std()))
        rows_tex.append(f"{name} & " + " & ".join(fmt(m, s) for m, s in cells) + r" \\")
        rows_md.append(f"| {name.replace('$n$', 'n')} | " + " | ".join(f"{m:.3f}" for m, _ in cells) + " |")
        if key == "no_cleaning" or key == "remove:smoothed_sc|spatial":
            rows_tex.append(r"\midrule")
    tex = "\n".join([r"\begin{tabular}{l ccc}", r"\toprule",
                     r"Training labels & Uniform & Class-pair & Clustered \\", r"\midrule",
                     *rows_tex, r"\bottomrule", r"\end{tabular}"])
    md = "\n".join(["| Training labels | Uniform | Class-pair | Clustered |", "|---|---|---|---|", *rows_md])
    return tex, md


def table_block_size(abl: pd.DataFrame) -> str:
    g = abl.groupby(["block_size", "detector"])["ap"].agg(["mean", "std"]).unstack("detector")
    rows = [f"{bs:g} & {fmt(g.loc[bs, ('mean', 'self_confidence')], g.loc[bs, ('std', 'self_confidence')])} & "
            f"{fmt(g.loc[bs, ('mean', 'smoothed_sc')], g.loc[bs, ('std', 'smoothed_sc')])} \\\\" for bs in g.index]
    return "\n".join([r"\begin{tabular}{c cc}", r"\toprule",
                      r"Block size (km) & Self-confidence & Smoothed SC \\", r"\midrule",
                      *rows, r"\bottomrule", r"\end{tabular}"])


REAL_CONFIGS = [  # (features, detector, cv, label)
    ("-", "neighbour_vote", "-", "Neighbour vote"),
    ("spectral", "self_confidence", "random", "SC, random CV"),
    ("spectral", "self_confidence", "spatial", "SC, spatial CV"),
    ("spectral", "smoothed_sc", "random", "Smoothed SC, random CV"),
    ("spectral", "smoothed_sc", "spatial", "Smoothed SC, spatial CV"),
    ("spectral+xy", "self_confidence", "random", "SC, random CV, +x,y"),
    ("spectral+xy", "self_confidence", "spatial", "SC, spatial CV, +x,y"),
    ("spectral+xy", "smoothed_sc", "random", "Smoothed SC, random CV, +x,y"),
    ("spectral+xy", "smoothed_sc", "spatial", "Smoothed SC, spatial CV, +x,y"),
]
SCENE_LABEL = {"indian_pines": "Indian Pines", "pavia_university": "Pavia University"}


def table_real(real: pd.DataFrame) -> tuple[str, str]:
    """AP per scene and noise regime on the real hyperspectral scenes."""
    noises = ("uniform", "pair", "clustered")
    tex = [r"\begin{tabular}{l ccc}", r"\toprule", r"Detector & Uniform & Class-pair & Clustered \\"]
    md = ["| Scene | Detector | Uniform | Class-pair | Clustered |", "|---|---|---|---|---|"]
    for scene in real.scene.unique():
        sub = real[real.scene == scene]
        means = {n: [sub[(sub.noise == n) & (sub.features == fs) & (sub.detector == dt) & (sub.cv == cv)].ap.mean()
                     for fs, dt, cv, _ in REAL_CONFIGS] for n in noises}
        best = {n: int(np.nanargmax(means[n])) for n in noises}
        tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\textit{{{SCENE_LABEL[scene]}}}}} \\"]
        for i, (fs, dt, cv, label) in enumerate(REAL_CONFIGS):
            cells = []
            for n in noises:
                r = sub[(sub.noise == n) & (sub.features == fs) & (sub.detector == dt) & (sub.cv == cv)].ap
                cell = fmt(r.mean(), r.std())
                cells.append(rf"\textbf{{{cell}}}" if best[n] == i else cell)
            tex.append(f"{label} & " + " & ".join(cells) + r" \\")
            md.append(f"| {SCENE_LABEL[scene]} | {label} | " + " | ".join(
                f"{'**' if best[n] == i else ''}{means[n][i]:.3f}{'**' if best[n] == i else ''}" for n in noises) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex), "\n".join(md)


def table_real_downstream(down: pd.DataFrame) -> str:
    methods = [("no_cleaning", "No cleaning"),
               ("remove:neighbour_vote|-", "Remove top-$n$: neighbour vote"),
               ("remove:self_confidence|random", "Remove top-$n$: SC, random CV"),
               ("remove:smoothed_sc|spatial", "Remove top-$n$: smoothed SC, spatial CV"),
               ("oracle_remove_noisy", "Oracle: remove all corrupted"),
               ("oracle_clean_labels", "Oracle: all labels correct")]
    noises = ("uniform", "pair", "clustered")
    tex = [r"\begin{tabular}{l ccc}", r"\toprule", r"Training labels & Uniform & Class-pair & Clustered \\"]
    for scene in down.scene.unique():
        sub = down[down.scene == scene]
        tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\textit{{{SCENE_LABEL[scene]}}}}} \\"]
        for key, name in methods:
            cells = [sub[(sub.noise == n) & (sub.method == key)].macro_f1 for n in noises]
            tex.append(f"{name} & " + " & ".join(fmt(c.mean(), c.std()) for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex)


def paired_test(df: pd.DataFrame, a: tuple, b: tuple, keys: list[str]) -> tuple[float, int, int, float]:
    """Median AP difference a - b, wins of a, number of pairs, Wilcoxon signed-rank p (two-sided)."""
    from scipy.stats import wilcoxon

    def pick(cfg):
        fs, dt, cv = cfg
        return df[(df.features == fs) & (df.detector == dt) & (df.cv == cv)].set_index(keys).ap

    diff = (pick(a) - pick(b)).dropna()
    p = wilcoxon(diff).pvalue if len(diff) > 0 and (diff != 0).any() else float("nan")
    return float(diff.median()), int((diff > 0).sum()), len(diff), float(p)


def table_stats(det: pd.DataFrame, real: pd.DataFrame | None) -> str:
    comparisons = [
        ("Spatial vs random CV (SC, +x,y)", ("spectral+xy", "self_confidence", "spatial"), ("spectral+xy", "self_confidence", "random")),
        ("Smoothed vs raw SC (spatial CV, +x,y)", ("spectral+xy", "smoothed_sc", "spatial"), ("spectral+xy", "self_confidence", "spatial")),
        ("Smoothed vs raw SC (spatial CV, spectral)", ("spectral", "smoothed_sc", "spatial"), ("spectral", "self_confidence", "spatial")),
        ("Smoothed vs raw SC (random CV, spectral)", ("spectral", "smoothed_sc", "random"), ("spectral", "self_confidence", "random")),
        ("Smoothed SC + spatial CV vs standard CL", ("spectral", "smoothed_sc", "spatial"), ("spectral", "self_confidence", "random")),
    ]
    sources = [("Synthetic", det, ["rate", "seed"])]
    if real is not None:
        sources.append(("Real scenes", real, ["scene", "seed"]))

    def p_fmt(p):
        return "--" if np.isnan(p) else (r"$<$0.001" if p < 0.001 else f"{p:.3f}")

    rows = []
    for source, df, keys in sources:
        rows.append(r"\midrule")
        rows.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{source}}}}} \\")
        for label, a, b in comparisons:
            cells = []
            for noise in ("uniform", "pair", "clustered"):
                med, wins, n, p = paired_test(df[df.noise == noise], a, b, keys)
                cells.append(f"{med:+.3f} ({wins}/{n}, {p_fmt(p)})")
            rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join([r"\begin{tabular}{l ccc}", r"\toprule",
                      r"Comparison ($a$ vs $b$) & Uniform & Class-pair & Clustered \\", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--paper", default=str(ROOT / "paper"))
    args = parser.parse_args()
    res, paper = Path(args.results), Path(args.paper)
    figs, tabs = paper / "figures", paper / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    det = pd.read_csv(res / "detection.csv")
    down = pd.read_csv(res / "downstream.csv")
    abl_s = pd.read_csv(res / "ablation_smoothing.csv")
    abl_b = pd.read_csv(res / "ablation_block_size.csv")

    fig_study_area(figs)
    fig_ap_vs_rate(det, figs, "spectral+xy", "fig2_ap_vs_rate_xy")
    fig_ap_vs_rate(det, figs, "spectral", "fig2b_ap_vs_rate_spectral")
    fig_smoothing(abl_s, figs)

    det_tex, det_md = table_detection(det, 0.2)
    down_tex, down_md = table_downstream(down, 0.2)
    (tabs / "detection.tex").write_text(det_tex, encoding="utf-8")
    (tabs / "downstream.tex").write_text(down_tex, encoding="utf-8")
    (tabs / "cl_flags.tex").write_text(table_cl_flags(det, 0.2), encoding="utf-8")
    (tabs / "block_size.tex").write_text(table_block_size(abl_b), encoding="utf-8")

    real = pd.read_csv(res / "real_detection.csv") if (res / "real_detection.csv").exists() else None
    (tabs / "stats.tex").write_text(table_stats(det, real), encoding="utf-8")
    real_md = ""
    if real is not None:
        real_tex, real_md = table_real(real)
        (tabs / "real_detection.tex").write_text(real_tex, encoding="utf-8")
        (tabs / "real_downstream.tex").write_text(
            table_real_downstream(pd.read_csv(res / "real_downstream.csv")), encoding="utf-8")
        real_md = f"\n## Real hyperspectral scenes, average precision (20% injected noise)\n\n{real_md}\n"

    n_seeds = det.seed.nunique()
    (res / "summary.md").write_text(
        f"## Label-error detection, average precision (20% noise, {n_seeds} seeds)\n\n{det_md}\n\n"
        f"## Downstream macro-F1 on clean test labels (20% noise, features: spectral + x,y)\n\n{down_md}\n"
        + real_md,
        encoding="utf-8",
    )
    print((res / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
