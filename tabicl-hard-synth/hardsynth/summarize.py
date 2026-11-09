"""Collect results.json files into Markdown tables for the journal.

    python -m hardsynth.summarize journal/<iter>/*/results.json --out journal/<iter>/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(paths):
    rows = []
    for p in paths:
        r = json.loads(Path(p).read_text())
        metric = "accuracy" if r["task"] == "classification" else "rmse"
        for rec in r["records"]:
            rows.append({"dataset": Path(p).parent.name, "family": r["family"], "difficulty": r["difficulty"],
                         "task": r["task"], "model": rec["model"], "n_train": rec["n_train"], "metric": metric,
                         "value": rec[metric], "bayes": r["bayes"][metric], "fit_s": rec["fit_seconds"],
                         "dir": str(Path(p).parent)})
    return pd.DataFrame(rows)


def gap_table(df: pd.DataFrame) -> str:
    """One row per dataset: Bayes reference and each model's best value, plus
    the gain of the largest training size over the second largest (the
    'does more data help' signal)."""
    lines = []
    for ds, g in df.groupby("dataset", sort=False):
        metric = g["metric"].iloc[0]
        higher_better = metric == "accuracy"
        cells = [ds, g["family"].iloc[0], f"{g['difficulty'].iloc[0]:.2f}", metric, f"{g['bayes'].iloc[0]:.3f}"]
        for model in ("lgbm", "mlp", "linear", "tabicl", "tabicl-ft"):
            gm = g[g["model"] == model].sort_values("n_train")
            if len(gm) == 0:
                cells.append("")
                continue
            best = gm["value"].max() if higher_better else gm["value"].min()
            last, prev = gm.iloc[-1], (gm.iloc[-2] if len(gm) > 1 else None)
            delta = "" if prev is None else f" ({'+' if last['value'] - prev['value'] >= 0 else ''}{last['value'] - prev['value']:.3f} at {int(last['n_train']):,})"
            cells.append(f"{best:.3f}{delta}")
        lines.append("| " + " | ".join(cells) + " |")
    header = "| dataset | family | difficulty | metric | Bayes | LightGBM | MLP | linear | TabICL | TabICL-ft |"
    return "\n".join([header, "|" + "---|" * 10] + lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid-svg", type=Path, default=None, help="also draw all learning curves as small multiples")
    a = p.parse_args(argv)
    df = load(a.results)
    a.out.write_text(gap_table(df) + "\n")
    print(gap_table(df))
    if a.grid_svg:
        plot_grid(df, a.grid_svg)



def plot_grid(df: pd.DataFrame, path: Path, ncols: int = 4):
    """Small multiples: one panel per dataset, metric vs training rows, with
    the Bayes reference dashed and a single shared legend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .evaluate import INK, PALETTE

    datasets = list(dict.fromkeys(df["dataset"]))
    nrows = -(-len(datasets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), dpi=100, squeeze=False)
    fig.patch.set_facecolor(INK["surface"])
    for ax, ds in zip(axes.ravel(), datasets):
        g = df[df["dataset"] == ds]
        metric = g["metric"].iloc[0]
        ax.set_facecolor(INK["surface"])
        for model in PALETTE:
            gm = g[g["model"] == model].sort_values("n_train")
            if len(gm):
                ax.plot(gm["n_train"], gm["value"], color=PALETTE[model], lw=2, marker="o", ms=4, label=model)
        ax.axhline(g["bayes"].iloc[0], color=INK["secondary"], lw=1.2, ls="--", label="Bayes optimal")
        ax.set_xscale("log")
        ax.set_title(ds, fontsize=9, loc="left", color=INK["primary"])
        ax.grid(True, color=INK["grid"], lw=0.8)
        ax.tick_params(labelsize=7, colors=INK["secondary"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK["axis"])
        if metric == "accuracy":
            ax.set_ylim(min(0.3, g["value"].min() - 0.02), min(1.0, g["bayes"].iloc[0] + 0.05))
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("training rows", fontsize=8, color=INK["primary"])
    for ax in axes[:, 0]:
        ax.set_ylabel(df["metric"].iloc[0], fontsize=8, color=INK["primary"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, format=Path(path).suffix.lstrip(".") or "svg")
    plt.close(fig)

if __name__ == "__main__":
    main()
