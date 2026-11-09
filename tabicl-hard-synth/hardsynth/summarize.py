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
    a = p.parse_args(argv)
    df = load(a.results)
    a.out.write_text(gap_table(df) + "\n")
    print(gap_table(df))


if __name__ == "__main__":
    main()
