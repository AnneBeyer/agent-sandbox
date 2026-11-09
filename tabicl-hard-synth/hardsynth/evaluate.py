"""Learning curves of baseline models on a generated dataset.

Example::

    python -m hardsynth.evaluate --data data/deep_interaction_d0.8 \
        --models lgbm mlp linear --train-sizes 1000 10000 100000 1000000 \
        --out journal/<iteration>/deep_interaction_d0.8

Writes ``results.json`` (one record per model x training size with metrics
and timings) and ``learning_curve.svg``.  The Bayes reference stored in the
dataset is drawn as a dashed line.

Models
------
lgbm       LightGBM with early stopping on 10% of the training subset,
           categoricals handled natively.
mlp        scikit-learn MLP (256-256) on target-encoded categoricals and
           standardised numerics, early stopping.
linear     logistic regression / ridge on the same preprocessing.
tabicl     TabICL (zero-shot in-context learning).  Needs the ``tabicl``
           pixi environment and the Hugging Face checkpoint.  The context
           is a random subsample of the training subset of at most
           ``--tabicl-max-context`` rows.
tabicl-ft  TabICL fine-tuned on the training subset with
           ``tabicl.FinetunedTabICLClassifier`` / ``...Regressor``
           (``--ft-epochs``).  Hook for the fine-tuning follow-up.

Training subsets are random subsamples of the training split.  For the
temporal family this means "same history length, fewer rows"; to study
"longer history" generate several datasets instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# dataviz default categorical palette (light surface), fixed slot order
PALETTE = {"lgbm": "#2a78d6", "mlp": "#eb6834", "linear": "#1baf7a", "tabicl": "#eda100", "tabicl-ft": "#e87ba4"}
INK = {"primary": "#0b0b0b", "secondary": "#5f5e5a", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}


def load_dataset(data_dir: Path):
    meta = json.loads((data_dir / "metadata.json").read_text())
    table = pq.read_table(data_dir / "data.parquet", memory_map=True)
    df = table.to_pandas(self_destruct=True)
    for c in meta["categorical"]:
        df[c] = df[c].astype("category")
    return df, meta


def _preprocessor(meta):
    cat = meta["categorical"]
    num = [c for c in meta["columns"] if c not in cat]
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(("cat", TargetEncoder(target_type="auto"), cat))
    return ColumnTransformer(transformers)


def make_model(name: str, meta: dict, args):
    task = meta["task"]
    cls = task == "classification"
    if name == "lgbm":
        import lightgbm as lgb

        params = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63, min_child_samples=20, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, n_jobs=args.n_jobs, verbose=-1, random_state=args.seed)
        return ("lgbm", lgb.LGBMClassifier(**params) if cls else lgb.LGBMRegressor(**params))
    if name == "mlp":
        kw = dict(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=200, n_iter_no_change=10,
                  batch_size=512, learning_rate_init=1e-3, random_state=args.seed)
        return ("sk", make_pipeline(_preprocessor(meta), MLPClassifier(**kw) if cls else MLPRegressor(**kw)))
    if name == "linear":
        est = LogisticRegression(max_iter=500, C=1.0) if cls else Ridge(alpha=1.0)
        return ("sk", make_pipeline(_preprocessor(meta), est))
    if name == "tabicl":
        from tabicl import TabICLClassifier, TabICLRegressor

        kw = dict(n_estimators=args.tabicl_n_estimators, batch_size=args.tabicl_batch_size)
        if args.device:
            kw["device"] = args.device
        return ("tabicl", TabICLClassifier(**kw) if cls else TabICLRegressor(**kw))
    if name == "tabicl-ft":
        from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

        kw = dict(epochs=args.ft_epochs, verbose=False)
        if args.device:
            kw["device"] = args.device
        return ("tabicl-ft", FinetunedTabICLClassifier(**kw) if cls else FinetunedTabICLRegressor(**kw))
    raise KeyError(name)


def _fit(kind, model, Xtr, ytr, args, rng):
    if kind == "lgbm":
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.1, random_state=args.seed)
        import lightgbm as lgb

        cb = [lgb.early_stopping(50, verbose=False)]
        try:  # LightGBM >= 4.6 renamed eval_set
            model.fit(Xa, ya, eval_X=Xv, eval_y=yv, callbacks=cb)
        except TypeError:
            model.fit(Xa, ya, eval_set=[(Xv, yv)], callbacks=cb)
        return {"n_trees": int(model.best_iteration_ or model.n_estimators)}
    if kind == "tabicl":
        if len(Xtr) > args.tabicl_max_context:
            idx = rng.choice(len(Xtr), args.tabicl_max_context, replace=False)
            Xtr, ytr = Xtr.iloc[idx], ytr[idx]
        model.fit(Xtr, ytr)
        return {"context_rows": int(len(Xtr))}
    model.fit(Xtr, ytr)
    return {}


def _chunked(fn, X, chunk):
    if not chunk or len(X) <= chunk:
        return fn(X)
    return np.concatenate([fn(X.iloc[i : i + chunk]) for i in range(0, len(X), chunk)])


def _metrics(task, model, Xte, yte, n_classes, chunk=None):
    out = {}
    if task == "classification":
        proba = _chunked(model.predict_proba, Xte, chunk)
        pred = np.asarray(model.classes_)[proba.argmax(1)]
        out["accuracy"] = float(accuracy_score(yte, pred))
        labels = np.asarray(model.classes_)
        out["log_loss"] = float(log_loss(yte, proba, labels=labels))
        try:
            if n_classes == 2:
                out["roc_auc"] = float(roc_auc_score(yte, proba[:, 1]))
            else:
                out["roc_auc"] = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=labels))
        except ValueError:
            out["roc_auc"] = float("nan")
    else:
        pred = _chunked(model.predict, Xte, chunk)
        out["rmse"] = float(np.sqrt(mean_squared_error(yte, pred)))
        out["r2"] = float(r2_score(yte, pred))
    return out


def evaluate(args) -> dict:
    df, meta = load_dataset(args.data)
    task = meta["task"]
    feats = [c for c in meta["columns"] if c not in set(getattr(args, "drop_columns", None) or [])]
    if getattr(args, "drop_columns", None):
        meta = dict(meta, columns=feats, categorical=[c for c in meta["categorical"] if c in feats])
    train = df[df["split"] == 0]
    test = df[df["split"] == 1]
    if args.test_rows and len(test) > args.test_rows:
        test = test.sample(args.test_rows, random_state=args.seed)
    Xte, yte = test[feats], test["y"].to_numpy()
    bayes = {}
    if task == "classification":
        bayes["accuracy"] = float(test["_p_bayes"].mean())
    else:
        bayes["rmse"] = float(np.sqrt(np.mean((test["y"] - test["_y_clean"]) ** 2)))
        bayes["r2"] = float(1 - np.sum((test["y"] - test["_y_clean"]) ** 2) / np.sum((test["y"] - test["y"].mean()) ** 2))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train))
    records = []
    for n in args.train_sizes:
        if n > len(train):
            print(f"skip n={n}: only {len(train)} training rows", file=sys.stderr)
            continue
        sub = train.iloc[perm[:n]]
        Xtr, ytr = sub[feats], sub["y"].to_numpy()
        for name in args.models:
            if name == "mlp" and n > args.mlp_max_rows:
                continue
            kind, model = make_model(name, meta, args)
            t0 = time.time()
            info = _fit(kind, model, Xtr, ytr, args, rng)
            t1 = time.time()
            chunk = args.tabicl_predict_chunk if kind in ("tabicl", "tabicl-ft") else None
            m = _metrics(task, model, Xte, yte, meta["n_classes"], chunk)
            t2 = time.time()
            rec = {"model": name, "n_train": int(n), **m, "fit_seconds": round(t1 - t0, 2), "predict_seconds": round(t2 - t1, 2), **info}
            records.append(rec)
            print(json.dumps(rec), file=sys.stderr)
    result = {"dataset": str(args.data), "family": meta["family"], "difficulty": meta["difficulty"], "task": task,
              "n_test": int(len(test)), "bayes": bayes, "knobs": meta["knobs"], "records": records,
              "dropped_columns": list(getattr(args, "drop_columns", None) or []),
              "hardware": meta.get("hardware"), "versions": meta.get("versions"), "command": " ".join(sys.argv)}
    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / "results.json"
    if getattr(args, "append", False) and out_file.exists():
        old = json.loads(out_file.read_text())
        new_keys = {(r["model"], r["n_train"]) for r in records}
        kept = [r for r in old["records"] if (r["model"], r["n_train"]) not in new_keys]
        result["records"] = kept + records
        old_n = old.get("n_test")
        old_n = old_n.get("previous", old_n) if isinstance(old_n, dict) else old_n
        if old_n and old_n >= len(test):  # keep the Bayes reference of the larger test sample
            result["bayes"] = old["bayes"]
        result["n_test"] = {"previous": old_n, "this_run": int(len(test))}
        result["commands"] = old.get("commands", [old.get("command")]) + [result["command"]]
    out_file.write_text(json.dumps(result, indent=2))
    plot_learning_curve(result, args.out / "learning_curve.svg")
    return result


def plot_learning_curve(result: dict, path: Path, title: str | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    task = result["task"]
    metric = "accuracy" if task == "classification" else "rmse"
    recs = pd.DataFrame(result["records"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
    fig.patch.set_facecolor(INK["surface"])
    ax.set_facecolor(INK["surface"])
    for name in PALETTE:  # fixed slot order
        sub = recs[recs["model"] == name] if len(recs) else recs
        if len(sub) == 0:
            continue
        sub = sub.sort_values("n_train")
        ax.plot(sub["n_train"], sub[metric], color=PALETTE[name], lw=2, marker="o", ms=5, label=name)
    b = result["bayes"][metric]
    ax.axhline(b, color=INK["secondary"], lw=1.2, ls="--")
    ax.annotate("Bayes optimal", xy=(1, b), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", color=INK["secondary"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("training rows", color=INK["primary"])
    ax.set_ylabel(metric, color=INK["primary"])
    ax.set_title(title or f"{result['family']} (difficulty {result['difficulty']})", color=INK["primary"], fontsize=11, loc="left")
    ax.grid(True, color=INK["grid"], lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK["axis"])
    ax.tick_params(colors=INK["secondary"])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, format=Path(path).suffix.lstrip(".") or "svg")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--models", nargs="+", default=["lgbm", "mlp", "linear"], choices=list(PALETTE))
    p.add_argument("--train-sizes", nargs="+", type=lambda s: int(s.replace("_", "")), default=[1000, 10_000, 100_000])
    p.add_argument("--test-rows", type=int, default=100_000, help="subsample the test split to this many rows")
    p.add_argument("--mlp-max-rows", type=int, default=300_000, help="skip the sklearn MLP above this size")
    p.add_argument("--tabicl-max-context", type=int, default=100_000)
    p.add_argument("--tabicl-n-estimators", type=int, default=8, help="TabICL ensemble size (memory and time scale with it)")
    p.add_argument("--tabicl-batch-size", type=int, default=8, help="ensemble members per forward pass")
    p.add_argument("--tabicl-predict-chunk", type=int, default=5000, help="test rows per predict call for TabICL models")
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--append", action="store_true", help="merge into an existing results.json instead of overwriting it")
    p.add_argument("--drop-columns", nargs="*", default=None,
                   help="feature columns to hide from every model, e.g. group_id (BeyondArena drops the group index)")
    evaluate(p.parse_args(argv))


if __name__ == "__main__":
    main()
