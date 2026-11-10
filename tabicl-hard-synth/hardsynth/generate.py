"""Generate one dataset of a family as a chunked Parquet file plus metadata.

Example::

    python -m hardsynth.generate --family deep_interaction --difficulty 0.8 \
        --n-samples 10_000_000 --n-classes 3 --out data/deep_interaction_d0.8

Output layout::

    <out>/data.parquet     features..., y, _p_bayes | _y_clean, split (0 train, 1 test)
    <out>/metadata.json    generator config, derived knobs, column types,
                           Bayes reference on the test split, versions, timing

The last ``--test-size`` rows are the test split.  For ``temporal_drift``
they are the latest rows in time; for ``grouped_entities`` they are drawn
from entities that never occur in the training rows; for the other
families rows are IID so the position carries no information.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .families import FAMILIES, make_family


def _versions() -> dict:
    import pandas, pyarrow, scipy, sklearn  # noqa: E401

    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
           "pyarrow": pyarrow.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    try:
        import lightgbm

        out["lightgbm"] = lightgbm.__version__
    except Exception:  # pragma: no cover
        pass
    return out


def _hardware() -> dict:
    info = {"platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    info["mem_total_gb"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    return info


def _to_table(sample, split_value: int, task: str) -> pa.Table:
    cols = {}
    for k, v in sample.X.items():
        cols[k] = pa.array(v)
    cols["y"] = pa.array(sample.y)
    cols["_p_bayes" if task == "classification" else "_y_clean"] = pa.array(sample.bayes)
    cols["split"] = pa.array(np.full(len(sample.y), split_value, dtype=np.int8))
    return pa.table(cols)


def generate(
    family: str,
    out: Path,
    n_samples: int,
    test_size: int,
    chunk_size: int = 1_000_000,
    max_gb: float = 8.0,
    verbose: bool = True,
    **family_kwargs,
) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if test_size >= n_samples:
        raise ValueError("test_size must be smaller than n_samples")
    fam = make_family(family, n_total=n_samples, **family_kwargs)
    n_cols = len(fam.sample(8, 0).X) + 2
    est_gb = n_samples * n_cols * 4 / 1024**3
    if est_gb > max_gb:
        raise SystemExit(
            f"{family}: {n_samples:,} rows x {n_cols} columns is about {est_gb:.1f} GB in memory "
            f"(limit --max-gb {max_gb}). Reduce --n-samples or raise --max-gb."
        )

    n_train = n_samples - test_size
    t0 = time.time()
    writer = None
    bayes_sum, bayes_sq, n_test_written = 0.0, 0.0, 0
    y_test_sq = 0.0
    class_counts = None
    pos = 0
    path = out / "data.parquet"
    while pos < n_samples:
        n = min(chunk_size, n_samples - pos)
        # never let a chunk straddle the train/test boundary
        if pos < n_train < pos + n:
            n = n_train - pos
        is_test = pos >= n_train
        if is_test and fam.split_kind == "grouped":
            sample = fam.sample_test(n, pos)
        else:
            sample = fam.sample(n, pos)
        table = _to_table(sample, int(is_test), fam.task)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writer.write_table(table)
        if is_test:
            bayes_sum += float(sample.bayes.astype(np.float64).sum())
            if fam.task == "regression":
                bayes_sq += float(((sample.y - sample.bayes) ** 2).sum())
                y_test_sq += float(((sample.y - sample.y.mean()) ** 2).sum())
            n_test_written += n
        elif fam.task == "classification":
            bc = np.bincount(sample.y, minlength=fam.n_classes)
            class_counts = bc if class_counts is None else class_counts + bc
        pos += n
        if verbose:
            print(f"  {family}: {pos:,}/{n_samples:,} rows ({time.time() - t0:.1f}s)", file=sys.stderr)
    writer.close()
    seconds = time.time() - t0

    meta = fam.describe()
    meta.update(
        {
            "n_samples": n_samples,
            "n_train": n_train,
            "n_test": test_size,
            "chunk_size": chunk_size,
            "columns": [c for c in table.column_names if c not in ("y", "split", "_p_bayes", "_y_clean")],
            "target": "y",
            "bayes_column": "_p_bayes" if fam.task == "classification" else "_y_clean",
            "file_bytes": path.stat().st_size,
            "generation_seconds": round(seconds, 2),
            "versions": _versions(),
            "hardware": _hardware(),
            "command": " ".join(sys.argv),
        }
    )
    if fam.task == "classification":
        meta["bayes_accuracy_test"] = bayes_sum / max(1, n_test_written)
        meta["train_class_counts"] = class_counts.tolist() if class_counts is not None else None
    else:
        rmse = (bayes_sq / max(1, n_test_written)) ** 0.5
        meta["bayes_rmse_test"] = rmse
        meta["bayes_r2_test"] = 1 - bayes_sq / max(1e-12, y_test_sq)
    with open(out / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=_json_default)
    return meta


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-samples", type=lambda s: int(s.replace("_", "")), default=1_000_000)
    p.add_argument("--test-size", type=lambda s: int(s.replace("_", "")), default=100_000)
    p.add_argument("--chunk-size", type=lambda s: int(s.replace("_", "")), default=1_000_000)
    p.add_argument("--n-features", type=int, default=30)
    p.add_argument("--difficulty", type=float, default=0.5)
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--n-classes", type=int, default=2)
    p.add_argument("--label-noise", type=float, default=0.3, help="std of Gaussian noise added to the unit-variance latent")
    p.add_argument("--imbalance", type=float, default=0.0, help="class proportions ~ (c+1)^-imbalance; 0 = balanced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=8.0, help="refuse datasets above this in-memory size")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    kwargs = dict(n_features=a.n_features, difficulty=a.difficulty, task=a.task, label_noise=a.label_noise,
                  imbalance=a.imbalance, seed=a.seed)
    if a.family != "many_classes":
        kwargs["n_classes"] = a.n_classes
    meta = generate(a.family, a.out, a.n_samples, a.test_size, a.chunk_size, a.max_gb, not a.quiet, **kwargs)
    print(json.dumps({k: meta[k] for k in ("family", "n_samples", "n_features", "difficulty", "knobs", "generation_seconds",
                                            *(k for k in ("bayes_accuracy_test", "bayes_rmse_test") if k in meta))},
                     indent=2, default=_json_default))


if __name__ == "__main__":
    main()
