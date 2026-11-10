"""Generate (and optionally evaluate) a grid of datasets described in YAML.

    python -m hardsynth.suite --config suite.yaml --grid smoke --data-dir data
    python -m hardsynth.suite --config suite.yaml --grid smoke --data-dir data \
        --evaluate --models lgbm mlp linear --train-sizes 1000 10000 100000 --out journal/<iter>

Each dataset entry in the YAML becomes ``<data-dir>/<name>``.  Entries take the
same keys as ``hardsynth.generate`` (family, difficulty, n_samples, ...);
``defaults`` in the grid fills missing keys.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .evaluate import evaluate as run_evaluate
from .generate import generate


def dataset_specs(config: dict, grid: str):
    g = config["grids"][grid]
    defaults = dict(config.get("defaults", {}))
    defaults.update(g.get("defaults", {}))
    for entry in g["datasets"]:
        spec = dict(defaults)
        spec.update(entry)
        name = spec.pop("name", None) or f"{spec['family']}_d{spec['difficulty']}"
        yield name, spec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parent.parent / "suite.yaml")
    p.add_argument("--grid", default="smoke")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--only", nargs="*", default=None, help="dataset names to include")
    p.add_argument("--n-samples", type=lambda s: int(s.replace("_", "")), default=None, help="override rows for every dataset")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="evaluation output directory (one subdir per dataset)")
    p.add_argument("--models", nargs="+", default=["lgbm", "mlp", "linear"])
    p.add_argument("--train-sizes", nargs="+", type=lambda s: int(s.replace("_", "")), default=[1000, 10_000, 100_000])
    p.add_argument("--test-rows", type=int, default=100_000)
    p.add_argument("--mlp-max-rows", type=int, default=300_000)
    p.add_argument("--tabicl-max-context", type=int, default=100_000)
    p.add_argument("--tabicl-n-estimators", type=int, default=8)
    p.add_argument("--tabicl-batch-size", type=int, default=8)
    p.add_argument("--tabicl-predict-chunk", type=int, default=5000)
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gb", type=float, default=8.0)
    p.add_argument("--drop-columns", nargs="*", default=None)
    p.add_argument("--append", action="store_true")
    a = p.parse_args(argv)

    config = yaml.safe_load(a.config.read_text())
    for name, spec in dataset_specs(config, a.grid):
        if a.only and name not in a.only:
            continue
        out = a.data_dir / name
        n_samples = a.n_samples or spec.pop("n_samples", 1_000_000)
        spec.pop("n_samples", None)
        test_size = spec.pop("test_size", min(100_000, n_samples // 10))
        chunk = spec.pop("chunk_size", 1_000_000)
        if a.skip_existing and (out / "metadata.json").exists():
            print(f"[{name}] exists, skipping generation", file=sys.stderr)
        else:
            t0 = time.time()
            meta = generate(spec.pop("family"), out, n_samples, test_size, chunk, a.max_gb, verbose=False, **spec)
            print(f"[{name}] generated {n_samples:,} rows in {time.time() - t0:.0f}s; "
                  f"bayes={meta.get('bayes_accuracy_test', meta.get('bayes_rmse_test')):.3f}", file=sys.stderr)
        if a.evaluate:
            ev = argparse.Namespace(data=out, out=(a.out or a.data_dir / "eval") / name, models=a.models,
                                    train_sizes=a.train_sizes, test_rows=a.test_rows, mlp_max_rows=a.mlp_max_rows,
                                    tabicl_max_context=a.tabicl_max_context, tabicl_n_estimators=a.tabicl_n_estimators,
                                    tabicl_batch_size=a.tabicl_batch_size, tabicl_predict_chunk=a.tabicl_predict_chunk, ft_epochs=a.ft_epochs, device=a.device,
                                    n_jobs=a.n_jobs, seed=a.seed, drop_columns=a.drop_columns, append=a.append)
            t0 = time.time()
            run_evaluate(ev)
            print(f"[{name}] evaluated in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
