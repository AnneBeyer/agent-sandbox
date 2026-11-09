# tabicl-hard-synth

Synthetic tabular datasets (up to 10M rows) that are deliberately *outside*
the TabICL pre-training prior, as a test bed for the question: **when does
fine-tuning TabICL pay off compared with falling back to gradient-boosted
trees or an MLP?**

The families mirror the regimes in which BeyondArena ("Beyond IID: How
General Are Tabular Foundation Models, Really?", arXiv:2606.30410) finds
tabular foundation models losing to tuned trees and RealMLP: >= 100k rows,
high-dimensional tables, temporal splits, grouped (unseen-entity) splits and
high-cardinality categoricals. Each family has one `difficulty` knob in
[0, 1]; 0 is inside the prior, 1 is far from it. Every row carries the
Bayes-optimal reference (`_p_bayes` or `_y_clean`), so learning curves can
be read against the irreducible error.

| family | leaves the prior through | difficulty 0 -> 1 | split |
|---|---|---|---|
| `prior_like` | nothing (control) | label noise 0.1 -> 1.0 | random |
| `deep_interaction` | high-order interactions, needs many rows | 9 -> 1.7M lookup-table cells | random |
| `high_cardinality` | categoricals with level effects | 3 x 10 -> 6 x 10 000 levels, pair effects | random |
| `many_classes` | number of classes | 10 -> 500 classes (Bayes acc. fixed at 0.75) | random |
| `high_dimensional` | width, sparse signal, heavy tails | 100 -> ~3 000 columns, t(10) -> t(2) | random |
| `temporal_drift` | covariate + concept drift, seasons | none -> 2 std mean drift, 90 deg coefficient rotation | last rows |
| `grouped_entities` | unseen entities, memorisable id | 100 -> 10 000 entities, unobserved effect std 0 -> 1 | unseen groups |
| `irregular` | high-frequency / parity targets | 0.5 -> 8 cycles per std, parity of 2 -> 6 bits | random |

Shared knobs: `n_samples`, `n_features`, `task` (classification/regression),
`n_classes`, `label_noise` (std of Gaussian noise on the unit-variance
latent), `imbalance` (class proportions ~ c^-imbalance), `seed`.

## Usage

```bash
pixi install                     # CPU baselines
pixi run test

# one dataset: chunked Parquet + metadata.json
pixi run generate --family deep_interaction --difficulty 0.8 \
    --n-samples 10_000_000 --test-size 200_000 --n-classes 3 --out data/deep_interaction_d0.8

# learning curves of LightGBM / MLP / logistic regression against the Bayes line
pixi run evaluate --data data/deep_interaction_d0.8 --out results/deep_interaction_d0.8 \
    --models lgbm mlp linear --train-sizes 1000 10000 100000 1000000

# a whole grid (see suite.yaml): smoke = 300k rows, full = 10M rows
pixi run suite --grid smoke --data-dir data --evaluate --out results/smoke

# TabICL zero-shot and fine-tuned (GPU recommended; downloads the Hugging Face checkpoint)
pixi run -e tabicl evaluate --data data/deep_interaction_d0.8 --out results/ft \
    --models lgbm tabicl tabicl-ft --train-sizes 10000 100000 1000000 --tabicl-max-context 100000

# on a CPU with little memory: fewer ensemble members, one per forward pass, chunked prediction
pixi run -e tabicl evaluate --data data/prior_like_d0.3 --out results/cpu --models tabicl \
    --train-sizes 1000 10000 --test-rows 5000 --device cpu \
    --tabicl-n-estimators 4 --tabicl-batch-size 1 --tabicl-predict-chunk 1000 --append
```

`--append` merges new model rows into an existing `results.json`; `--drop-columns group_id`
hides a column from every model (BeyondArena drops the group index for grouped data).

`data/` is git-ignored; 10M rows x 30 features take about 25 s to generate
and 1.3 GB on disk.

Output columns: the features (categoricals as `int32` codes, listed in
`metadata.json`), `y`, `_p_bayes` (classification: max class posterior) or
`_y_clean` (regression: E[y|x]) and `split` (0 train, 1 test).

## Layout

```
hardsynth/families.py   the eight generators
hardsynth/generate.py   CLI, chunked Parquet writer, metadata
hardsynth/evaluate.py   learning curves (LightGBM, MLP, linear, TabICL, TabICL fine-tuned)
hardsynth/suite.py      run a YAML grid
hardsynth/summarize.py  results.json -> Markdown table
suite.yaml              smoke and full grids
tests/                  unit tests (pixi run test)
journal/                experiment reports
```
