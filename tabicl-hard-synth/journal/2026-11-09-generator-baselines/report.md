# Iteration 1: a generator for tables outside the TabICL prior, with CPU baselines

Date: 2026-11-09. Code: `tabicl-hard-synth/hardsynth/`. Grid: `suite.yaml`, `smoke`.

## Goal

Build synthetic tables (up to 10M rows) on which TabICL is expected to lose
to gradient-boosted trees or an MLP, so that the next iterations can ask:
at which data size, and how far from the pre-training prior, does
fine-tuning TabICL pay for itself instead of falling back to LightGBM or an
MLP?

## What "outside the prior" means here

The released `tabicl` package (2.2.0) ships the pre-training prior
(`tabicl.prior`). Reading its defaults and the TabICLv2 training recipe gives
a concrete picture of the data the model has seen:

| prior property | TabICLv2 pre-training | where real data leaves it (BeyondArena) |
|---|---|---|
| rows per table | 400 to about 60 000 (log-uniform in the last training stages) | tables with >= 100k rows, up to 1M |
| columns | 2 to 100 | up to 22k |
| classes | at most 10 | many-class tasks |
| categorical levels | mostly 2 to 9, at most 200 | high-cardinality ids, free text |
| row structure | IID rows | temporal splits, grouped (unseen-entity) splits |
| target functions | random MLP/tree/graph SCMs, smooth, noise std <= 0.3 | irregular, high-frequency, deep interactions |
| feature scale | standardised, clipped at 4 std at inference | heavy tails, outliers |

BeyondArena (arXiv:2606.30410) reports that TabPFN-2.6 / TabICLv2 / TabDPT
win on tiny-to-medium IID tables but are beaten by tuned XGBoost,
CatBoost, LightGBM and RealMLP on the large (>= 100k rows), high-dimensional,
temporal and grouped subsets. The subsets overlap (large tables are
over-represented among the temporal and high-dimensional ones), so the
paper cannot say which axis causes the loss. A synthetic generator with one
knob per axis can.

## The eight families

Each family has a single `difficulty` knob in [0, 1]. At 0 the family is
close to the prior; at 1 it is far outside. All families share `n_samples`,
`n_features`, `task`, `n_classes`, `label_noise`, `imbalance` and `seed`.

| family | axis | difficulty 0 -> 1 | split |
|---|---|---|---|
| `prior_like` | control, inside the prior | label noise std 0.1 -> 1.0 | random |
| `deep_interaction` | large data / high-order interactions: target is a random lookup table over the joint discretisation of k features | 9 -> 1.7M table cells; table weight 0.5 -> 1.0 | random |
| `high_cardinality` | level-specific effects of Zipf-distributed categoricals plus a hashed pair interaction | 3 columns x 10 levels -> 6 x 10 000 levels; pair weight 0 -> 1 | random |
| `many_classes` | Gaussian class clusters in an 8-d latent space, injective warp to 20 observed features, noise features | 10 -> 500 classes; Bayes accuracy calibrated to 0.75 | random |
| `high_dimensional` | 20 informative columns under 16 random ReLU units and products; half the rest are low-rank confounders, half noise; Student-t marginals | 100 -> about 3 000 columns; t(10) -> t(2) | random |
| `temporal_drift` | feature means drift along sine curves, the coefficient vector rotates with time, a seasonal term; `time` is a feature | no drift -> 2 std mean drift, 90 degree rotation, season weight 1 | last rows |
| `grouped_entities` | rows belong to entities with observed covariates, individual x entity interactions and an unobserved entity effect; `group_id` is a feature | 100 -> 10 000 training entities; unobserved effect std 0 -> 1 | unseen entities |
| `irregular` | sums of high-frequency sinusoids plus a parity (XOR) of sign bits | 0.5 -> 8 cycles per std; parity of 2 -> 6 bits, weight 0 -> 1 | random |

Every family exposes the Bayes-optimal reference per row: `_p_bayes`, the
largest class posterior, for classification, and `_y_clean` = E[y | x] for
regression. For the latent-threshold families the posterior is exact under
the Gaussian label noise; for `many_classes` it is exact because the warp
from latent to observed space is injective; for `grouped_entities` it is
the unseen-entity oracle that does not know the entity's random effect.
Learning curves can therefore be read as "distance to the ceiling", not
only as "model A vs model B".

### Why these should be hard for in-context learning but not for trees

* `deep_interaction`: with b^k independent cell values, a learner needs on
  the order of 10 x b^k rows to resolve the table. At difficulty 0.8 there
  are 78 125 cells, so accuracy keeps improving up to about 1M rows. An
  in-context learner sees at most its context (TabICL recommends <= 100k
  rows, the pre-training prior never exceeded 60k); the rows beyond the
  context are simply unused unless the model is fine-tuned.
* `high_cardinality`: TabICL ordinal-encodes categoricals and warns above
  40 levels; the prior has at most 200 levels and typically < 10. LightGBM
  handles the levels natively and the MLP uses target encoding.
* `many_classes`: above 10 classes TabICL falls back to hierarchical
  classification over class subsets.
* `high_dimensional` and `irregular`: the prior's functions are smooth
  tree/MLP/GP mixtures with <= 100 columns; heavy tails are clipped at 4
  std, which destroys the signal in the tails.
* `temporal_drift` and `grouped_entities`: in-context learning weights all
  context rows equally and has no notion of recency or of an entity; trees
  split on `time` and MLPs extrapolate the smooth drift at least locally.
  These two families also test memorisation: the `group_id` column lets a
  model fit entity effects that do not transfer to unseen entities.

## Generation cost

`python -m hardsynth.generate --family deep_interaction --difficulty 0.8
--n-samples 10_000_000 --test-size 200_000 --n-classes 3`: 10M rows x 30
features in 22 s on this 4-core cloud VM (numpy 2.5.3, pyarrow, zstd
Parquet), 1.3 GB on disk, chunks of 1M rows so peak memory stays below
1 GB. The wide family is the exception: 10M x 3 000 float32 would be 120 GB,
so the `full` grid caps it at 0.5M to 1M rows.

<!-- RESULTS -->
