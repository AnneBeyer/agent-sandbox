"""Dataset families that are deliberately far from the TabICL pre-training prior.

Background
----------
TabICLv2 (and TabPFN) are pre-trained on synthetic tables drawn from
MLP/tree/graph structural causal models.  The released ``tabicl`` package
ships that prior (``tabicl.prior``); its defaults tell us what the model has
seen during pre-training:

* 2 to 100 features, 400 to about 60 000 rows per table, at most 10 classes;
* categorical features with mostly 2 to 9 levels (at most 200);
* IID rows: no time order, no entities/groups shared between rows;
* target functions from random graph SCMs with eight function types
  (MLP, GP, tree ensembles, discretisation, ...) spanning several
  smoothness levels; noise enters through unobserved node dimensions;
* features are standardised and clipped at 4 standard deviations at inference.

BeyondArena ("Beyond IID: How General Are Tabular Foundation Models,
Really?", arXiv:2606.30410) reports that tabular foundation models lose to
tuned gradient-boosted trees and RealMLP exactly where real data leaves that
prior: tables with >= 100k rows, high-dimensional tables, temporal splits,
grouped (unseen-entity) splits, high-cardinality categoricals and text.

Each family below pushes one of these axes with a single ``difficulty`` knob
in [0, 1].  ``difficulty=0`` is close to the prior; ``difficulty=1`` is far
from it.  Every family exposes the *Bayes-optimal* reference for each row
(``_p_bayes`` for classification: the largest class posterior;
``_y_clean`` for regression: E[y | x]) so that learning curves can be read
against the irreducible error instead of against each other only.

All families are stationary "worlds": the random parameters are drawn once
in ``__init__`` from ``seed`` and ``sample(n, start)`` can be called
repeatedly to produce independent chunks of the same distribution, so that
tables with 10M rows can be written without holding them in memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised

CAL_N = 50_000  # calibration sample used to standardise latents / set thresholds


@dataclass
class Sample:
    """One chunk of a dataset.

    X: column name -> array (float32 for numeric, int32 codes for categorical)
    y: target (int32 class codes or float32)
    bayes: per-row Bayes reference (max posterior or E[y|x])
    """

    X: Dict[str, np.ndarray]
    y: np.ndarray
    bayes: np.ndarray
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


def _lerp(a: float, b: float, d: float) -> float:
    return a + (b - a) * float(np.clip(d, 0.0, 1.0))


def _zipf_probs(k: int, exponent: float) -> np.ndarray:
    p = 1.0 / np.arange(1, k + 1, dtype=np.float64) ** exponent
    return p / p.sum()


class Family:
    """Base class. Subclasses implement ``_latent`` (latent-threshold scheme)
    or override ``sample`` entirely."""

    name = "base"
    #: columns that are categorical (int codes) in the output
    categorical: List[str] = ()
    #: how the dataset should be split: "random", "temporal" or "grouped"
    split_kind = "random"
    #: which axis of the prior this family leaves
    axis = ""

    def __init__(
        self,
        n_features: int = 30,
        difficulty: float = 0.5,
        task: str = "classification",
        n_classes: int = 2,
        label_noise: float = 0.3,
        imbalance: float = 0.0,
        seed: int = 0,
        n_total: Optional[int] = None,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.n_features = int(n_features)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.task = task
        self.n_classes = int(n_classes) if task == "classification" else 0
        self.label_noise = float(label_noise)
        self.imbalance = float(imbalance)
        self.seed = int(seed)
        self.n_total = n_total  # needed by temporal families
        self.world = np.random.default_rng([self.seed, 7919])
        self._build_world()
        self._calibrate()

    # -- to be implemented by subclasses -----------------------------------
    def _build_world(self) -> None:
        raise NotImplementedError

    def _latent(self, rng: np.random.Generator, n: int, start: int):
        """Return (X dict, z latent (n,), noise_extra (n,) or None, extra dict).

        ``noise_extra`` is additional *irreducible* noise that is not iid
        Gaussian (e.g. unobserved group effects).  Its per-row std must be
        given by ``self._extra_noise_std`` so that the Bayes reference can
        account for it.
        """
        raise NotImplementedError

    _extra_noise_std = 0.0

    # -- shared machinery ---------------------------------------------------
    _calibrating = False

    def _calibrate(self) -> None:
        rng = np.random.default_rng([self.seed, 104729])
        self._calibrating = True
        try:
            _, z, _, _ = self._latent(rng, CAL_N, 0)
        finally:
            self._calibrating = False
        self.z_mean = float(np.mean(z))
        self.z_std = float(np.std(z) + 1e-12)
        z = (z - self.z_mean) / self.z_std
        self.sigma = math.sqrt(self.label_noise**2 + self._extra_noise_std**2) + 1e-9
        if self.task == "classification":
            noisy = z + self.sigma * rng.standard_normal(z.shape)
            props = self.class_proportions()
            cum = np.cumsum(props)[:-1]
            self.thresholds = np.quantile(noisy, cum)
        else:
            self.thresholds = None

    def class_proportions(self) -> np.ndarray:
        c = np.arange(1, self.n_classes + 1, dtype=np.float64)
        p = c ** (-self.imbalance)
        return p / p.sum()

    def _finalize(self, rng, z, noise_extra):
        z = (z - self.z_mean) / self.z_std
        eps = self.label_noise * rng.standard_normal(z.shape)
        if noise_extra is not None:
            eps = eps + noise_extra / self.z_std
        if self.task == "regression":
            y = (z + eps).astype(np.float32)
            return y, z.astype(np.float32)
        noisy = z + eps
        y = np.searchsorted(self.thresholds, noisy, side="right").astype(np.int32)
        # exact class posteriors under Gaussian noise with std self.sigma
        edges = np.concatenate([[-np.inf], self.thresholds, [np.inf]])
        cdf = ndtr((edges[None, :] - z[:, None]) / self.sigma)
        post = np.diff(cdf, axis=1)
        return y, post.max(axis=1).astype(np.float32)

    def sample(self, n: int, start: int = 0) -> Sample:
        rng = np.random.default_rng([self.seed, 15485863, int(start)])
        X, z, noise_extra, extra = self._latent(rng, n, start)
        y, bayes = self._finalize(rng, z, noise_extra)
        return Sample(X=X, y=y, bayes=bayes, extra=extra)

    def describe(self) -> dict:
        return {
            "family": self.name,
            "axis": self.axis,
            "split_kind": self.split_kind,
            "n_features": self.n_features,
            "difficulty": self.difficulty,
            "task": self.task,
            "n_classes": self.n_classes,
            "label_noise": self.label_noise,
            "imbalance": self.imbalance,
            "seed": self.seed,
            "categorical": list(self.categorical),
            "knobs": self.knobs(),
        }

    def knobs(self) -> dict:
        """Family-specific derived parameters, for the metadata file."""
        return {}


# --------------------------------------------------------------------------
# helpers shared by several families


def _noise_block(rng, n, k, dtype=np.float32):
    return rng.standard_normal((n, k)).astype(dtype)


def _random_mlp(world, d_in, hidden, d_out, n_layers=2, scale=1.0):
    """Weights of a small random MLP (same spirit as the TabICL MLP-SCM)."""
    dims = [d_in] + [hidden] * (n_layers - 1) + [d_out]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append((world.standard_normal((a, b)) * scale / math.sqrt(a), world.standard_normal(b) * 0.1))
    return layers


def _apply_mlp(X, layers):
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.tanh(h)
    return h


# --------------------------------------------------------------------------


class PriorLike(Family):
    """Control family: close to the TabICL prior.

    Standard-normal features, a shallow random tanh-MLP latent, a few
    low-cardinality categoricals and <= 10 classes.  ``difficulty`` only
    scales the label noise (0 -> 0.1, 1 -> 1.0).  TabICL is expected to do
    well here; this family calibrates what "fine-tuning helps" means.
    """

    name = "prior_like"
    axis = "none (in-prior control)"

    def _build_world(self):
        w = self.world
        self.d_num = max(2, self.n_features - 3)
        self.n_cat = self.n_features - self.d_num
        self.cat_levels = [int(w.integers(2, 8)) for _ in range(self.n_cat)]
        self.categorical = [f"cat_{i}" for i in range(self.n_cat)]
        self.layers = _random_mlp(w, self.d_num + self.n_cat, 32, 1, n_layers=3, scale=1.5)
        self.cat_embed = [w.standard_normal(k) for k in self.cat_levels]
        self.label_noise = _lerp(0.1, 1.0, self.difficulty)

    def _latent(self, rng, n, start):
        Xn = rng.standard_normal((n, self.d_num))
        cats = [rng.integers(0, k, n) for k in self.cat_levels]
        emb = np.stack([e[c] for e, c in zip(self.cat_embed, cats)], axis=1) if cats else np.zeros((n, 0))
        z = _apply_mlp(np.concatenate([Xn, emb], axis=1), self.layers)[:, 0]
        X = {f"num_{j}": Xn[:, j].astype(np.float32) for j in range(self.d_num)}
        for i, c in enumerate(cats):
            X[f"cat_{i}"] = c.astype(np.int32)
        return X, z, None, {}

    def knobs(self):
        return {"n_numeric": self.d_num, "cat_levels": self.cat_levels, "effective_label_noise": self.label_noise}


class DeepInteraction(Family):
    """Large-data regime: a functional-ANOVA target with interaction terms of
    increasing order.  Each of the k interacting features is cut into b
    equiprobable bins; the latent is a weighted sum of random lookup tables
    over single features (order 1), over pairs (order 2), triples, ... up to
    order k.  A learner resolves order-o terms once it has seen enough rows
    per b**o cell, so accuracy keeps improving with data for a long time
    (marginals with 1k rows, pairs with 10k, triples with 100k, ...), the
    way gradient-boosted trees improve on large real tables.  An in-context
    learner with a bounded context cannot use the extra rows.

    difficulty 0 -> k=2, b=3, order weights decay fast (almost additive);
    difficulty 1 -> k=6, b=6, the high-order terms carry most of the
    variance (46 656 cells in the top-order table).
    """

    name = "deep_interaction"
    axis = "large data / high-order interactions"

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.k = min(self.n_features, 2 + int(round(4 * d)))
        self.bins = 3 + int(round(3 * d))
        self.order_decay = _lerp(0.35, 1.6, d)  # weight of order o is decay**(o-1)
        self.terms_per_order = 3
        self.edges = np.array([math.sqrt(2) * _erfinv(2 * q - 1) for q in np.linspace(0, 1, self.bins + 1)[1:-1]])
        self.terms = []  # (order, feature index tuple, table, weight)
        for order in range(1, self.k + 1):
            if order == 1:
                subsets = [(j,) for j in range(self.k)]
            elif order == self.k:
                subsets = [tuple(range(self.k))]
            else:
                subsets = []
                while len(subsets) < self.terms_per_order:
                    cand = tuple(sorted(w.choice(self.k, order, replace=False).tolist()))
                    if cand not in subsets:
                        subsets.append(cand)
            weight = self.order_decay ** (order - 1) / math.sqrt(len(subsets))
            for sub in subsets:
                self.terms.append((order, np.array(sub), w.standard_normal(self.bins**order).astype(np.float32), weight))
        self.n_cells_top = self.bins**self.k
        self.beta = w.standard_normal(self.n_features) / math.sqrt(self.n_features)

    def _latent(self, rng, n, start):
        Xn = rng.standard_normal((n, self.n_features))
        binned = np.searchsorted(self.edges, Xn[:, : self.k])  # (n, k) in [0, bins)
        z = 0.2 * np.tanh(Xn @ self.beta)  # weak smooth part over all features
        for order, sub, table, weight in self.terms:
            cell = np.zeros(n, dtype=np.int64)
            for i, j in enumerate(sub):
                cell += binned[:, j] * self.bins**i
            z = z + weight * table[cell]
        X = {f"num_{j}": Xn[:, j].astype(np.float32) for j in range(self.n_features)}
        return X, z, None, {}

    def knobs(self):
        var = {}
        for order, sub, table, weight in self.terms:
            var[order] = var.get(order, 0.0) + weight**2
        tot = sum(var.values())
        return {"k_interacting": self.k, "bins": self.bins, "order_decay": self.order_decay,
                "n_cells_top_order": int(self.n_cells_top),
                "variance_share_by_order": {int(o): round(v / tot, 3) for o, v in sorted(var.items())}}


def _erfinv(x):
    from scipy.special import erfinv

    return float(erfinv(x))


class HighCardinality(Family):
    """High-cardinality categoricals with level-specific effects (the
    "target encoding" regime).  m categorical columns with Zipf-distributed
    level frequencies, each level carrying its own random effect, plus a
    pairwise interaction between the two first columns whose effect is only
    defined on the (hashed) pair.  Rare levels are frequent in the tail, so
    the test set contains levels unseen or barely seen in training.

    difficulty 0 -> 3 columns with 10 levels (inside the prior);
    difficulty 1 -> 6 columns with 10 000 levels and strong pair effects.
    """

    name = "high_cardinality"
    axis = "high-cardinality categoricals"

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.n_cat = 3 + int(round(3 * d))
        self.cardinality = int(round(10 ** _lerp(1.0, 4.0, d)))
        self.zipf_exp = 1.1
        self.pair_weight = _lerp(0.0, 1.0, d)
        self.n_num = max(0, self.n_features - self.n_cat)
        self.categorical = [f"cat_{i}" for i in range(self.n_cat)]
        self.probs = _zipf_probs(self.cardinality, self.zipf_exp)
        self.effects = [w.standard_normal(self.cardinality).astype(np.float32) for _ in range(self.n_cat)]
        self.n_pair_buckets = 1 << 20
        self.pair_effects = w.standard_normal(self.n_pair_buckets).astype(np.float32)
        self.pair_salt = int(w.integers(1, 2**31 - 1))
        self.beta = w.standard_normal(self.n_num) / math.sqrt(max(1, self.n_num))
        self.cdf = np.cumsum(self.probs)

    def _draw_levels(self, rng, n):
        u = rng.random(n)
        return np.minimum(np.searchsorted(self.cdf, u), self.cardinality - 1).astype(np.int32)

    def _latent(self, rng, n, start):
        cats = [self._draw_levels(rng, n) for _ in range(self.n_cat)]
        z = sum(e[c] for e, c in zip(self.effects, cats)) / math.sqrt(self.n_cat)
        if self.pair_weight > 0:
            h = (cats[0].astype(np.int64) * 1_000_003 + cats[1].astype(np.int64) * 7919 + self.pair_salt) % self.n_pair_buckets
            z = z + self.pair_weight * self.pair_effects[h]
        X = {}
        if self.n_num:
            Xn = rng.standard_normal((n, self.n_num))
            z = z + np.tanh(Xn @ self.beta)
            X.update({f"num_{j}": Xn[:, j].astype(np.float32) for j in range(self.n_num)})
        for i, c in enumerate(cats):
            X[f"cat_{i}"] = c
        return X, z, None, {}

    def knobs(self):
        return {"n_categorical": self.n_cat, "cardinality": self.cardinality, "zipf_exponent": self.zipf_exp, "pair_weight": self.pair_weight}


class ManyClasses(Family):
    """Many classes (the prior has at most 10).  Gaussian class clusters in
    an 8-dimensional latent space pushed through an injective warp into the
    observed numeric features, plus noise features.  Because the warp is
    injective the exact Bayes posterior is available.

    difficulty 0 -> 10 classes; difficulty 1 -> 500 classes.  ``n_classes``
    is ignored for this family.  Within-class spread is set so that the
    Bayes accuracy stays roughly between 0.6 and 0.8.
    """

    name = "many_classes"
    axis = "many classes"

    def __init__(self, **kw):
        kw.setdefault("task", "classification")
        if kw["task"] != "classification":
            raise ValueError("many_classes is classification-only")
        super().__init__(**kw)

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.n_classes = int(round(10 ** _lerp(1.0, math.log10(500), d)))
        self.d_lat = 8
        self.centers = w.standard_normal((self.n_classes, self.d_lat))
        self.props = self.class_proportions()
        self.target_bayes_acc = 0.75
        self.spread = 1.0  # set by _calibrate
        self.class_scale = np.exp(0.3 * w.standard_normal(self.n_classes))  # per-class isotropic std factor
        self.d_obs = min(max(self.d_lat, 20), self.n_features)
        q, _ = np.linalg.qr(w.standard_normal((self.d_obs, self.d_obs)))
        self.rot = q[:, : self.d_lat]  # (d_obs, d_lat) orthonormal columns
        q2, _ = np.linalg.qr(w.standard_normal((self.d_obs, self.d_obs)))
        self.rot2 = q2
        self.n_noise = self.n_features - self.d_obs
        # elementwise monotone but wiggly warp x + c*sin(w x) with c*w < 1
        self.warp_w = w.uniform(2.0, 4.0, self.d_obs)
        self.warp_c = 0.95 / self.warp_w

    def _calibrate(self):
        """Bisect the within-class spread so that the Bayes accuracy on a
        calibration sample matches ``target_bayes_acc`` (default 0.75)."""
        lo, hi = 0.02, 5.0
        for _ in range(25):
            self.spread = math.sqrt(lo * hi)
            acc = float(self._posterior(np.random.default_rng([self.seed, 104729]), 20_000)[1].mean())
            if acc > self.target_bayes_acc:
                lo = self.spread
            else:
                hi = self.spread
        self.spread = math.sqrt(lo * hi)

    def _posterior(self, rng, n):
        y = rng.choice(self.n_classes, size=n, p=self.props).astype(np.int32)
        sd = self.spread * self.class_scale  # (C,)
        u = self.centers[y] + sd[y, None] * rng.standard_normal((n, self.d_lat))
        logp = np.empty((n, self.n_classes), dtype=np.float64)
        uu = (u**2).sum(1)[:, None]
        cc = (self.centers**2).sum(1)[None, :]
        for s in range(0, n, 20_000):
            e = slice(s, min(n, s + 20_000))
            logp[e] = -(uu[e] + cc - 2 * u[e] @ self.centers.T) / (2 * sd[None, :] ** 2)
        logp += (np.log(self.props) - self.d_lat * np.log(sd))[None, :]
        logp -= logp.max(1, keepdims=True)
        post = np.exp(logp)
        post /= post.sum(1, keepdims=True)
        return y, post.max(1).astype(np.float32), u

    def sample(self, n, start=0):
        rng = np.random.default_rng([self.seed, 15485863, int(start)])
        y, bayes, u = self._posterior(rng, n)
        obs = u @ self.rot.T  # (n, d_obs), injective
        obs = obs + self.warp_c * np.sin(self.warp_w * obs)  # monotone elementwise warp
        obs = obs @ self.rot2.T
        X = {f"num_{j}": obs[:, j].astype(np.float32) for j in range(self.d_obs)}
        if self.n_noise > 0:
            Xn = _noise_block(rng, n, self.n_noise)
            X.update({f"noise_{j}": Xn[:, j] for j in range(self.n_noise)})
        return Sample(X=X, y=y, bayes=bayes)

    def knobs(self):
        return {"n_classes": self.n_classes, "latent_dim": self.d_lat, "spread": self.spread, "n_noise_features": self.n_noise,
                "class_scale_range": [float(self.class_scale.min()), float(self.class_scale.max())]}


class HighDimensional(Family):
    """Wide tables with sparse signal and heavy-tailed marginals.  Only 20
    features are informative (through random ReLU units over pairs and
    products); half of the remaining columns are pure noise and half are
    low-rank "confounders" correlated with the informative ones.  Marginals
    are Student-t.

    difficulty 0 -> 100 columns, t(10); difficulty 1 -> ~3000 columns, t(2).
    ``n_features`` is ignored (set from difficulty).  Memory scales with the
    width: 10M x 3000 float32 is 120 GB, so cap the row count accordingly.
    """

    name = "high_dimensional"
    axis = "high dimensionality / heavy tails"

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.n_features = int(round(10 ** _lerp(2.0, 3.5, d)))
        self.n_inf = 20
        self.df = _lerp(10.0, 2.0, d)
        rest = self.n_features - self.n_inf
        self.n_conf = rest // 2
        self.n_noise = rest - self.n_conf
        self.units = 16
        self.pairs = w.integers(0, self.n_inf, (self.units, 2))
        self.unit_w = w.standard_normal((self.units, 2))
        self.unit_b = w.standard_normal(self.units) * 0.5
        self.unit_out = w.standard_normal(self.units) / math.sqrt(self.units)
        self.prod_pairs = w.integers(0, self.n_inf, (4, 2))
        self.conf_mix = w.standard_normal((self.n_inf, self.n_conf)) / math.sqrt(self.n_inf)
        perm = w.permutation(self.n_features)
        self.perm = perm  # hide which columns are informative

    def _t(self, rng, shape):
        return (rng.standard_t(self.df, shape) / math.sqrt(self.df / (self.df - 2)) if self.df > 2 else rng.standard_t(self.df, shape)).astype(np.float32)

    def _latent(self, rng, n, start):
        Xi = self._t(rng, (n, self.n_inf))
        g = np.tanh(Xi / 2)  # bounded view used by the latent (heavy tails act as outliers)
        h = g[:, self.pairs[:, 0]] * self.unit_w[:, 0] + g[:, self.pairs[:, 1]] * self.unit_w[:, 1] + self.unit_b
        z = np.maximum(h, 0) @ self.unit_out
        for a, b in self.prod_pairs:
            z = z + g[:, a] * g[:, b] * 0.5
        conf = (g @ self.conf_mix) + 0.7 * self._t(rng, (n, self.n_conf))
        noise = self._t(rng, (n, self.n_noise))
        full = np.concatenate([Xi, conf.astype(np.float32), noise], axis=1)[:, self.perm]
        X = {f"f_{j}": full[:, j] for j in range(self.n_features)}
        return X, z, None, {}

    def knobs(self):
        inv = np.argsort(self.perm)
        return {"n_informative": self.n_inf, "n_confounder": self.n_conf, "n_noise": self.n_noise, "student_t_df": self.df,
                "informative_columns": [f"f_{int(inv[i])}" for i in range(self.n_inf)]}


class TemporalDrift(Family):
    """Time-ordered rows with covariate drift, concept drift and seasonality.
    Rows carry a ``time`` feature in [0, 1].  Feature means drift along
    smooth curves, the coefficient vector rotates with time (concept drift),
    and the latent has a seasonal term.  The split is temporal: the last
    rows are the test set, so test rows live in a regime never seen.

    difficulty 0 -> no drift (IID); difficulty 1 -> feature means move by 2
    std, coefficients rotate by 90 degrees over the table, strong seasons.
    Requires ``n_total`` (set by the generator) to map rows to time.
    """

    name = "temporal_drift"
    axis = "temporal shift"
    split_kind = "temporal"

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.drift_amp = _lerp(0.0, 2.0, d)
        self.rot_angle = _lerp(0.0, math.pi / 2, d)
        self.season_w = _lerp(0.0, 1.0, d)
        self.n_seasons = 6
        self.mean_freq = w.uniform(0.5, 1.5, self.n_features)
        self.mean_phase = w.uniform(0, 2 * math.pi, self.n_features)
        self.beta0 = w.standard_normal(self.n_features) / math.sqrt(self.n_features)
        b1 = w.standard_normal(self.n_features)
        b1 -= b1 @ self.beta0 * self.beta0 / (self.beta0 @ self.beta0)
        self.beta1 = b1 / np.linalg.norm(b1) * np.linalg.norm(self.beta0)
        self.inter = w.integers(0, self.n_features, (4, 2))

    def _time(self, n, start):
        if self._calibrating:  # cover the whole time range when setting thresholds
            return np.linspace(0, 1, n)
        total = self.n_total or max(n, CAL_N)
        return (np.arange(start, start + n) / max(1, total - 1)).clip(0, 1)

    def _latent(self, rng, n, start):
        t = self._time(n, start)
        means = self.drift_amp * np.sin(2 * math.pi * self.mean_freq[None, :] * t[:, None] + self.mean_phase[None, :])
        Xn = rng.standard_normal((n, self.n_features)) + means
        ang = self.rot_angle * t
        beta_t = np.cos(ang)[:, None] * self.beta0[None, :] + np.sin(ang)[:, None] * self.beta1[None, :]
        z = np.tanh((Xn * beta_t).sum(1) * 1.5)
        for a, b in self.inter:
            z = z + 0.3 * Xn[:, a] * Xn[:, b] / 2
        z = z + self.season_w * np.sin(2 * math.pi * self.n_seasons * t)
        X = {"time": t.astype(np.float32)}
        X.update({f"num_{j}": Xn[:, j].astype(np.float32) for j in range(self.n_features)})
        return X, z, None, {}

    def knobs(self):
        return {"drift_amplitude_std": self.drift_amp, "coef_rotation_rad": self.rot_angle, "season_weight": self.season_w, "n_seasons": self.n_seasons}


class GroupedEntities(Family):
    """Hierarchical data: every row belongs to an entity (group).  Entities
    have observed covariates (shared by their rows) and an *unobserved*
    random effect.  Rows interact individual features with the entity
    covariates.  The split is grouped: test rows come from entities that
    never appear in training, so memorising the ``group_id`` column (which
    is given as a high-cardinality categorical) does not transfer.

    difficulty 0 -> 100 training entities, no unobserved effect (the group id
    is a harmless distractor);  difficulty 1 -> 10 000 entities, unobserved
    effect std 1.0 (an irreducible, entity-level noise that in-sample models
    can memorise through the id column but that does not transfer).
    The Bayes reference is the *unseen-entity* oracle (it does not know the
    random effect).
    """

    name = "grouped_entities"
    axis = "grouped / unseen-entity shift"
    split_kind = "grouped"
    categorical = ["group_id"]

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.n_groups = int(round(10 ** _lerp(2.0, 4.0, d)))
        self.n_test_groups = max(10, self.n_groups // 5)
        self.n_all_groups = self.n_groups + self.n_test_groups
        self.group_sd = _lerp(0.0, 1.0, d)
        self._extra_noise_std = self.group_sd
        self.d_grp = 3
        self.d_ind = max(2, self.n_features - self.d_grp)
        self.grp_cov = w.standard_normal((self.n_all_groups, self.d_grp))
        self.grp_eff = (w.standard_normal(self.n_all_groups) * self.group_sd).astype(np.float64)
        self.probs = _zipf_probs(self.n_groups, 1.0)
        self.cdf = np.cumsum(self.probs)
        self.layers = _random_mlp(w, self.d_ind + self.d_grp + self.d_ind * self.d_grp, 32, 1, n_layers=2, scale=1.5)
        self.test_mode = False

    def _draw_groups(self, rng, n):
        if self.test_mode:
            return self.n_groups + rng.integers(0, self.n_test_groups, n)
        return np.minimum(np.searchsorted(self.cdf, rng.random(n)), self.n_groups - 1)

    def _latent(self, rng, n, start):
        g = self._draw_groups(rng, n)
        Xi = rng.standard_normal((n, self.d_ind))
        Zg = self.grp_cov[g]
        inter = (Xi[:, :, None] * Zg[:, None, :]).reshape(n, -1)
        z = _apply_mlp(np.concatenate([Xi, Zg, inter], axis=1), self.layers)[:, 0]
        X = {"group_id": g.astype(np.int32)}
        X.update({f"grp_{j}": Zg[:, j].astype(np.float32) for j in range(self.d_grp)})
        X.update({f"num_{j}": Xi[:, j].astype(np.float32) for j in range(self.d_ind)})
        return X, z, self.grp_eff[g], {}

    def sample_test(self, n, start=0):
        self.test_mode = True
        try:
            return self.sample(n, start)
        finally:
            self.test_mode = False

    def knobs(self):
        return {"n_train_groups": self.n_groups, "n_test_groups": self.n_test_groups, "group_effect_std": self.group_sd, "d_group_cov": self.d_grp}


class Irregular(Family):
    """Irregular, high-frequency target functions (Grinsztajn et al. 2022
    style): sums of high-frequency sinusoids of single features plus a
    parity (XOR) of sign bits over several features.  Smooth priors
    interpolate badly; trees need many splits; MLPs need many rows.

    difficulty 0 -> frequency ~0.5 cycles per std, parity of 2 bits with
    weight 0; difficulty 1 -> ~8 cycles per std, parity of 6 bits, weight 1.
    """

    name = "irregular"
    axis = "irregular target functions"

    def _build_world(self):
        d = self.difficulty
        w = self.world
        self.k_sin = min(self.n_features, 8)
        self.omega = 2 * math.pi * _lerp(0.5, 8.0, d) * w.uniform(0.7, 1.3, self.k_sin)
        self.phase = w.uniform(0, 2 * math.pi, self.k_sin)
        self.amp = w.uniform(0.5, 1.0, self.k_sin)
        self.k_par = min(self.n_features, 2 + int(round(4 * d)))
        self.par_weight = _lerp(0.0, 1.0, d)
        self.par_omega = 2 * math.pi * _lerp(0.25, 2.0, d)

    def _latent(self, rng, n, start):
        Xn = rng.standard_normal((n, self.n_features))
        z = (self.amp * np.sin(self.omega * Xn[:, : self.k_sin] + self.phase)).sum(1) / math.sqrt(self.k_sin)
        bits = (np.sin(self.par_omega * Xn[:, : self.k_par]) > 0).astype(np.int32)
        parity = bits.sum(1) % 2
        z = z + self.par_weight * (2.0 * parity - 1.0)
        X = {f"num_{j}": Xn[:, j].astype(np.float32) for j in range(self.n_features)}
        return X, z, None, {}

    def knobs(self):
        return {"n_sinusoids": self.k_sin, "cycles_per_std": [float(o / (2 * math.pi)) for o in self.omega],
                "parity_bits": self.k_par, "parity_weight": self.par_weight}


FAMILIES = {
    cls.name: cls
    for cls in (PriorLike, DeepInteraction, HighCardinality, ManyClasses, HighDimensional, TemporalDrift, GroupedEntities, Irregular)
}


def make_family(name: str, **kwargs) -> Family:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}")
    return FAMILIES[name](**kwargs)
