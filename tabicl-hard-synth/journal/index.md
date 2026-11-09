# tabicl-hard-synth journal

Synthetic tabular datasets that leave the TabICL pre-training prior along
the axes where BeyondArena finds tabular foundation models losing to
gradient-boosted trees and MLPs. Purpose: a controlled test bed for deciding
when fine-tuning TabICL is worth its cost versus falling back to LightGBM or
an MLP.

| date | iteration | summary |
|---|---|---|
| 2026-11-09 | [Generator and CPU baselines](2026-11-09-generator-baselines/report.md) | Eight dataset families with a difficulty knob and exact Bayes references; LightGBM / MLP / linear learning curves on 300k-row versions (plus a 1M-row deep-interaction curve) and zero-shot TabICL on CPU at 1k and 10k context; 10M rows generate in 25 s. TabICL is the most sample-efficient model at 10k rows on 9 of 14 datasets, but a baseline with 100k rows beats it on 15 of 16. |
