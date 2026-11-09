# tabicl-hard-synth journal

Synthetic tabular datasets that leave the TabICL pre-training prior along
the axes where BeyondArena finds tabular foundation models losing to
gradient-boosted trees and MLPs. Purpose: a controlled test bed for deciding
when fine-tuning TabICL is worth its cost versus falling back to LightGBM or
an MLP.

| date | iteration | summary |
|---|---|---|
| 2026-11-09 | [Generator and CPU baselines](2026-11-09-generator-baselines/report.md) | Eight dataset families with a difficulty knob and exact Bayes references; LightGBM / MLP / linear learning curves on 300k-row versions (plus a 1M-row deep-interaction curve); 10M rows generate in 25 s. |
