# Artifact layout

Large per-dataset experiment outputs are stored locally and ignored by Git.
Compact paper figures, aggregate results, and computational-cost tables are
kept versionable.

- `datasets/<dataset>/classifier`: LOSO TinierHAR checkpoints and histories.
- `datasets/<dataset>/original`: original-classifier evaluation.
- `datasets/<dataset>/prior_*`: prior-prototype zero-shot evaluation.
- `datasets/<dataset>/protonet_*`: standard ProtoNet baseline.
- `datasets/<dataset>/map_*`: supervised BayaHAR MAP evaluation.
- `datasets/<dataset>/map_em_*`: weakly supervised BayaHAR MAP-EM evaluation.
- `datasets/<dataset>/{pda,oftta,logistic}_*`: paper baselines.
- `figures/`: generated camera-ready plots.
- `results/paper_results.csv`: frozen adaptation-curve values used by the paper.
- `results/paper_overview_results.csv`: frozen overview means and LOSO standard
  deviations.
- `tables/computational_cost/`: backbone and adaptation operation counts.

Every evaluation directory contains a `summary.json` plus aggregate and
per-episode CSV files. All methods for a dataset share the fold manifest under
`datasets/<dataset>/shared_splits/`.
