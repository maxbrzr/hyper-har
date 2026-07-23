# BayaHAR

Official experiment code for **BayaHAR: Lightweight Bayesian Few-Shot User
Adaptation for On-Device Personalized Human Activity Recognition**.

BayaHAR turns a pretrained HAR classifier into a Prototypical Network with
source-domain prior prototypes. It supports:

- zero-shot inference with prior prototypes
- supervised few-shot adaptation with a closed-form MAP prototype update
- weakly supervised adaptation with a single closed-form MAP-EM update

The adaptation steps are source-free at deployment, gradient-free, and operate
only on the classifier embedding space. The paper is included at
[paper/baya-har_cam-ready.pdf](paper/baya-har_cam-ready.pdf).

## Repository layout

```text
src/baya_har/
├── models/                 # TinierHAR backbone used in the paper
├── training/               # Cross-entropy LOSO training
├── experiments/            # One module per paper experiment or figure
└── cli.py                  # Unified `baya-har` command
paper/                      # Manuscript
artifacts/                  # Local outputs; see artifacts/README.md
```

## Setup

The project uses Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The four datasets are loaded through
[`whar-datasets`](https://github.com/teco-kit/whar-datasets): HHAR, WEAR,
HARTH, and HAPT. By default, raw and cached data are read from `datasets/`.
Use `--datasets-dir` to select a different location.

## Reproduce the paper

This is the paper configuration: 3-second windows, 50% overlap during
classifier training, no overlap during adaptation, 100 episodes per support
size, and support sizes 0–16. A full run trains one TinierHAR model per LOSO
fold on all four datasets and is therefore computationally expensive.

```bash
# Train the pretrained classifiers.
uv run baya-har train

# Evaluate only MAP and MAP-EM on HARTH at 0, 1, 4, and 16 shots.
uv run baya-har evaluate \
  --datasets harth \
  --methods map map-em \
  --shots 0,1,4,16

# Recreate the plots from completed evaluations from live runs or frozen paper runs.
uv run baya-har figures --results-source live

# Recreate the 1-shot/16-shot operation counts from saved checkpoints.
uv run baya-har flops
```

Use `--device cpu`, `--device cuda`, or `--device mps` to override automatic
device selection. `--max-folds 1` is useful for a smoke run; `--force` replaces
completed outputs.

## Experiment stages

| CLI method | Paper role | Labels | Gradients during adaptation |
|---|---|---:|---:|
| `original` | Original TinierHAR classifier | None | No |
| `prior` | Repurposed zero-shot prior prototypes | None | No |
| `protonet` | Standard ProtoNet baseline | Full | No |
| `logistic` | Logistic-probe performance ceiling | Full | Yes |
| `map` | BayaHAR MAP prototype estimation | Full | No |
| `map-em` | BayaHAR MAP-EM prototype estimation | Activity set only | No |
| `pda` | PDA baseline | None | No |
| `oftta` | Offline OFTTA baseline | None | No |


All methods use the same pretrained classifier checkpoints and shared LOSO
fold manifests. Support and query samples are disjoint.

## Artifacts

Large per-dataset runs are excluded from Git because they contain model
checkpoints and per-episode records. The cleaned layout is:

```text
artifacts/
├── datasets/<dataset>/     # shared splits, checkpoints, and method outputs
├── figures/                # paper overview and adaptation-curve figures
├── results/                # frozen paper aggregate values
└── tables/                 # FLOP accounting
```

Frozen aggregate values are stored in `artifacts/results/paper_results.csv`
for adaptation curves and `artifacts/results/paper_overview_results.csv` for
the complete overview, including LOSO standard deviations. Figure generation
uses these paper values by default; pass `--results-source live` to read only
the freshly generated outputs under `artifacts/datasets/`.

## Citation

If you use BayaHAR in your research, please cite the associated
[paper](https://arxiv.org/abs/2606.04798):

```bibtex
@article{burzer2026uncertainty,
  title={Uncertainty-Aware (Un) Supervised Few-Shot User Adaptation for On-Device Personalized Human Activity Recognition},
  author={Burzer, Maximilian and Riedel, Till and Beigl, Michael and R{\"o}ddiger, Tobias},
  journal={arXiv preprint arXiv:2606.04798},
  year={2026}
}
```

## License

See [LICENSE](LICENSE).
