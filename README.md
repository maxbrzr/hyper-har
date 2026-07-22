# BayaHAR

Official experiment code for **“BayaHAR: Lightweight Bayesian Few-Shot User
Adaptation for On-Device Personalized Human Activity Recognition”** (ISWC
2026).

BayaHAR turns a pretrained HAR classifier into a Prototypical Network with
source-domain prior prototypes. It supports:

- zero-shot inference with prior prototypes
- supervised few-shot adaptation with a closed-form MAP prototype update
- weakly supervised adaptation with a single closed-form MAP-EM update

The adaptation steps are source-free at deployment, gradient-free, and operate
only on the classifier embedding space. The camera-ready paper is included at
[paper/baya-har_cam-ready.pdf](paper/baya-har_cam-ready.pdf).

## Repository layout

```text
src/baya_har/
├── models/                 # TinierHAR backbone used in the paper
├── training/               # Cross-entropy LOSO training
├── experiments/            # One module per paper experiment or figure
└── cli.py                  # Unified `baya-har` command
paper/                      # Camera-ready manuscript
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

Run the full LOSO pipeline, all 100-episode few-shot evaluations, Figures 2–4,
and the computational-cost table:

```bash
uv run baya-har reproduce
```

This is the camera-ready configuration: 3-second windows, 50% overlap during
classifier training, no overlap during adaptation, 100 episodes per support
size, and support sizes 0–16. A full run trains one TinierHAR model per LOSO
fold on all four datasets and is therefore computationally expensive.

Stages can also be run independently:

```bash
# Train the pretrained classifiers.
uv run baya-har train

# Evaluate only MAP and MAP-EM on HARTH at 0, 1, 4, and 16 shots.
uv run baya-har evaluate \
  --datasets harth \
  --methods map map-em \
  --shots 0,1,4,16

# Recreate the plots from completed evaluations.
uv run baya-har figures

# Plot freshly generated dataset results instead of frozen paper values.
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
fold manifests. Support and query samples are disjoint. Macro-F1 is averaged
over classes observed in the ground truth or predictions. Globally defined
classes absent from both are not inserted with an F1 score of zero.

## Artifacts

Large per-dataset runs are excluded from Git because they contain model
checkpoints and per-episode records. The compact camera-ready figures, result
table, and FLOP table remain versionable. The cleaned layout is:

```text
artifacts/
├── datasets/<dataset>/     # shared splits, checkpoints, and method outputs
├── figures/                # paper overview and adaptation-curve figures
├── results/                # frozen camera-ready aggregate values
└── tables/                 # FLOP accounting
```

Frozen aggregate values are stored in `artifacts/results/paper_results.csv`
for adaptation curves and `artifacts/results/paper_overview_results.csv` for
the complete overview, including LOSO standard deviations. Figure generation
uses these paper values by default; pass `--results-source live` to read only
the freshly generated outputs under `artifacts/datasets/`.

## License

See [LICENSE](LICENSE).
