# Running this code (2026 update)

The upstream README predates the death of its own instructions: the pinned
packages (TF 1.12, Sonnet 1.27, matplotlib 1.5) no longer install on any
modern Python, and the official dataset bucket (`gs://grid-cells-datasets`)
was deleted from GCS — every endpoint returns `404 NoSuchBucket`, there is no
Wayback snapshot, and no mirror exists on HuggingFace/Kaggle/Zenodo/GitHub
(upstream issues #3/#4/#5 about the dead link are unanswered). This repo
therefore carries a small set of additions that make the code runnable today:

- **Python 3 port** of the original Python 2 sources (mechanical changes only).
- **`Dockerfile` / `requirements.txt`** — pinned environment on
  `tensorflow/tensorflow:1.15.5-py3` (the last TF1 release) with Sonnet 1.36.
  CPU-only: TF1-era CUDA does not run on current GPUs.
- **`generate_trajectories.py`** — re-simulates the deleted dataset
  (1M foraging trajectories, Raudies–Hasselmo motion model, 15 s at
  dt = 0.02 s subsampled to 100 steps of 0.15 s) into the exact TFRecord
  schema `dataset_reader.py` expects. DeepMind only ever released the
  *reader*, so the 3-component `ego_vel` encoding is an inferred convention
  (speed, sin/cos of per-step heading change) — a reconstruction, not the
  original data.

## Quickstart

```shell
docker build -t grid-cells:tf1 .

# ~2.4 GB, a few minutes on many cores
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -v $PWD:/workspace grid-cells:tf1 \
  python generate_trajectories.py --root data

# Paper-faithful supervised run: 512-unit linear layer, 3e5 gradient steps
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp/mpl \
  -v $PWD:/workspace grid-cells:tf1 \
  python train.py --task_root=data --saver_results_directory=results \
  --model_nh_bottleneck=512 --training_epochs=300 --saver_eval_time=25
```

Each evaluation writes `results/rates_and_sac_epoch_NNNN.pdf` (ratemaps and
spatial autocorrelograms of all linear-layer units, sorted by 60° grid score)
and logs the top/mean grid score. ~30 gradient steps/s on a 48-core CPU
(~3 h for the full 3×10⁵ steps).

Note `train.py`'s default `--model_nh_bottleneck` is 256, but the paper's
linear layer g has **512** units (129/512 = 25.2% grid-like) — pass 512
explicitly, as above, for the paper configuration.

## What this can and cannot reproduce

This repo is DeepMind's official release for the paper, but it covers **only
the supervised path-integration experiments** (Fig. 1 and related Extended
Data). An audit of the paper against the code:

| Paper result | Status |
|---|---|
| Grid-like units emerge in the linear layer (Fig. 1d) | **Qualitative only** — ratemaps/SACs/grid scores are produced, but the significance thresholds (0.37/0.50/0.47) come from a field-shuffle null that was never released, so the 25.2% grid / 10.2% border / 8.7% HD classifications can't be computed exactly |
| Path-integration error (16 cm vs 91 cm untrained, Fig. 1a–c) | **No** — no position decoder or error metric in the repo |
| Grid-scale clustering, ratio ≈ 1.5 (Fig. 1e) | **No** — scale measurement, GMM/BIC fit and discreteness shuffle were bespoke Matlab, never released |
| RL agent self-localizes (Fig. 2a–e) | **No** — the entire RL codebase (A3C grid-cell agent, vision module, replay) was withheld as proprietary |
| Goal-directed vector navigation beats baselines (Figs. 2f–k, 3) | **No** — the DM-Lab *tasks* are public (`explore_goal_locations`, `explore_obstructed_goals`), but no agent code exists |
| Lesion / fake-goal / shortcut probes (Figs. 2i, 4) | **No** — require the unreleased agent and custom mazes |
| Training data | **Reconstruction** — original deleted; regenerated here from the Methods description |

Two further caveats from the replication literature: the exact numeric
hyper-parameters live in Supplementary Methods Table 1 (not in the main PDF
or this repo), and grid emergence is known to be fragile — the upstream
README warns results vary with seeds, one community replication achieved
path integration with *zero* grid cells, and Schaeffer, Khona & Fiete
(NeurIPS 2022, "No Free Lunch from Deep Learning in Neuroscience") found
grid-like units in only ~7% of >3,500 trained networks, knife-edge sensitive
to the place-cell tuning width. Expect qualitative, not exact, reproduction —
and run multiple seeds. For a modern, better-understood reimplementation see
`ganguli-lab/grid-pattern-formation`.
