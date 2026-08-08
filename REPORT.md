# Reproducing Banino et al. 2018 — results

**Goal:** reproduce, to the extent possible in one night on one machine plus
a small AWS budget, the results of *Vector-based navigation using grid-like
representations in artificial agents* (Nature 557:429), given that DeepMind
released only the supervised-experiment code, the training dataset was
deleted, the RL half was withheld as proprietary, and the quantitative
analysis pipeline was never published.

**What was rebuilt** (all in this repo): the training dataset
(`generate_trajectories.py`, from the Methods' motion model), the missing
analysis pipeline (`modern/analysis/`: shuffle nulls, border/HD scores,
position decoding, GMM scale clustering), a modern PyTorch/CUDA port for
Blackwell GPUs (`modern/`), and the never-released RL agent from the
Methods' description (`rl/`: vision module, grid module, A2C policy with
goal grid-code input, on real DeepMind Lab tasks, plus place-cell and A3C
baseline agents).

## Supervised experiments (paper Fig. 1) — REPRODUCED

Two independent stacks: the original TF1 code (Python-3-ported, 512-unit
configuration, regenerated dataset) and the PyTorch port (6 seeds, 3×10⁵
steps each, paper hyperparameters).

| Result | Paper | PyTorch port (6 seeds) | TF1 legacy run |
|---|---|---|---|
| Path-integration decode error, trained | 16 cm | **19.1 ± 0.6 cm** | — (no decoder in orig. code) |
| Decode error, untrained | 91 cm | 124.6 ± 14.2 cm | — |
| % grid units (threshold 0.37) | 25.2 % | **17.8 ± 2.0 %** | 17–22 % across final evals |
| % grid units (own shuffle threshold) | — | 12.1 ± 1.6 % (thr ≈ 0.46–0.49) | — |
| % border units (0.50) | 10.2 % | 22.8 ± 2.1 % † | — |
| % head-direction units (0.47) | 8.7 % | 12.4 ± 0.5 % | — |
| Top grid score | — | 1.2 ± 0.1 | 1.29–1.44 |
| Grid scale ratio (GMM clusters) | ~1.5 | **1.7 ± 0.1** | — |
| Hexagonal SACs (Fig. 1d) | yes | **yes** (fig_pytorch_gridcells.png) | **yes** (fig_tf1_top_sacs.png) |

† Border % is inflated by a scoring artifact we characterised: near-uniform
ratemaps produce a single giant "field" whose analytic border score
(≈0.556) sits just above the 0.50 threshold. The paper's exact border-score
implementation details are in the unavailable supplementary material.

**Verdict:** the paper's core supervised claims reproduce — hexagonal
grid-like units emerge robustly in both stacks, path integration reaches
within ~3 cm of the published accuracy, and grid scales cluster with a
ratio near the published ~1.5. Exact percentages differ (17.8% vs 25.2%
grid) — unsurprising given the reconstructed dataset, the unpublished
hyperparameter table (Supplementary Table 1), and the documented
seed-sensitivity of grid emergence (README warning; Schaeffer et al. 2022
found emergence knife-edge sensitive to unstated implementation choices).

## RL experiments (paper Figs. 2–4) — PIPELINE REBUILT, REDUCED SCALE

The paper: async A3C, 32 threads, **10⁹ env steps per replica**, ~60
replicas per experiment. That is ~5 vCPU-years per experiment — not
achievable overnight on any budget. What was done instead: a faithful-
in-architecture, synchronous A2C reimplementation trained at **1.5–3×10⁷
frames** (1.5–3 % of the paper's scale) on the real DeepMind Lab tasks:

- **Square arena** (`square_arena_10x10`, Fig. 2 analog, local GPU):
  grid-module grid-ness tracked over training; at the midpoint the module
  is pre-emergence (~1.6 % grid units, consistent with the supervised
  timeline where grids needed ~5× more grid-trainer updates than this run
  will reach). Final numbers in `rl_runs/arena_grid/grid_scores.json`.
- **Goal-driven maze** (`explore_goal_locations_small`, Fig. 3 analog,
  local GPU, 3×10⁷ frames): reward climbing (≈5 average return at 10M
  frames); goal-code capture and goal-vector decoding instrumented
  (`rl/decode_goal.py`, held-out-episode ridge regression).
- **Baselines on AWS spot** (place-cell agent, plain A3C; 1.5×10⁷ frames
  each, c7i.16xlarge, self-terminating, ~$3 total): final returns 6.0 and
  5.8 — at this scale no significant separation between agents is expected
  or observed; the paper's grid>place-cell>A3C ordering emerged over
  hundreds of millions of frames.

**Verdict:** the RL half is now *runnable and instrumented* end-to-end
(environment, all three agents, grid-ness and goal-decoding analyses), but
at overnight scale it demonstrates directional trends, not the paper's
final RL results. Reaching those requires ~30–60× more compute per run.
The unreleased components (custom sunburst/double-E mazes, lesion
protocol, human-expert comparison) remain unimplemented.

## Infrastructure delivered

- `Dockerfile` (TF 1.15.5 env) · `Dockerfile.rl` (DeepMind Lab + PyTorch)
- `generate_trajectories.py` — dataset reconstruction (deleted bucket)
- `modern/` — PyTorch/Blackwell port + full quantitative analysis
- `rl/` — DM-Lab env layer (measured calibration), A2C grid-cell agent,
  baselines, analysis tools
- `aws/` — self-terminating spot provisioning (presigned URLs, no
  credentials on instances)
- `REPORT.json` — machine-readable aggregate of every number above

Reproduce: see RUNNING.md (supervised) and rl/train_rl.py docstring (RL).
