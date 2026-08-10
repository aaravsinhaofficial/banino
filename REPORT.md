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
ratio near the published ~1.5.

**Verified against the actual Supplementary Information** (fetched from
Nature; now in this repo as Banino-2018-SI.pdf): our supervised
configuration matches SI Table 1 on *every* parameter — motion model
(T=15 s, L=2.2 m, d=0.03 m, Rayleigh 0.13 m/s, rotation N(0, 330°/s),
dt=0.02 s, rho_RH=0.25, Delta_RH=90°), ensembles (N=256, sigma=0.01 m,
M=12, kappa=20) and training (clip 1e-5, batch 10, length 100, lr 1e-5,
momentum 0.9, L2 1e-5, 3x10^5 steps). The 17.8% vs 25.2% gap is therefore
attributable to seed variance, the still-inferred ego_vel encoding
convention, and analysis-procedure details — not to wrong hyperparameters
(Schaeffer et al. 2022 document exactly this kind of sensitivity).

## RL experiments (paper Figs. 2–4) — PIPELINE REBUILT, REDUCED SCALE

The paper: async A3C, 32 threads, **10⁹ env steps per replica**, ~60
replicas per experiment. That is ~5 vCPU-years per experiment — not
achievable overnight on any budget. What was done instead: a faithful-
in-architecture, synchronous A2C reimplementation trained at **1.5–3×10⁷
frames** (1.5–3 % of the paper's scale) on the real DeepMind Lab tasks:

- **Square arena** (`square_arena_10x10`, Fig. 2 analog, local GPU, 2×10⁷
  frames, complete): the grid module did **not** develop grid cells —
  0.2–6 % of units above threshold across training (chance ≈ 3.5 %), no
  upward trend, even as its path-integration loss steadily improved. This
  matches the supervised timeline: the module received ~3×10⁴ trainer
  updates, whereas grids first emerged in the supervised runs after
  ~5×10⁴–3×10⁵ updates.
- **Goal-driven maze** (`explore_goal_locations_small`, Fig. 3 analog,
  local GPU, 3×10⁷ frames, complete): final smoothed return ≈ 5–7,
  finishing slightly above both baselines at frame parity but within
  noise (fig_rl_returns.png). Goal-vector decoding from the policy LSTM
  (held-out-episode ridge regression, `rl/decode_goal.py`): **negative** —
  R² ≤ 0 for goal distance and direction (direction error 82° vs 90°
  chance; shuffled-target control equivalent). The paper's decodable goal
  vector has not formed at this scale. Grid module in the maze: ~2 %
  grid-like (chance).
- **Baselines on AWS spot** (place-cell agent, plain A3C; 1.5×10⁷ frames
  each, c7i.16xlarge, self-terminating, ~$3 total): final returns 6.0 and
  5.8 — no significant separation between the three agents, as expected:
  the paper's grid > place-cell > A3C ordering emerged over hundreds of
  millions of frames.

### Warm-start experiment (added after the overnight runs)

A follow-up tested whether the *supervised* grid network can be transplanted
into the agent: the RL grid module was initialised from the best supervised
checkpoint (21 % grid units, 19.3 cm decode; velocity input re-encoded to the
supervised convention, visual-code input columns zeroed, place-cell ensembles
aligned at ±1.1 m) and trained 20 M frames on the goal maze
(fig_warmstart_returns.png). Outcome, honestly read from full curves:

- **No clear navigational advantage at this scale.** All four agents (warm
  grid, scratch grid, place-cell, A3C) reach return ≈ 5–6 within 1–2 M frames
  and end in the same noisy 4–6 band (tail averages 4.9–6.2). The warm agent
  posts the highest transient peak of any run (≈ 8.8 at 7–8 M frames vs 7.9
  for the best baseline), which is suggestive but n=1.
- The transferred representation survives fine-tuning (maze grid fraction
  4.1 % vs ≈ 2 % from scratch; positive mean grid score), but the goal
  vector remains undecodable (R² ≤ 0, direction error 88° ≈ chance).
- Interpretation: a pretrained path integrator is not the binding
  constraint at 2×10⁷ frames — the paper's separation presumably needs the
  10⁸–10⁹-frame regime regardless of how good the spatial code is.

**Verdict:** the RL half is *runnable and instrumented* end-to-end
(environment, all three agents, grid-ness and goal-decoding analyses), and
the overnight-scale results are cleanly **negative in the directions the
paper predicts they should be at 1.5–3 % of the published training
budget**: no grid emergence in the agent's grid module yet, no decodable
goal vector yet, no agent separation yet. Reaching the paper's RL results
requires ~30–60× more compute per run (≈$1–2k per experiment on spot
CPU, or equivalent GPU-actor time). The unreleased components (custom
sunburst/double-E mazes, lesion protocol, human-expert comparison) remain
unimplemented.

## FULL-SCALE RL REPLICATION (10⁹ frames per agent) — COMPLETED

Eight cells trained to the paper's full 10⁹-frame budget with SI Table 2
hyperparameters (grid + place-cell + A3C on the goal maze and goal-doors;
grid + place-cell on a custom open square arena with the same task
semantics). Two mid-run incidents, both documented in git history: the
SI's unclipped grid-network recipe (lr 1e-3) destabilised in 4 of 4 grid
trainers and required a clip (at the paper's own supervised clip/lr
ratio) plus one-shot grid-module resets. Frozen-policy evaluation, mean
score over 100 episodes (the SI's protocol):

| Task | grid | place-cell | A3C | Paper (grid / place) |
|---|---|---|---|---|
| Goal maze | 7.6 ± 1.0 | 7.1 ± 1.2 | 6.1 ± 1.0 | 289 / 238 |
| Goal-doors | 5.1 ± 0.8 | 5.3 ± 0.8 | 3.9 ± 0.7 | 284.3 / 90.5 |
| Square arena | 7.8 ± 1.4 | 10.5 ± 1.5 | — | — |

**What reproduced at full scale:**
- **Grid cells emerge inside the RL agent** (Fig 2g analog): 14.5 % of the
  arena grid agent's module units score above 0.37 with a top grid score
  of 1.32 (paper: 21.4 %); the place-cell control's independently trained
  module reaches 16.4 %. At reduced scale this was chance-level.
- **A metric goal vector forms in the policy LSTM** (Fig 2j analog):
  held-out-episode ridge decoding from the arena grid agent recovers goal
  distance at R² = 0.55 (shuffled control −0.15; MAE 0.50 m) and goal
  direction to 24.4° mean error (chance 90°). The maze agent carries the
  direction signal (36.9°) but not distance.
- A3C is reliably the worst agent on both maze tasks, and the agent
  ordering on the mazes is direction-consistent with the paper (grid ≥
  place-cell > A3C), within error bars for grid-vs-place.

**What did not reproduce:** absolute navigation performance and clear
agent separation. All maze agents plateau at scores ~40–55× below the
published benchmarks and overlap for most of training; in the arena the
place-cell agent ends above the grid agent. The published scores imply
~29 goal reaches per 90 s episode versus our ~0.5–1 — a qualitative
behavioural gap, not a tuning-size one. Candidate causes, in order of
suspicion: synchronous A2C versus their async A3C with shared-statistics
RMSProp; our single hyperparameter draw versus their best-30-of-60
replica selection; unstated details of episode/reward structure. The
representational claims survive this gap; the performance claims remain
unverified by this replication.

**Post-hoc correction from the retrieved SI** (Banino-2018-SI.pdf, Table 2):
three of our guessed RL hyperparameters were wrong — entropy cost 1e-3 vs
the paper's [6e-5, 1e-4]; agent grid-network lr 1e-5 (we reused the
supervised recipe) vs the paper's **1e-3, no clip, L2 1e-4**; actor-critic
BPTT 20 vs 100. All reduced-scale runs above used the mis-guessed values —
in particular, the agent's grid network was training ~100× slower than the
paper's, which compounds the under-training explanation for the missing
grid emergence. rl/train_rl.py defaults now follow SI Table 2. SI also
supplies the full-scale targets any future run should be compared against
(goal maze, mean score over 100 episodes: grid 289 vs place-cell 238;
goal-doors: 284.3 vs 90.5; human expert 346.5).

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
