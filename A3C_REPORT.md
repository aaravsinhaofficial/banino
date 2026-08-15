# Async A3C run — what changed, and what the previous RL results actually were

**Date:** 2026-08-15. **Brief:** replace the synchronous A2C with true async
A3C (shared-statistics RMSProp), use the paper's hyperparameters, and try to
reach the paper's performance within 12 hours and $500.

## Headline

The architecture was not the problem. Chasing it surfaced a defect that
invalidates the performance half of the previous full-scale run:

> **Every RL number this project produced before today sits at the
> untrained-network chance line.**

Measured with an untrained (random-init) network through the *same*
100-episode protocol used for every trained cell:

| task | untrained network | previously reported "results" |
|---|---|---|
| `explore_goal_locations_small` | **7.2 ± 1.07** | grid 7.6 ± 1.0, place-cell 7.1 ± 1.2, A3C 6.1 ± 1.0 |

All three previously reported scores are within noise of an untrained
network, so the "grid ≥ place-cell > A3C ordering is direction-consistent
with the paper" claim in REPORT.md was noise around a policy that had never
learned anything.

## Root cause: the rollout loss averaged where A3C accumulates

`pol_loss = -(logp * adv).mean()` over the T=100-step BPTT window. Mnih et
al. 2016 — which the paper's Methods explicitly say it followed ("RMSProp
with shared gradient statistics"), and whose Algorithm 1 reads
`dθ ← dθ + ∇log π(a|s)(R − V)` — *accumulates* gradients along the rollout.
Averaging divides every policy and value gradient by T. With RMSProp's
eps=0.1 in the denominator the resulting parameter steps were ~1e-6, and the
policy never left uniform: entropy sat at exactly ln(n_actions) for the whole
of training.

The old synchronous trainer averaged over T×n = 3200, which is why the
full-scale 10⁹-frame sync run also plateaued at chance.

Two independent checks that SI Table 2's constants assume sum reduction:
entropy 8e-5 summed over 100 steps is a standard ~8e-3 per-step bonus, and
baseline cost 0.5 is the textbook A3C value. Under mean reduction both are
absurdly small.

## How it was isolated

1. `rl/probe_env.py` — measures a task's reward structure. Goal maze: +10 per
   goal, 1350 decision steps per episode, and a uniform-random walker scores
   ~0.8. So the trained agents' 6–9 was not "weak navigation", it was noise.
2. `rl/bandit_env.py` — a dense-reward probe env. The same A3C machinery goes
   from random to 89% of optimal in 90 seconds, which rules out the learner,
   the shared-memory Hogwild updates, the shared RMSProp, and BPTT.
3. `pol_norm` / `adv_abs` logging — on DM-Lab the policy parameter norm was
   frozen to four decimals while the dense probe moved it +0.045 over the same
   number of updates: gradients ~100× too small, exactly as the reduction bug
   predicts.

## Two further fidelity gaps found and fixed

- **Action set.** Ours had 6 disjoint actions, omitting DeepMind Lab's
  combined *forward+look-left/right* — an agent had to stop moving to turn.
  The paper's agents used the standard DM-Lab set. Fixed to 8 actions.
- **Reward scale.** The goal levels pay +10 unclipped, so at a reward step the
  summed value loss (~50) dominated the grad-norm-40 clip and swamped the
  policy term. Clipped to ±1 for the learning signal; **reported scores stay
  raw** and comparable to the paper's 289. Evaluation and goal-decoding clip
  identically because `prev_reward` is a policy input.

## A methodological note worth keeping

An untrained *network* is not a uniform-random *action* policy. Uniform random
scores ~0.8 on the goal maze; the untrained network scores 7.2, because its
arbitrary LSTM biases produce correlated, directed motion that explores far
better than coin-flipping. Comparing against an assumed "random ≈ 0" sets the
bar roughly 9× too low — which is how the earlier run's chance-level scores
came to be written up as a reproduced agent ordering.

Task difficulty for an undirected walker also varies enormously, which decides
where learning can bootstrap in a small budget:

| task | goals per episode (random) |
|---|---|
| `explore_goal_locations_small` (paper Fig. 3) | 0.08 |
| `square_arena_goal` (paper Fig. 2) | 0.83 |
| `seekavoid_arena_01` (dense control) | ~3.5 rewards/episode |

## This run

Seven cells, all with the fixes, ~9,000 fps each, self-terminating after a
100-episode frozen-policy eval.

| cell | agent | task | purpose |
|---|---|---|---|
| a3c_goal_grid | grid | goal maze | paper Fig. 3 claim |
| a3c_goal_place | place-cell | goal maze | matched control |
| a3c_goal_a3c | A3C | goal maze | matched baseline |
| a3c_goal_grid_hi | grid, lr 2e-4 | goal maze | second hyperparameter draw |
| a3c_arena_grid | grid | square arena | paper Fig. 2 claim (grid > place) |
| a3c_arena_place | place-cell | square arena | matched control |
| a3c_seekavoid | A3C | seekavoid | diagnostic control: can it learn a dense real DM-Lab task? |

Deviations from the paper, stated up front: 64 worker threads on the maze
cells where the paper used 32 (throughput; at lr 1e-4 the effective update
rate lands at the top of the SI's sampled range — the arena cells use the
paper's 32), reward clipping (not mentioned in the SI), and ~2.8×10⁸ frames
per cell against the paper's 10⁹ per replica with best-30-of-60 selection.

## Measured chance lines

Every score below is judged against an untrained random-init network run
through the identical 100-episode protocol, per task:

| task | untrained network | uniform-random actions |
|---|---|---|
| `explore_goal_locations_small` | 7.2 ± 1.07 | ~0.8 |
| `square_arena_goal` | 3.2 ± 0.60 | ~8.3 (6-episode probe) |
| `seekavoid_arena_01` | 1.12 ± 0.11 | ~1.5 (6-episode probe) |

## Interim results (~25M of ~280M frames per cell)

**The fixes work.** Every cell has left its chance band, and policy entropy
falls monotonically from its ln(n_actions) ceiling — the quantity that stayed
pinned at exactly the maximum for the whole of the previous 10⁹-frame run.

- **Dense control (`seekavoid`) — decisive, and now evaluated.** Frozen-policy
  eval at 34.4M frames: **16.65 ± 0.42 vs chance 1.12 ± 0.11**, a 15×
  improvement (~36σ), with training entropy 2.08 → 0.65. This pipeline learns
  a real DM-Lab task from raw vision, so any shortfall on the paper's task is
  about reward sparsity and compute, not implementation. The cell was then
  retired early and its capacity given to a second arena seed.
- **Goal maze (paper Fig. 3).** All four cells climb out of the untrained
  band at ~8–9M frames; 12–18 by 25M against chance 7.2. No agent separation
  yet (the paper's central claim), and it is far too early to expect one.
- **Square arena (paper Fig. 2) — large swings, no stable ranking.** The grid
  agent's training return oscillates violently: 37.8 at 23M frames, down to
  19.8 at 74M, back up to 44.4 at 108M. The place-cell control is far
  steadier (17 → 21 → 19). At some checkpoints the grid agent is 2–2.6× the
  control and it looks like a clean reproduction of Fig. 2f; at others they
  are at parity.

  A second independent seed does not track the first: at ~20M frames seed 1
  gave grid 23.5 vs place-cell 16.9, while seed 2 gave 18.7 vs 17.8, and at
  ~60M seed 2 has place-cell marginally *ahead* (25.6 vs 24.4).

  The maze cells swap rank the same way — grid led at 108M (20.8 vs 16.2),
  and at 142M place-cell and the lr-2e-4 grid draw lead instead (20.8 vs
  13.8).

  **The 50-episode rolling training return is too noisy to rank agents**, and
  any claim drawn from a single snapshot of it — in either direction — would
  be unsound. Only the 100-episode frozen-policy evals with error bars can
  settle the ordering, and with 1–2 seeds per configuration they will only
  resolve large gaps. This is the same trap that produced the previous
  report's false agent ordering; it is avoided here by having measured
  chance lines, replicate seeds, and a held-out eval protocol.

  Candidate mechanism for the instability, not tested for want of time and
  vCPU quota: the grid module is retrained from replay throughout (its loss
  still oscillates 4.4–5.9 late in training), so the policy's input
  representation keeps shifting under a policy already fitted to it.
  `rl/train_a3c.py --freeze_grid_after N` is implemented to test exactly
  this and is the obvious follow-up.

![learning curves](report/fig_a3c.png)

### Goal-vector decoding, and a caveat about it

Mid-run check on the arena grid agent (54M frames, ridge regression from the
policy LSTM on held-out episodes): goal distance R² = 0.285 (shuffled control
−0.013), MAE 0.45 m, direction error 39.2° against 90° chance. That is the
paper's Fig. 2j/k quantity, and unlike the previous run's reported R² = 0.55
it comes from an agent that actually navigates (35.0 vs chance 3.2) rather
than one sitting at the untrained baseline.

The caveat, which applies to the paper's analysis as much as ours: the policy
LSTM *receives the current grid code and the goal grid code as inputs*, so
some goal-distance information is present in its state by construction, not
by learning. A decode R² above zero is therefore not by itself evidence of a
learned vector computation. What carries weight is the paper's actual
comparison — grid agent versus place-cell agent under the identical protocol
— which is run for both arena cells at the end.

## Results

Single checkpoints of the 50-episode training return are far too noisy to
rank agents — cells trade places between snapshots, and picking a
favourable one can "show" almost any ordering. Averaging over frame windows
removes that, and the two tasks then give opposite answers.

### Square arena (paper Fig. 2) — the grid advantage reproduces

Mean training return over 20M-frame windows, grid agent vs place-cell
control, two independent seeds:

| window | seed 1 grid | seed 1 place | ratio | seed 2 grid | seed 2 place | ratio |
|---|---|---|---|---|---|---|
| 0–20M | 11.8 | 9.4 | 1.25 | 12.7 | 12.0 | 1.05 |
| 20–40M | 31.0 | 17.1 | 1.82 | 23.0 | 19.2 | 1.20 |
| 40–60M | 31.5 | 21.3 | 1.48 | 30.0 | 24.3 | 1.23 |
| 60–80M | 21.2 | 21.1 | 1.00 | 35.9 | 25.2 | 1.42 |
| 80–100M | 26.2 | 21.2 | 1.23 | 32.2 | 30.5 | 1.06 |
| 100–120M | 35.1 | 19.8 | 1.77 | 48.3 | 23.5 | 2.06 |
| 120–140M | 31.2 | 23.0 | 1.36 | 48.9 | 25.4 | 1.93 |
| 140–160M | 36.2 | 25.6 | 1.42 | — | — | — |
| 160–200M | 32.3 | 28.4 | 1.14 | — | — | — |

The grid agent leads in **17 of 17 windows across both seeds**, by ~1.4× on
average. That is the qualitative claim of Fig. 2f — a grid-cell agent
navigates to a hidden goal in an open arena better than a place-cell agent —
reproduced here at ~2×10⁸ frames per cell.

Caveats worth stating: windows within a run are correlated, so the effective
sample is closer to 2 seeds than 17 windows; treating seeds as the unit,
2 of 2 favour grid, which alone is weak evidence. The frozen-policy evals
below are the clean test.

### Goal maze (paper Fig. 3) — no agent separation

Mean training return over all windows past 40M frames (chance 7.2):

| grid | grid lr 2e-4 | place-cell | A3C (no grid inputs) |
|---|---|---|---|
| 23.7 | 24.5 | 24.1 | 24.2 |

Four agents, four essentially identical numbers. The paper reports grid >
place-cell > A3C on this family of tasks; at our scale there is no ordering
to find. Note the A3C baseline has **no grid or goal-code inputs at all** and
matches the grid agent exactly, so nothing about the maze result is
attributable to the grid representation.

That the two tasks disagree is itself coherent: an open arena is where a
metric, vector-like spatial code should help most, while the maze demands
routing around walls, for which a Euclidean goal vector is a poorer guide.

### Absolute performance vs the paper

All agents learn — 3–7× their measured chance lines, versus the previous
full-scale run which never left chance. But the paper's benchmark is 289 on
the goal maze (~29 goal arrivals per episode); our best maze cells reach
~35–50 (3–5 arrivals). We are ~6–8× short, on ~2.4×10⁸ frames per cell
against the paper's 10⁹ per replica with 60 replicas and best-30 selection.
