"""Measure the task's reward structure under a random policy.

Answers the questions needed to interpret a score against the paper's
benchmark: how much is one goal worth, how long is an episode in decision
steps, and how often does a random walker score?

Usage (inside the dmlab container):
  python3 -m rl.probe_env --episodes 6 \
      --level contributed/dmlab30/explore_goal_locations_small
"""

import argparse
import collections
import json

import numpy as np

from rl.env import ACTIONS, LabEnv


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--level',
                  default='contributed/dmlab30/explore_goal_locations_small')
  ap.add_argument('--episodes', type=int, default=6)
  ap.add_argument('--action_repeat', type=int, default=4)
  ap.add_argument('--arena_cells', type=int, default=11)
  ap.add_argument('--seed', type=int, default=7)
  args = ap.parse_args()

  env = LabEnv(args.level, args.seed, arena_cells=args.arena_cells,
               cell_m=0.25)
  rng = np.random.default_rng(args.seed)
  per_ep, all_events = [], collections.Counter()
  steps_per_ep = []
  for _ in range(args.episodes):
    env.reset()
    total, steps, events = 0.0, 0, 0
    while True:
      a = int(rng.integers(0, len(ACTIONS)))
      _, r, done = env.step(a, args.action_repeat)
      steps += 1
      if r != 0:
        events += 1
        all_events[round(float(r), 3)] += 1
      total += r
      if done:
        break
    per_ep.append(total)
    steps_per_ep.append(steps)
  env.close()

  res = dict(
      level=args.level,
      episodes=args.episodes,
      mean_return_random=float(np.mean(per_ep)),
      returns=per_ep,
      mean_steps_per_episode=float(np.mean(steps_per_ep)),
      reward_event_magnitudes=dict(all_events),
      total_reward_events=sum(all_events.values()),
  )
  print(json.dumps(res, indent=1))


if __name__ == '__main__':
  main()
