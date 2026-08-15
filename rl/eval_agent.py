"""Frozen-policy evaluation: mean score over N episodes (SI benchmark
protocol — the paper reports 'average score over 100 episodes').

No learning, no exploration changes: the stochastic policy acts exactly as
in training (grid-module dropout stays active, as during acting).

Usage (inside dmlab-rl container):
  python3 -m rl.eval_agent --ckpt rl_runs/fs_goal_grid/ckpt_final.pt \
      --level contributed/dmlab30/explore_goal_locations_small \
      --agent grid --episodes 100 --out rl_runs/fs_goal_grid
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.nets import GridModule, PolicyNet, VisionCNN  # noqa: E402
from rl.vec_env import SubprocVecEnv                  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--ckpt', required=True)
  ap.add_argument('--level', required=True)
  ap.add_argument('--agent', choices=['grid', 'placecell', 'a3c'],
                  required=True)
  ap.add_argument('--episodes', type=int, default=100)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--arena_cells', type=int, default=11)
  ap.add_argument('--vel_encoding', choices=['paper', 'raw', 'supervised'],
                  default=None)
  ap.add_argument('--action_repeat', type=int, default=4)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--fake', action='store_true')
  ap.add_argument('--out', required=True)
  ap.add_argument('--grid_eval_mode', action='store_true',
                  help='Run the grid module in eval() mode (dropout off) '
                       'instead of train() mode, so the policy sees the '
                       'expected grid code rather than a fresh 50%% mask '
                       'each step.')
  ap.add_argument('--lesion_goal', action='store_true',
                  help="Zero the goal code fed to the policy LSTM (the "
                       "paper's Extended Data Fig. 6c lesion). Used here to "
                       'test whether the grid agent\'s advantage over our '
                       'place-cell control is just the goal code, which that '
                       'control was never given.')
  ap.add_argument('--reward_clip', type=float, default=1.0,
                  help='Must match training: prev_reward is a policy input, '
                       'so the agent has to see the same scale it learned '
                       'on. Scores are always reported raw.')
  args = ap.parse_args()

  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  ck = torch.load(args.ckpt, map_location=dev)
  n_actions = ck['policy']['pi.weight'].shape[0]  # match the trained head
  # Grid LSTM input width tells us the velocity encoding the run used.
  n_vel = ck['grid']['cell.weight_ih'].shape[1] - 256 - 12
  vision, grid, policy = VisionCNN().to(dev), GridModule(n_vel=n_vel).to(dev), \
      PolicyNet(n_actions=n_actions).to(dev)
  vision.load_state_dict(ck['vision'])
  grid.load_state_dict(ck['grid'])
  policy.load_state_dict(ck['policy'])
  vision.eval()
  policy.eval()   # no dropout layers, but freeze norm-free semantics anyway
  # The training loop leaves the grid module in train() mode while acting, so
  # the policy sees a fresh 50%-dropped, 2x-rescaled grid code every step.
  # The paper does not specify this, and its lesion protocol (training with
  # 20% dropout on the goal code so the LSTM "would become robust") implies
  # the code fed to the policy was not already 50% corrupted.
  # --grid_eval_mode evaluates with dropout off, i.e. the expected code.
  grid.eval() if args.grid_eval_mode else grid.train()

  step_dt = args.action_repeat / 60.0

  # Default to whatever the checkpoint was trained with: a 4-wide grid
  # input means the paper encoding [u, v, sin, cos], 3 means raw [u, v, w].
  if args.vel_encoding is None:
    args.vel_encoding = 'paper' if n_vel == 4 else 'raw'

  def encode_vel(obs):
    v = obs['vel']
    dtheta = v[:, 2] * step_dt
    if args.vel_encoding == 'paper':
      obs['vel'] = np.stack([v[:, 0], v[:, 1],
                             np.sin(dtheta), np.cos(dtheta)],
                            axis=1).astype(np.float32)
    elif args.vel_encoding == 'supervised':
      obs['vel'] = np.stack([np.hypot(v[:, 0], v[:, 1]),
                             np.sin(dtheta), np.cos(dtheta)],
                            axis=1).astype(np.float32)
    return obs

  n = args.n_envs
  venv = SubprocVecEnv(n, args.level, base_seed=424242, fake=args.fake,
                       env_kwargs=dict(arena_cells=args.arena_cells,
                                       cell_m=0.25) if not args.fake else None)
  obs = encode_vel(venv.reset_all())
  pol_state = policy.zero_state(n, dev)
  grid_state = grid.zero_state(n, dev)
  goal_code = torch.zeros(n, 512, device=dev)
  prev_action = torch.zeros(n, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)
  ep_return = np.zeros(n)
  scores = []

  with torch.no_grad():
    while len(scores) < args.episodes:
      rgb = torch.as_tensor(obs['rgb'], device=dev).permute(0, 3, 1, 2).float() / 255.
      vel = torch.as_tensor(obs['vel'], device=dev)
      pc_l, hd_l = vision(rgb)
      vis_pc, vis_hd = torch.softmax(pc_l, -1), torch.softmax(hd_l, -1)
      mask = (torch.rand(n, 1, device=dev) < 0.05).float()
      g, grid_state = grid.step(vel, vis_pc, vis_hd, mask, grid_state)
      if args.agent == 'grid':
        g_in = g
        goal_in = torch.zeros_like(goal_code) if args.lesion_goal else goal_code
      elif args.agent == 'placecell':
        g_in = torch.cat([vis_pc, vis_hd,
                          torch.zeros(n, 512 - 268, device=dev)], -1)
        goal_in = (torch.zeros_like(goal_code) if args.lesion_goal
                   else goal_code)
      else:
        g_in = torch.zeros_like(g)
        goal_in = torch.zeros_like(goal_code)
      pi, _, pol_state = policy.step(rgb, g_in, goal_in, prev_action,
                                     prev_reward, pol_state)
      a = torch.distributions.Categorical(logits=pi).sample()
      obs, rewards, dones = venv.step(a.cpu().numpy())
      obs = encode_vel(obs)
      r_raw = torch.as_tensor(rewards, device=dev)
      r_t = r_raw.clamp(-args.reward_clip, args.reward_clip) \
          if args.reward_clip > 0 else r_raw
      hit = (r_raw > 0).float().unsqueeze(1)
      goal_code = hit * g_in.detach() + (1 - hit) * goal_code
      ep_return += rewards
      for i in range(n):
        if dones[i]:
          scores.append(float(ep_return[i]))
          ep_return[i] = 0.0
          goal_code[i].zero_()
          grid_state[0][i].zero_(); grid_state[1][i].zero_()
          pol_state[0][i].zero_(); pol_state[1][i].zero_()
      prev_action = a
      prev_reward = r_t

  venv.close()
  scores = np.array(scores[:args.episodes])
  res = dict(mean_score=float(scores.mean()), std=float(scores.std()),
             sem=float(scores.std() / np.sqrt(len(scores))),
             n_episodes=int(len(scores)), ckpt=args.ckpt)
  print(json.dumps(res))
  with open(os.path.join(args.out, 'eval_scores.json'), 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
