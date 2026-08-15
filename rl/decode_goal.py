"""Goal distance/direction decoding from the policy LSTM (Fig 2j/3i analog).

Runs the trained agent for eval episodes, infers the goal position from the
agent's position at reward events, and after the first goal hit of each
episode regresses (Euclidean goal distance, sin/cos of allocentric goal
direction) from the policy LSTM hidden state with ridge regression
(sklearn, as the paper's Reporting Summary states). Reports R^2 and mean
absolute errors, plus a shuffled-target control.

Usage (inside dmlab container for real env, anywhere with --fake):
  python3 -m rl.decode_goal --ckpt rl_runs/grid1/ckpt_final.pt --fake \
      --episodes 40 --out rl_runs/grid1
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
  ap.add_argument('--level',
                  default='contributed/dmlab30/explore_goal_locations_small')
  ap.add_argument('--agent', choices=['grid', 'placecell', 'a3c'],
                  default='grid')
  ap.add_argument('--episodes', type=int, default=40)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--fake', action='store_true')
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--out', required=True)
  ap.add_argument('--vel_encoding', choices=['raw', 'supervised'],
                  default='raw', help='Must match the training run.')
  ap.add_argument('--action_repeat', type=int, default=4)
  ap.add_argument('--arena_cells', type=int, default=11)
  args = ap.parse_args()

  step_dt = args.action_repeat / 60.0

  def encode_vel(obs):
    if args.vel_encoding == 'supervised':
      v = obs['vel']
      dtheta = v[:, 2] * step_dt
      obs['vel'] = np.stack([np.hypot(v[:, 0], v[:, 1]),
                             np.sin(dtheta), np.cos(dtheta)],
                            axis=1).astype(np.float32)
    return obs

  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  ck = torch.load(args.ckpt, map_location=dev)
  n_actions = ck['policy']['pi.weight'].shape[0]  # match the trained head
  vision, grid, policy = VisionCNN().to(dev), GridModule().to(dev), \
      PolicyNet(n_actions=n_actions).to(dev)
  vision.load_state_dict(ck['vision'])
  grid.load_state_dict(ck['grid'])
  policy.load_state_dict(ck['policy'])

  n = args.n_envs
  venv = SubprocVecEnv(n, args.level, base_seed=7777, fake=args.fake,
                       env_kwargs=dict(arena_cells=args.arena_cells,
                                       cell_m=0.25) if not args.fake else None)
  obs = encode_vel(venv.reset_all())
  pol_state = policy.zero_state(n, dev)
  grid_state = grid.zero_state(n, dev)
  goal_code = torch.zeros(n, 512, device=dev)
  goal_pos = np.full((n, 2), np.nan)
  prev_action = torch.zeros(n, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)

  H, D, G = [], [], []
  ep_ids = np.arange(n)
  next_ep = n
  done_eps = 0
  with torch.no_grad():
    while done_eps < args.episodes:
      rgb = torch.as_tensor(obs['rgb'], device=dev).permute(0, 3, 1, 2).float() / 255.
      vel = torch.as_tensor(obs['vel'], device=dev)
      pc_l, hd_l = vision(rgb)
      vis_pc, vis_hd = torch.softmax(pc_l, -1), torch.softmax(hd_l, -1)
      mask = (torch.rand(n, 1, device=dev) < 0.05).float()
      g, grid_state = grid.step(vel, vis_pc, vis_hd, mask, grid_state)
      if args.agent == 'grid':
        g_in, goal_in = g, goal_code
      else:
        g_in, goal_in = torch.zeros_like(g), torch.zeros_like(goal_code)
      pi, _, pol_state = policy.step(rgb, g_in, goal_in, prev_action,
                                     prev_reward, pol_state)
      a = torch.distributions.Categorical(logits=pi).sample()

      # Record decoding pairs where the goal is known (post first hit).
      for i in range(n):
        if not np.isnan(goal_pos[i, 0]):
          delta = goal_pos[i] - obs['pos'][i]
          dist = float(np.linalg.norm(delta))
          ang = float(np.arctan2(delta[1], delta[0]))
          H.append(pol_state[0][i].cpu().numpy())
          D.append([dist, np.sin(ang), np.cos(ang)])
          G.append(ep_ids[i])

      prev_pos = obs['pos'].copy()
      obs, rewards, dones = venv.step(a.cpu().numpy())
      obs = encode_vel(obs)
      for i in range(n):
        if rewards[i] > 0:
          # The agent teleports on pickup, so the goal is where it WAS.
          goal_pos[i] = prev_pos[i]
        if dones[i]:
          done_eps += 1
          goal_pos[i] = np.nan
          ep_ids[i] = next_ep
          next_ep += 1
          goal_code[i].zero_()
          grid_state[0][i].zero_(); grid_state[1][i].zero_()
          pol_state[0][i].zero_(); pol_state[1][i].zero_()
      prev_action = a
      # Clipped to match training: prev_reward is a policy input.
      prev_reward = torch.as_tensor(rewards, device=dev).clamp(-1.0, 1.0)

  venv.close()
  H = np.asarray(H); D = np.asarray(D); G = np.asarray(G)
  from sklearn.linear_model import Ridge
  from sklearn.model_selection import GroupShuffleSplit
  res = {'n_samples': int(len(H)), 'n_episodes': int(len(np.unique(G)))}
  if len(H) > 500 and len(np.unique(G)) >= 8:
    # Held-out-episodes split: consecutive steps are strongly correlated,
    # so a random row split would leak.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=0).split(H, D, groups=G))
    Xtr, Xte, ytr, yte = H[tr], H[te], D[tr], D[te]
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    yp = m.predict(Xte)
    ss_res = ((yte - yp) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1 - ss_res / ss_tot
    perm = np.random.RandomState(0).permutation(len(ytr))
    m0 = Ridge(alpha=1.0).fit(Xtr, ytr[perm])
    yp0 = m0.predict(Xte)
    r2_0 = 1 - ((yte - yp0) ** 2).sum(0) / ss_tot
    ang_err = np.arctan2(yte[:, 1], yte[:, 2]) - np.arctan2(yp[:, 1],
                                                            yp[:, 2])
    ang_err = (ang_err + np.pi) % (2 * np.pi) - np.pi
    res.update(
        r2_dist=float(r2[0]), r2_sin=float(r2[1]), r2_cos=float(r2[2]),
        mae_dist_m=float(np.abs(yte[:, 0] - yp[:, 0]).mean()),
        r2_dist_shuffled=float(r2_0[0]),
        dir_err_deg=float(np.rad2deg(np.abs(ang_err)).mean()))
  print(json.dumps(res, indent=1))
  with open(os.path.join(args.out, f'goal_decode_{args.agent}.json'), 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
