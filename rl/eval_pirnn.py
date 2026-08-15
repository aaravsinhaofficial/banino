"""Frozen-policy evaluation of the pirnn agent, with optional unit ablation.

Mirrors eval_agent.py's protocol (stochastic policy, N-episode mean score)
for the --agent pirnn variant of train_rl.py, adding:
  --ablate        comma-separated raw RNN unit indices, or @path to a JSON
                  list, to lesion before evaluation
  --ablation_mode 'full' (source-repo weight surgery: unit silenced in
                  recurrence, input, and readout) or 'readout' (dynamics
                  intact; only the code the policy sees is masked)
Also reports the place-cell position-decoding error of the (possibly
lesioned) RNN state along the behaved trajectories — the mechanism-level
readout matching the source repo's ablation metric.

Usage:
  python3 -m rl.eval_pirnn --run rl_runs/pirnn_fake_s0 \
      --episodes 200 --tag predictive --ablate @units.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.nets import PolicyNet                       # noqa: E402
from rl.pi_rnn import BOX_HALF, PIRNN               # noqa: E402
from rl.vec_env import SubprocVecEnv                # noqa: E402


def parse_units(spec):
  if not spec:
    return None
  if spec.startswith('@'):
    with open(spec[1:]) as f:
      return [int(u) for u in json.load(f)]
  return [int(u) for u in spec.split(',') if u.strip()]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run', required=True,
                  help='train_rl output dir of a --agent pirnn run')
  ap.add_argument('--ckpt', default='final',
                  help="'final', 'latest', or explicit checkpoint path")
  ap.add_argument('--episodes', type=int, default=200)
  ap.add_argument('--n_envs', type=int, default=8)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--base_seed', type=int, default=424242)
  ap.add_argument('--ablate', default=None)
  ap.add_argument('--ablation_mode', choices=['full', 'readout'],
                  default='full')
  ap.add_argument('--tag', default='sham')
  ap.add_argument('--pirnn_ckpt', default=None,
                  help='override the RNN checkpoint recorded in config.json')
  ap.add_argument('--random_policy', action='store_true',
                  help='Chance line: keep the policy at its (seeded) random '
                       'init instead of loading the checkpoint — an untrained '
                       'network explores far better than uniform-random '
                       'actions, so this is the floor scores must beat.')
  ap.add_argument('--out', default=None)
  args = ap.parse_args()

  with open(os.path.join(args.run, 'config.json')) as f:
    cfg = json.load(f)
  assert cfg['agent'] == 'pirnn', f"--run must be a pirnn run, got {cfg['agent']}"

  if args.ckpt == 'final':
    ckpt_path = os.path.join(args.run, 'ckpt_final.pt')
  elif args.ckpt == 'latest':
    cands = sorted(glob.glob(os.path.join(args.run, 'ckpt_*.pt')))
    ckpt_path = cands[-1]
  else:
    ckpt_path = args.ckpt

  dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(args.base_seed)  # reproducible sampling + re-anchors
  units = parse_units(args.ablate)
  pirnn = PIRNN(args.pirnn_ckpt or cfg['pirnn_ckpt'], device=dev,
                ablate_units=units, ablation_mode=args.ablation_mode)
  env_half = cfg.get('pirnn_env_half') or cfg['arena_half_m']
  scale = BOX_HALF / env_half

  ck = torch.load(ckpt_path, map_location=dev)
  n_actions = ck['policy']['pi.weight'].shape[0]  # match the trained head
  policy = PolicyNet(n_actions=n_actions, n_grid=pirnn.Ng).to(dev)
  if args.random_policy:
    ckpt_path = f'untrained(seed {args.base_seed}, arch from {ckpt_path})'
  else:
    policy.load_state_dict(ck['policy'])
  policy.eval()
  reward_clip = cfg.get('reward_clip', 0.0) or 0.0

  n = args.n_envs
  venv = SubprocVecEnv(
      n, cfg['level'], base_seed=args.base_seed, fake=cfg['fake'],
      env_kwargs=dict(blind=cfg.get('fake_blind', 'none')) if cfg['fake']
      else dict(arena_cells=cfg['arena_cells'], cell_m=0.25))
  obs = venv.reset_all()

  pol_state = policy.zero_state(n, dev)
  h_pi = torch.zeros(n, pirnn.Ng, device=dev)
  prev_pos = torch.zeros(n, 2, device=dev)
  reinit = torch.ones(n, dtype=torch.bool, device=dev)
  goal_code = torch.zeros(n, pirnn.Ng, device=dev)
  prev_action = torch.zeros(n, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)
  ep_return = np.zeros(n)
  scores = []
  dec_err_sum, dec_err_n = 0.0, 0

  with torch.no_grad():
    while len(scores) < args.episodes:
      rgb = torch.as_tensor(obs['rgb'], device=dev).permute(
          0, 3, 1, 2).float() / 255.
      pos_t = torch.as_tensor(obs['pos'], device=dev) * scale
      reanchor = torch.rand(n, device=dev) < cfg.get('pirnn_reanchor', 0.05)
      h_pi, _ = pirnn.advance(h_pi, pos_t - prev_pos, reinit | reanchor, pos_t)
      prev_pos = pos_t
      reinit = torch.zeros_like(reinit)
      g = pirnn.g_for_policy(h_pi)
      dec = pirnn.decode_pos(h_pi)
      dec_err_sum += (dec - pos_t).norm(dim=1).sum().item()
      dec_err_n += n
      pi, _, pol_state = policy.step(rgb, g, goal_code, prev_action,
                                     prev_reward, pol_state)
      a = torch.distributions.Categorical(logits=pi).sample()
      obs, rewards, dones = venv.step(a.cpu().numpy())
      r_raw = torch.as_tensor(rewards, device=dev)
      # prev_reward is a policy input: clip to the scale it was trained on.
      r_t = r_raw.clamp(-reward_clip, reward_clip) if reward_clip > 0 \
          else r_raw
      d_t = torch.as_tensor(dones.astype(np.float32), device=dev)
      hit = (r_raw > 0).float().unsqueeze(1)
      goal_code = hit * g.detach() + (1 - hit) * goal_code
      reinit = (r_raw > 0) | (d_t > 0)
      ep_return += rewards
      for i in range(n):
        if dones[i]:
          scores.append(float(ep_return[i]))
          ep_return[i] = 0.0
          goal_code[i].zero_()
          pol_state[0][i].zero_(); pol_state[1][i].zero_()
      prev_action = a
      prev_reward = r_t

  venv.close()
  scores = np.array(scores[:args.episodes])
  res = dict(tag=args.tag,
             mean_score=float(scores.mean()), std=float(scores.std()),
             sem=float(scores.std() / np.sqrt(len(scores))),
             n_episodes=int(len(scores)),
             scores=[float(s) for s in scores],
             decode_err_cm=round(100.0 * dec_err_sum / max(dec_err_n, 1), 3),
             n_ablated=0 if units is None else len(units),
             ablated_units=units, ablation_mode=args.ablation_mode,
             ckpt=ckpt_path, base_seed=args.base_seed)
  print(json.dumps({k: v for k, v in res.items()
                    if k not in ('scores', 'ablated_units')}))
  out = args.out or os.path.join(args.run, 'ablation_evals',
                                 f'{args.tag}.json')
  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, 'w') as f:
    json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
