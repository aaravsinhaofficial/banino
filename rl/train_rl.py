"""Scaled-down RL reproduction: A2C grid-cell agent on DeepMind Lab.

Documented deviations from the paper (which used async A3C, 32 threads,
1e9 env steps): synchronous batched A2C over N parallel envs, interleaved
(not threaded) vision/grid trainers, 2M-frame replay, and far fewer env
steps — pass --frames to set the budget. Baselines: --agent grid (full),
--agent placecell (policy sees the vision module's place/HD code instead
of grid/goal codes), --agent a3c (no grid inputs at all).

Run inside the dmlab container (or --fake anywhere):
  python3 -m rl.train_rl --level contributed/dmlab30/explore_goal_locations_small \
      --out rl_runs/grid1 --agent grid --frames 20000000
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modern import targets as targets_lib          # noqa: E402
from rl import agent as agent_lib                  # noqa: E402
from rl.nets import GridModule, PolicyNet, VisionCNN  # noqa: E402
from rl.vec_env import SubprocVecEnv               # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--level',
                  default='contributed/dmlab30/explore_goal_locations_small')
  ap.add_argument('--out', required=True)
  ap.add_argument('--agent', choices=['grid', 'placecell', 'a3c'],
                  default='grid')
  ap.add_argument('--frames', type=int, default=20_000_000)
  ap.add_argument('--n_envs', type=int, default=32)
  ap.add_argument('--rollout', type=int, default=100)
  ap.add_argument('--action_repeat', type=int, default=4)
  # Defaults follow Supplementary Table 2 (SI, Banino-2018-SI.pdf): lr
  # sampled in [1e-6, 2e-4], entropy in [6e-5, 1e-4], baseline cost ~0.5,
  # discount 0.99, actor-critic BPTT 100 steps, grid-network lr 1e-3 with
  # L2 1e-4 (NOT the supervised 1e-5/clip recipe).
  ap.add_argument('--gamma', type=float, default=0.99)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--entropy', type=float, default=8e-5)
  ap.add_argument('--value_coef', type=float, default=0.5)
  ap.add_argument('--grid_lr', type=float, default=1e-3)
  ap.add_argument('--grid_l2', type=float, default=1e-4)
  ap.add_argument('--grid_clip', type=float, default=1e-3,
                  help='Elementwise grad clip for the grid trainer; 0 = off. '
                       'SI Table 2 lists no clip, but the unclipped 1e-3 '
                       'recipe destabilised in 3 of 4 grid trainers around '
                       'update ~14k; this default keeps the supervised '
                       "recipe's clip/lr ratio at the agent's lr.")
  ap.add_argument('--arena_half_m', type=float, default=1.375)
  ap.add_argument('--replay_per_env', type=int, default=62_500)
  ap.add_argument('--vis_updates', type=int, default=8)
  ap.add_argument('--grid_updates', type=int, default=2)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--fake', action='store_true')
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--resume', action='store_true',
                  help='Resume from <out>/resume.pt if present (models, '
                       'optimizers, frame/update counters; replay restarts '
                       'warm).')
  ap.add_argument('--reset_grid', action='store_true',
                  help='One-shot recovery: on resume, keep policy/vision but '
                       'reinitialize the grid module and its optimizer '
                       '(collapsed grid trainer). The next resume.pt save '
                       'persists the reset, so the flag is only needed once.')
  ap.add_argument('--arena_cells', type=int, default=11,
                  help='Maze frame size in cells (13 for square_arena_goal); '
                       'sets the position-centring offset in the env.')
  ap.add_argument('--pc_scale', type=float, default=0.1,
                  help='Place-cell scale for the agent ensembles (m). SI '
                       'Table 2 lists 40 engine units = 0.1 m.')
  ap.add_argument('--init_grid_from', default=None,
                  help='Supervised GridNetwork checkpoint to warm-start the '
                       'grid module from (velocity columns + recurrent '
                       'weights + bottleneck + heads; visual-code input '
                       'columns start at zero).')
  ap.add_argument('--vel_encoding', choices=['raw', 'supervised'],
                  default='raw',
                  help="'supervised' re-encodes env velocity as (ground "
                       "speed, sin/cos of per-step heading change) — the "
                       "convention the supervised checkpoint was trained "
                       "on. Use together with --init_grid_from.")
  args = ap.parse_args()

  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=1)
  dev = torch.device(args.device)
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)

  n = args.n_envs
  venv = SubprocVecEnv(n, args.level, base_seed=args.seed * 1000,
                       fake=args.fake,
                       env_kwargs=dict(arena_cells=args.arena_cells,
                                       cell_m=0.25) if not args.fake else None)

  step_dt = args.action_repeat / 60.0  # seconds per decision step

  def encode_vel(obs):
    """Optionally re-encode velocity in place, BEFORE replay ingestion, so
    acting, the replay-trained grid module, and offline decoding all see
    one convention."""
    if args.vel_encoding == 'supervised':
      v = obs['vel']
      dtheta = v[:, 2] * step_dt
      obs['vel'] = np.stack([np.hypot(v[:, 0], v[:, 1]),
                             np.sin(dtheta), np.cos(dtheta)],
                            axis=1).astype(np.float32)
    return obs

  obs = encode_vel(venv.reset_all())

  pc_ens = targets_lib.PlaceCellEnsemble(
      stdev=args.pc_scale,
      pos_min=-args.arena_half_m, pos_max=args.arena_half_m, device=dev)
  hd_ens = targets_lib.HeadDirectionCellEnsemble(device=dev)

  vision = VisionCNN().to(dev)
  grid = GridModule().to(dev)
  policy = PolicyNet().to(dev)
  if args.init_grid_from:
    sd = torch.load(args.init_grid_from, map_location='cpu')
    new = grid.state_dict()
    w_ih = torch.zeros_like(new['cell.weight_ih'])
    w_ih[:, :3] = sd['lstm.weight_ih_l0']   # velocity columns transfer;
    new['cell.weight_ih'] = w_ih            # visual-code columns start at 0
    new['cell.weight_hh'] = sd['lstm.weight_hh_l0']
    new['cell.bias_ih'] = sd['lstm.bias_ih_l0']
    new['cell.bias_hh'] = sd['lstm.bias_hh_l0']
    new['bottleneck.weight'] = sd['bottleneck.weight']
    for k in ('pc_logits.weight', 'pc_logits.bias',
              'hd_logits.weight', 'hd_logits.bias'):
      new[k] = sd[k]
    grid.load_state_dict(new)
    print(f'warm-started grid module from {args.init_grid_from}', flush=True)
  vis_opt = torch.optim.Adam(vision.parameters(), lr=1e-4)
  grid_opt = torch.optim.RMSprop(grid.parameters(), lr=args.grid_lr,
                                 momentum=0.9, alpha=0.9, eps=1e-10)
  pol_opt = torch.optim.RMSprop(policy.parameters(), lr=args.lr, alpha=0.99,
                                eps=0.1)
  replay = agent_lib.Replay(n, args.replay_per_env)

  pol_state = policy.zero_state(n, dev)
  grid_state = grid.zero_state(n, dev)
  goal_code = torch.zeros(n, 512, device=dev)
  prev_action = torch.zeros(n, dtype=torch.long, device=dev)
  prev_reward = torch.zeros(n, device=dev)
  episode_ids = np.arange(n, dtype=np.int64)
  next_episode = n
  ep_return = np.zeros(n)
  returns_hist = []
  gp_buf = []  # rolling (grid_code, pos, hd) for ratemap dumps

  def to_rgb(o):
    return torch.as_tensor(o['rgb'], device=dev).permute(0, 3, 1, 2).float() / 255.

  frames_done = 0
  update = 0
  resume_path = os.path.join(args.out, 'resume.pt')
  if args.resume and os.path.exists(resume_path):
    st = torch.load(resume_path, map_location=dev)
    vision.load_state_dict(st['vision'])
    policy.load_state_dict(st['policy'])
    vis_opt.load_state_dict(st['vis_opt'])
    pol_opt.load_state_dict(st['pol_opt'])
    if args.reset_grid:
      print('reset_grid: grid module + optimizer reinitialized', flush=True)
    else:
      grid.load_state_dict(st['grid'])
      grid_opt.load_state_dict(st['grid_opt'])
    frames_done = st['frames_done']
    update = st['update']
    next_episode = st['next_episode']
    print(f'resumed at frames={frames_done} update={update}', flush=True)

  def save_resume():
    tmp = resume_path + '.tmp'
    torch.save(dict(vision=vision.state_dict(), grid=grid.state_dict(),
                    policy=policy.state_dict(),
                    vis_opt=vis_opt.state_dict(),
                    grid_opt=grid_opt.state_dict(),
                    pol_opt=pol_opt.state_dict(),
                    frames_done=frames_done, update=update,
                    next_episode=int(next_episode)), tmp)
    os.replace(tmp, resume_path)

  t0 = time.time()
  metrics_f = open(os.path.join(args.out, 'metrics.jsonl'), 'a')

  while frames_done < args.frames:
    # ---------------- rollout ----------------
    roll = {k: [] for k in
            ('rgb', 'grid', 'goal', 'pa', 'pr', 'actions', 'rewards',
             'dones', 'masks_reset')}
    pol_state_start = (pol_state[0].detach(), pol_state[1].detach())
    for _ in range(args.rollout):
      rgb = to_rgb(obs)
      vel = torch.as_tensor(obs['vel'], device=dev)
      with torch.no_grad():
        pc_l, hd_l = vision(rgb)
        vis_pc = torch.softmax(pc_l, -1)
        vis_hd = torch.softmax(hd_l, -1)
        mask = (torch.rand(n, 1, device=dev) < 0.05).float()
        g, grid_state = grid.step(vel, vis_pc, vis_hd, mask, grid_state)
        if args.agent == 'grid':
          g_in, goal_in = g, goal_code
        elif args.agent == 'placecell':
          g_in = torch.cat([vis_pc, vis_hd,
                            torch.zeros(n, 512 - 268, device=dev)], -1)
          goal_in = torch.zeros_like(goal_code)
        else:
          g_in = torch.zeros_like(g)
          goal_in = torch.zeros_like(goal_code)
        pi, v, pol_state = policy.step(rgb, g_in, goal_in, prev_action,
                                       prev_reward, pol_state)
        a = torch.distributions.Categorical(logits=pi).sample()

      replay.add(obs, episode_ids)
      gp_buf.append((g.detach().cpu().numpy().astype(np.float16),
                     obs['pos'].copy(), obs['hd'].copy()))
      if len(gp_buf) > 2000:
        gp_buf.pop(0)

      roll['rgb'].append(rgb)
      roll['grid'].append(g_in)
      roll['goal'].append(goal_in)
      roll['pa'].append(prev_action)
      roll['pr'].append(prev_reward)
      roll['actions'].append(a)

      obs, rewards, dones = venv.step(a.cpu().numpy())
      obs = encode_vel(obs)
      frames_done += n * args.action_repeat  # env workers step with repeat 4
      r_t = torch.as_tensor(rewards, device=dev)
      d_t = torch.as_tensor(dones.astype(np.float32), device=dev)
      roll['rewards'].append(r_t)
      roll['dones'].append(d_t)

      # Goal-code capture: store the grid code at reward pickup.
      hit = (r_t > 0).float().unsqueeze(1)
      goal_code = hit * g.detach() + (1 - hit) * goal_code

      ep_return += rewards
      for i, d in enumerate(dones):
        if d:
          returns_hist.append(float(ep_return[i]))
          ep_return[i] = 0.0
          episode_ids[i] = next_episode
          next_episode += 1
          goal_code[i].zero_()
          grid_state[0][i].zero_(); grid_state[1][i].zero_()
          pol_state[0][i].zero_(); pol_state[1][i].zero_()
      prev_action = a
      prev_reward = r_t

    # ---------------- A2C update (recompute with grad) ----------------
    with torch.no_grad():
      rgb = to_rgb(obs)
      pc_l, hd_l = vision(rgb)
      _, v_boot, _ = policy.step(
          rgb, roll['grid'][-1], roll['goal'][-1], prev_action, prev_reward,
          (pol_state[0].detach(), pol_state[1].detach()))
    T = args.rollout
    returns = torch.zeros(T, n, device=dev)
    R = v_boot
    for t in reversed(range(T)):
      R = roll['rewards'][t] + args.gamma * (1 - roll['dones'][t]) * R
      returns[t] = R

    st = pol_state_start
    pi_list, v_list = [], []
    for t in range(T):
      pi, v, st = policy.step(roll['rgb'][t], roll['grid'][t],
                              roll['goal'][t], roll['pa'][t], roll['pr'][t],
                              st)
      pi_list.append(pi)
      v_list.append(v)
      st = (st[0] * (1 - roll['dones'][t]).unsqueeze(1),
            st[1] * (1 - roll['dones'][t]).unsqueeze(1))
    pi_all = torch.stack(pi_list)          # [T,n,A]
    v_all = torch.stack(v_list)            # [T,n]
    a_all = torch.stack(roll['actions'])   # [T,n]
    dist = torch.distributions.Categorical(logits=pi_all)
    adv = (returns - v_all).detach()
    pol_loss = -(dist.log_prob(a_all) * adv).mean()
    val_loss = 0.5 * (returns.detach() - v_all).pow(2).mean()
    ent = dist.entropy().mean()
    loss = pol_loss + args.value_coef * val_loss - args.entropy * ent
    pol_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 40.0)
    pol_opt.step()
    pol_state = (pol_state[0].detach(), pol_state[1].detach())

    # ---------------- supervised trainers ----------------
    vl = gl = None
    if replay.count * n > 50_000:
      for _ in range(args.vis_updates):
        vl = agent_lib.vision_update(vision, vis_opt, replay, pc_ens, hd_ens,
                                     rng, dev)
      for _ in range(args.grid_updates):
        gl = agent_lib.grid_update(grid, vision, grid_opt, replay, pc_ens,
                                   hd_ens, rng, dev, l2=args.grid_l2,
                                   clip=args.grid_clip)

    update += 1
    if update % 50 == 0:
      fps = frames_done / (time.time() - t0)
      avg_ret = float(np.mean(returns_hist[-50:])) if returns_hist else 0.0
      rec = dict(update=update, frames=frames_done, fps=round(fps),
                 avg_return_50=round(avg_ret, 2), ent=round(ent.item(), 3),
                 vis_loss=vl and round(vl, 4), grid_loss=gl and round(gl, 4))
      print(json.dumps(rec), flush=True)
      metrics_f.write(json.dumps(rec) + '\n')
      metrics_f.flush()
    if update % 500 == 0:
      save_resume()
    if update % 2000 == 0:
      torch.save({'vision': vision.state_dict(), 'grid': grid.state_dict(),
                  'policy': policy.state_dict()},
                 os.path.join(args.out, f'ckpt_{frames_done:09d}.pt'))
      if gp_buf:
        gs = np.concatenate([b[0] for b in gp_buf])
        ps = np.concatenate([b[1] for b in gp_buf])
        hs = np.concatenate([b[2] for b in gp_buf])
        np.savez_compressed(
            os.path.join(args.out, f'gridcodes_{frames_done:09d}.npz'),
            g=gs, pos=ps, hd=hs)

  save_resume()
  torch.save({'vision': vision.state_dict(), 'grid': grid.state_dict(),
              'policy': policy.state_dict()},
             os.path.join(args.out, 'ckpt_final.pt'))
  venv.close()


if __name__ == '__main__':
  main()
