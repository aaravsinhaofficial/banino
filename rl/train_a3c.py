"""True async A3C for the Banino et al. 2018 RL reproduction.

The synchronous A2C in rl/train_rl.py was this repo's one architectural
liberty against the paper; this trainer removes it. Faithful to the
Methods/SI:

  * N actor processes (paper: 32 "threads"), each owning ONE DeepMind Lab
    instance, acting continuously and applying Hogwild gradient updates to
    a policy net held in shared memory (no locks, no barriers).
  * Shared-statistics RMSProp (SI Table 2: "shared RMSProp algorithm"):
    the square-average statistics live in shared memory next to the
    parameters, so all workers accumulate into one optimizer state (Mnih
    et al. 2016).  Update rule and constants match the sync run's
    torch.optim.RMSprop (alpha 0.99, eps 0.1 added after sqrt, no
    momentum).
  * BPTT 100 with n-step returns over the same window, entropy 8e-5,
    baseline cost 0.5, discount 0.99, action repeat 4, grad-norm clip 40.
  * The vision and grid-network supervised learners run in TWO dedicated
    processes, asynchronously with acting (as in the paper), training the
    shared modules in place from a shared-memory replay of recent
    experience that the actors write lock-free (each actor owns one row of
    the ring buffer).  Update pacing reproduces the sync run's ratios:
    one vision update per 1600 env frames and one grid update per 6400
    (there: 8 and 2 per 12800-frame cycle); achieved rates are logged so
    any shortfall against the pace target is visible in metrics.jsonl.

Per-step computation (vision code, 5% visual-code mask, grid code, goal
code captured at reward, input wiring per agent type) matches
rl/train_rl.py, batch size 1; local model copies re-sync from the shared
ones after every rollout.  Two mismatches with the sync trainer are
A3C-inherent and intended: ~N x more policy updates per frame at the same
per-update magnitude (the regime the SI's learning-rate range was sampled
for), and Hogwild update races.

Run (inside the dmlab container):
  python3 -m rl.train_a3c --level contributed/dmlab30/explore_goal_locations_small \
      --out rl_runs/a3c_goal_grid --agent grid --workers 32 --frames 1000000000
"""

import argparse
import collections
import json
import os
import queue as queue_lib
import sys
import time
import traceback

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl import agent as agent_lib                     # noqa: E402
from rl.nets import GridModule, PolicyNet, VisionCNN  # noqa: E402


class SharedRMSprop:
  """A3C shared-statistics RMSProp over shared-memory params.

  Same math as torch.optim.RMSprop(momentum=0, centered=False):
    sq   <- alpha * sq + (1 - alpha) * g^2
    p    <- p - lr * g / (sqrt(sq) + eps)
  step() takes the gradient tensors explicitly (they come from a local
  model replica), so no .grad plumbing on the shared params is needed.
  """

  def __init__(self, params, lr, alpha=0.99, eps=0.1, square_avgs=None):
    self.params = list(params)
    self.lr, self.alpha, self.eps = lr, alpha, eps
    if square_avgs is None:
      square_avgs = [torch.zeros_like(p).share_memory_() for p in self.params]
    self.square_avgs = square_avgs

  @torch.no_grad()
  def step(self, grads):
    for p, g, sq in zip(self.params, grads, self.square_avgs):
      if g is None:
        continue
      sq.mul_(self.alpha).addcmul_(g, g, value=1.0 - self.alpha)
      p.addcdiv_(g, sq.sqrt().add_(self.eps), value=-self.lr)


class SharedReplayView:
  """rl.agent.Replay's sampling API over actor-owned shared ring buffers.

  Row i is written only by actor i; this reader tolerates the benign races
  (a torn window is rejected by the episode-id check, a torn frame is one
  noisy training sample)."""

  def __init__(self, frames, vels, poss, hds, eps, ptrs, counts):
    self.frames, self.vels, self.poss, self.hds = frames, vels, poss, hds
    self.eps, self.ptrs, self.counts = eps, ptrs, counts
    self.n, self.cap = frames.shape[0], frames.shape[1]

  def total_steps(self):
    return int(self.counts.sum())

  def sample_frames(self, batch, rng):
    es, ts = [], []
    while len(es) < batch:
      e = int(rng.integers(0, self.n))
      c = int(self.counts[e])
      if c == 0:
        continue
      es.append(e)
      ts.append(int(rng.integers(0, c)))
    es, ts = np.array(es), np.array(ts)
    return self.frames[es, ts], self.poss[es, ts], self.hds[es, ts]

  def sample_sequences(self, batch, seqlen, rng, max_tries=200):
    out = []
    tries = 0
    while len(out) < batch and tries < max_tries * batch:
      tries += 1
      e = int(rng.integers(0, self.n))
      c, p = int(self.counts[e]), int(self.ptrs[e])
      if c < self.cap:
        if c <= seqlen:
          continue
        s = int(rng.integers(0, c - seqlen))
      else:
        s = int(rng.integers(0, self.cap - seqlen))
        if s < p <= s + seqlen:  # window must not straddle the write head
          continue
      ep = self.eps[e, s:s + seqlen]
      if ep[0] < 0 or not (ep == ep[0]).all():
        continue
      out.append((e, s))
    gather = lambda arr: np.stack(  # noqa: E731
        [arr[e, s:s + seqlen] for e, s in out])
    if not out:
      return (np.zeros((0,)),) * 4
    return (gather(self.frames), gather(self.vels), gather(self.poss),
            gather(self.hds))


def _pull(dst, src):
  """Copy shared params into a local replica (no buffers in these nets)."""
  with torch.no_grad():
    for d, s in zip(dst.parameters(), src.parameters()):
      d.copy_(s)


def _make_env(cfg, rank):
  if cfg.get('bandit'):
    from rl.bandit_env import BanditEnv as Env
  elif cfg['fake']:
    from rl.fake_env import FakeEnv as Env
  else:
    from rl.env import LabEnv as Env
  return Env(cfg['level'], cfg['seed'] * 1000 + rank,
             cell_m=0.25, arena_cells=cfg['arena_cells'])


def _encode_vel1(v, cfg):
  """Single-obs velocity re-encoding (parity with train_rl.encode_vel)."""
  if cfg['vel_encoding'] != 'supervised':
    return v
  step_dt = cfg['action_repeat'] / 60.0
  dtheta = v[2] * step_dt
  return np.array([np.hypot(v[0], v[1]), np.sin(dtheta), np.cos(dtheta)],
                  dtype=np.float32)


def actor_proc(rank, cfg, shared, ctrl):
  """One async actor: own env, local nets, Hogwild updates to the shared
  policy via shared-statistics RMSProp."""
  try:
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'] * 100003 + rank)
    q = ctrl['queue']
    env = _make_env(cfg, rank)

    vision_l, grid_l, policy_l = VisionCNN(), GridModule(), PolicyNet()
    grid_l.train()  # dropout active while acting, as in the sync trainer
    opt = SharedRMSprop(shared['policy'].parameters(), lr=cfg['lr'],
                        square_avgs=shared['pol_sq'])

    frames_np = shared['frames'].numpy()
    vels_np = shared['vels'].numpy()
    poss_np = shared['poss'].numpy()
    hds_np = shared['hds'].numpy()
    eps_np = shared['eps'].numpy()
    ptrs_np = shared['ptrs'].numpy()
    counts_np = shared['counts'].numpy()
    cap = frames_np.shape[1]

    obs = env.reset()
    obs = dict(obs)
    obs['vel'] = _encode_vel1(obs['vel'], cfg)
    pol_state = ([torch.zeros(1, policy_l.nh)] * 2)
    grid_state = ([torch.zeros(1, grid_l.nh_lstm)] * 2)
    goal_code = torch.zeros(1, 512)
    prev_action = torch.zeros(1, dtype=torch.long)
    prev_reward = torch.zeros(1)
    ep_return = 0.0
    episode_id = 0
    ptr = 0
    count = 0
    T = cfg['rollout']
    gamma, rep = cfg['gamma'], cfg['action_repeat']
    step_i = 0

    while not ctrl['stop'].value:
      _pull(vision_l, shared['vision'])
      _pull(grid_l, shared['grid'])
      _pull(policy_l, shared['policy'])
      pol_state = (pol_state[0].detach(), pol_state[1].detach())

      pis, vs, acts, rews, dones = [], [], [], [], []
      g_in = goal_in = None
      for _ in range(T):
        rgb = torch.as_tensor(obs['rgb']).permute(2, 0, 1).float().div_(255.)
        rgb = rgb.unsqueeze(0)
        vel = torch.as_tensor(obs['vel']).unsqueeze(0)
        with torch.no_grad():
          pc_l, hd_l = vision_l(rgb)
          vis_pc = torch.softmax(pc_l, -1)
          vis_hd = torch.softmax(hd_l, -1)
          mask = (torch.rand(1, 1) < 0.05).float()
          g, grid_state = grid_l.step(vel, vis_pc, vis_hd, mask, grid_state)
        if cfg['agent'] == 'grid':
          g_in, goal_in = g, goal_code
        elif cfg['agent'] == 'placecell':
          g_in = torch.cat([vis_pc, vis_hd, torch.zeros(1, 512 - 268)], -1)
          goal_in = torch.zeros_like(goal_code)
        else:
          g_in = torch.zeros_like(g)
          goal_in = torch.zeros_like(goal_code)
        pi, v, pol_state = policy_l.step(rgb, g_in, goal_in, prev_action,
                                         prev_reward, pol_state)
        a = torch.distributions.Categorical(logits=pi.detach()).sample()

        frames_np[rank, ptr] = obs['rgb']
        vels_np[rank, ptr] = obs['vel']
        poss_np[rank, ptr] = obs['pos']
        hds_np[rank, ptr] = obs['hd']
        eps_np[rank, ptr] = episode_id
        ptr = (ptr + 1) % cap
        count = min(count + 1, cap)
        ptrs_np[rank] = ptr
        counts_np[rank] = count

        step_i += 1
        if step_i % cfg['gp_every'] == 0:
          try:
            q.put_nowait(('gp', g.detach().numpy().astype(np.float16)[0],
                          obs['pos'].copy(), np.float32(obs['hd'])))
          except queue_lib.Full:
            pass

        o, reward, done = env.step(int(a.item()), rep)
        obs = dict(o)
        obs['vel'] = _encode_vel1(obs['vel'], cfg)
        pis.append(pi)
        vs.append(v)
        acts.append(a)
        rews.append(float(reward))
        dones.append(bool(done))
        if reward > 0:
          goal_code = g.detach().clone()
        ep_return += reward
        if done:
          try:
            q.put(('ep', rank, float(ep_return)), timeout=0.5)
          except queue_lib.Full:
            pass
          ep_return = 0.0
          episode_id += 1
          obs = dict(env.reset())
          obs['vel'] = _encode_vel1(obs['vel'], cfg)
          goal_code = torch.zeros(1, 512)
          grid_state = ([torch.zeros(1, grid_l.nh_lstm)] * 2)
          pol_state = ([torch.zeros(1, policy_l.nh)] * 2)
        prev_action = a
        prev_reward = torch.tensor([rews[-1]], dtype=torch.float32)

      # Bootstrap with the last step's grid/goal inputs (parity with the
      # sync trainer, which reused roll['grid'][-1] for v(s_T)).
      with torch.no_grad():
        rgb_T = torch.as_tensor(obs['rgb']).permute(2, 0, 1).float()
        rgb_T = rgb_T.div_(255.).unsqueeze(0)
        _, v_boot, _ = policy_l.step(
            rgb_T, g_in.detach(), goal_in.detach(), prev_action, prev_reward,
            (pol_state[0].detach(), pol_state[1].detach()))

      R = v_boot.item()
      returns = [0.0] * T
      for t in reversed(range(T)):
        R = rews[t] + gamma * (0.0 if dones[t] else 1.0) * R
        returns[t] = R
      pi_all = torch.cat(pis)                       # [T, A]
      v_all = torch.cat(vs)                         # [T]
      a_all = torch.cat(acts)                       # [T]
      ret_t = torch.tensor(returns, dtype=torch.float32)
      dist = torch.distributions.Categorical(logits=pi_all)
      adv = (ret_t - v_all).detach()
      # Sum over the rollout, as in Mnih et al. 2016 (gradients are
      # ACCUMULATED over the BPTT window, not averaged). SI Table 2's
      # constants are written for this scaling: entropy 8e-5 summed over
      # 100 steps is a standard ~8e-3 per-step bonus, and baseline cost
      # 0.5 is the usual A3C value. Averaging instead divides every
      # gradient by 100, which is why the policy never left uniform.
      logp = dist.log_prob(a_all)
      pol_loss = -(logp * adv).sum()
      val_loss = 0.5 * (ret_t - v_all).pow(2).sum()
      ent_sum = dist.entropy().sum()
      ent = ent_sum / T                     # per-step, for logging
      loss = (pol_loss + cfg['value_coef'] * val_loss
              - cfg['entropy'] * ent_sum)
      for p in policy_l.parameters():
        p.grad = None
      loss.backward()
      torch.nn.utils.clip_grad_norm_(policy_l.parameters(), 40.0)
      opt.step([p.grad for p in policy_l.parameters()])

      with ctrl['frames'].get_lock():
        ctrl['frames'].value += T * rep
      try:
        q.put_nowait(('stat', rank, float(ent.item()),
                      float(v_all.detach().mean())))
      except queue_lib.Full:
        pass
    env.close()
  except Exception:
    try:
      ctrl['queue'].put(('died', rank, traceback.format_exc()), timeout=1)
    except Exception:
      pass
    raise


def trainer_proc(cfg, shared, ctrl, kind):
  """Async supervised learner ('vision' or 'grid') for one shared module,
  paced against the global frame counter to match the sync run's
  updates-per-frame.  The grid learner reads the shared vision module for
  visual codes (concurrently updated by the vision learner; benign)."""
  try:
    torch.set_num_threads(cfg['trainer_threads'])
    torch.manual_seed(cfg['seed'] * 7919 + (1 if kind == 'grid' else 0))
    rng = np.random.default_rng(cfg['seed'] + (999 if kind == 'vision'
                                               else 1999))
    q = ctrl['queue']
    from modern import targets as targets_lib
    pc_ens = targets_lib.PlaceCellEnsemble(
        stdev=cfg['pc_scale'], pos_min=-cfg['arena_half_m'],
        pos_max=cfg['arena_half_m'], device='cpu')
    hd_ens = targets_lib.HeadDirectionCellEnsemble(device='cpu')

    vision, grid = shared['vision'], shared['grid']  # trained in place
    if kind == 'vision':
      opt = torch.optim.Adam(vision.parameters(), lr=1e-4)
      per_frames = cfg['vis_per_frames']
    else:
      opt = torch.optim.RMSprop(grid.parameters(), lr=cfg['grid_lr'],
                                momentum=0.9, alpha=0.9, eps=1e-10)
      per_frames = cfg['grid_per_frames']
    replay = SharedReplayView(
        shared['frames'].numpy(), shared['vels'].numpy(),
        shared['poss'].numpy(), shared['hds'].numpy(), shared['eps'].numpy(),
        shared['ptrs'].numpy(), shared['counts'].numpy())

    state_path = os.path.join(cfg['out'], f'{kind}_trainer_state.pt')
    done = 0
    if cfg['resume'] and os.path.exists(state_path):
      if kind == 'grid' and cfg['reset_grid']:
        print('grid trainer: reset_grid -> fresh optimizer', flush=True)
      else:
        st = torch.load(state_path, map_location='cpu')
        opt.load_state_dict(st['opt'])
        done = st['done']
        print(f'{kind} trainer resumed: {done} updates', flush=True)

    loss = None
    last_save = last_stat = time.time()
    while not ctrl['stop'].value:
      if replay.total_steps() < 50_000:
        time.sleep(1.0)
        continue
      if done < ctrl['frames'].value // per_frames:
        if kind == 'vision':
          loss = agent_lib.vision_update(vision, opt, replay, pc_ens, hd_ens,
                                         rng, 'cpu')
        else:
          loss = agent_lib.grid_update(grid, vision, opt, replay, pc_ens,
                                       hd_ens, rng, 'cpu', l2=cfg['grid_l2'],
                                       clip=cfg['grid_clip'])
        done += 1
      else:
        time.sleep(0.05)
      now = time.time()
      if now - last_stat > 30:
        last_stat = now
        try:
          q.put_nowait(('trainer', kind, loss, done))
        except queue_lib.Full:
          pass
      if now - last_save > 600:
        last_save = now
        tmp = state_path + '.tmp'
        torch.save(dict(opt=opt.state_dict(), done=done), tmp)
        os.replace(tmp, state_path)
    torch.save(dict(opt=opt.state_dict(), done=done), state_path)
  except Exception:
    try:
      ctrl['queue'].put(('died', -1 if kind == 'vision' else -2,
                         traceback.format_exc()), timeout=1)
    except Exception:
      pass
    raise


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--level',
                  default='contributed/dmlab30/explore_goal_locations_small')
  ap.add_argument('--out', required=True)
  ap.add_argument('--agent', choices=['grid', 'placecell', 'a3c'],
                  default='grid')
  ap.add_argument('--frames', type=int, default=1_000_000_000)
  ap.add_argument('--workers', type=int, default=32,
                  help='Async actor processes (paper: 32).')
  ap.add_argument('--rollout', type=int, default=100)   # SI: BPTT 100
  ap.add_argument('--action_repeat', type=int, default=4)
  ap.add_argument('--gamma', type=float, default=0.99)
  ap.add_argument('--lr', type=float, default=1e-4)
  ap.add_argument('--entropy', type=float, default=8e-5)
  ap.add_argument('--value_coef', type=float, default=0.5)
  ap.add_argument('--grid_lr', type=float, default=1e-3)
  ap.add_argument('--grid_l2', type=float, default=1e-4)
  ap.add_argument('--grid_clip', type=float, default=1e-3)
  ap.add_argument('--arena_half_m', type=float, default=1.375)
  ap.add_argument('--arena_cells', type=int, default=11)
  ap.add_argument('--pc_scale', type=float, default=0.1)
  ap.add_argument('--vel_encoding', choices=['raw', 'supervised'],
                  default='raw')
  ap.add_argument('--replay_per_env', type=int, default=0,
                  help='Replay steps per actor; 0 = auto, so the total is '
                       '--replay_total steps whatever the worker count.')
  ap.add_argument('--replay_total', type=int, default=1_500_000,
                  help='Total replay steps across actors (frames dominate: '
                       '84*84*3 B/step, so 1.5M steps ~= 31.8 GB of shared '
                       'memory — /dev/shm must be sized for it).')
  ap.add_argument('--vis_per_frames', type=int, default=1600,
                  help='One vision update per this many env frames '
                       '(sync run: 8 per 12800-frame cycle).')
  ap.add_argument('--grid_per_frames', type=int, default=6400,
                  help='One grid update per this many env frames '
                       '(sync run: 2 per 12800-frame cycle).')
  ap.add_argument('--trainer_threads', type=int, default=6,
                  help='Torch threads per supervised-learner process.')
  ap.add_argument('--gp_every', type=int, default=32,
                  help='Sample (grid code, pos, hd) every k-th step for the '
                       'ratemap dumps.')
  ap.add_argument('--max_seconds', type=int, default=0,
                  help='Wall-clock budget for THIS session (a relaunch '
                       'passes the time remaining to its deadline); '
                       '0 = frames budget only.')
  ap.add_argument('--seed', type=int, default=1)
  ap.add_argument('--fake', action='store_true')
  ap.add_argument('--bandit', action='store_true',
                  help='Dense-reward probe env (rl/bandit_env.py): a unit '
                       'test that the policy-gradient path can learn at all.')
  ap.add_argument('--resume', action='store_true')
  ap.add_argument('--reset_grid', action='store_true')
  args = ap.parse_args()
  if args.replay_per_env <= 0:
    args.replay_per_env = max(10_000, args.replay_total // args.workers)

  os.makedirs(args.out, exist_ok=True)
  with open(os.path.join(args.out, 'config.json'), 'w') as f:
    json.dump(vars(args), f, indent=1)
  torch.manual_seed(args.seed)
  mp.set_sharing_strategy('file_system')
  ctx = mp.get_context('spawn')

  vision = VisionCNN()
  grid = GridModule()
  policy = PolicyNet()
  for m in (vision, grid, policy):
    m.share_memory()
  pol_sq = [torch.zeros_like(p).share_memory_() for p in policy.parameters()]

  W, cap = args.workers, args.replay_per_env
  shared = dict(
      vision=vision, grid=grid, policy=policy, pol_sq=pol_sq,
      frames=torch.zeros(W, cap, 84, 84, 3, dtype=torch.uint8).share_memory_(),
      vels=torch.zeros(W, cap, 3).share_memory_(),
      poss=torch.zeros(W, cap, 2).share_memory_(),
      hds=torch.zeros(W, cap).share_memory_(),
      eps=torch.full((W, cap), -1, dtype=torch.int64).share_memory_(),
      ptrs=torch.zeros(W, dtype=torch.int64).share_memory_(),
      counts=torch.zeros(W, dtype=torch.int64).share_memory_(),
  )

  frames_done = 0
  wall_prev = 0.0
  resume_path = os.path.join(args.out, 'resume.pt')
  if args.resume and os.path.exists(resume_path):
    st = torch.load(resume_path, map_location='cpu')
    vision.load_state_dict(st['vision'])
    policy.load_state_dict(st['policy'])
    for sq, saved in zip(pol_sq, st['pol_sq']):
      sq.copy_(saved)
    if args.reset_grid:
      print('reset_grid: grid module reinitialized', flush=True)
    else:
      grid.load_state_dict(st['grid'])
    frames_done = st['frames_done']
    wall_prev = st.get('wall_s', 0.0)
    print(f'resumed at frames={frames_done} wall_s={wall_prev:.0f}',
          flush=True)

  ctrl = dict(stop=ctx.Value('b', False), frames=ctx.Value('q', frames_done),
              queue=ctx.Queue(100_000))

  def spawn_actor(rank):
    p = ctx.Process(target=actor_proc, args=(rank, vars(args), shared, ctrl),
                    daemon=True)
    p.start()
    return p

  def spawn_trainer(kind):
    p = ctx.Process(target=trainer_proc,
                    args=(vars(args), shared, ctrl, kind), daemon=True)
    p.start()
    return p

  actors = {r: spawn_actor(r) for r in range(W)}
  trainers = {k: spawn_trainer(k) for k in ('vision', 'grid')}
  respawns = 0

  t0 = time.time()
  frames0 = frames_done
  returns_hist = collections.deque(maxlen=500)
  ents = collections.deque(maxlen=W * 4)
  gp_buf = collections.deque(maxlen=64_000)
  vl = gl = None
  vis_done = grid_done = 0
  n_ep = 0
  update = 0
  metrics_f = open(os.path.join(args.out, 'metrics.jsonl'), 'a')
  last_metrics = last_resume = last_ckpt = time.time()

  def save_resume():
    tmp = resume_path + '.tmp'
    torch.save(dict(vision=vision.state_dict(), grid=grid.state_dict(),
                    policy=policy.state_dict(),
                    pol_sq=[s.clone() for s in pol_sq],
                    frames_done=int(ctrl['frames'].value),
                    wall_s=wall_prev + (time.time() - t0)), tmp)
    os.replace(tmp, resume_path)

  def dump_gridcodes(tag):
    if not gp_buf:
      return
    snap = list(gp_buf)
    np.savez_compressed(
        os.path.join(args.out, f'gridcodes_{tag}.npz'),
        g=np.stack([s[0] for s in snap]),
        pos=np.stack([s[1] for s in snap]),
        hd=np.asarray([s[2] for s in snap], dtype=np.float32))

  def elapsed_total():
    return wall_prev + (time.time() - t0)

  try:
    while True:
      frames_now = ctrl['frames'].value
      if frames_now >= args.frames:
        print(f'frame budget reached: {frames_now}', flush=True)
        break
      if args.max_seconds and time.time() - t0 >= args.max_seconds:
        print(f'wall-clock budget reached: {time.time() - t0:.0f}s this '
              f'session ({elapsed_total():.0f}s total) at {frames_now} '
              'frames', flush=True)
        break

      # Drain reports.
      t_drain = time.time()
      while time.time() - t_drain < 5.0:
        try:
          msg = ctrl['queue'].get(timeout=0.5)
        except queue_lib.Empty:
          break
        kind = msg[0]
        if kind == 'ep':
          returns_hist.append(msg[2])
          n_ep += 1
        elif kind == 'stat':
          ents.append(msg[2])
        elif kind == 'gp':
          gp_buf.append((msg[1], msg[2], msg[3]))
        elif kind == 'trainer':
          if msg[1] == 'vision':
            vl, vis_done = msg[2], msg[3]
          else:
            gl, grid_done = msg[2], msg[3]
        elif kind == 'died':
          who = {-1: 'vision trainer', -2: 'grid trainer'}.get(
              msg[1], f'actor {msg[1]}')
          print(f'{who} died:\n{msg[2]}', flush=True)

      # Respawn dead processes (DM-Lab can crash; keep the fleet full).
      if respawns < 300:
        for r, p in list(actors.items()):
          if not p.is_alive():
            print(f'respawning actor {r}', flush=True)
            actors[r] = spawn_actor(r)
            respawns += 1
        for k, p in list(trainers.items()):
          if not p.is_alive():
            print(f'respawning {k} trainer', flush=True)
            trainers[k] = spawn_trainer(k)
            respawns += 1

      now = time.time()
      if now - last_metrics > 30:
        last_metrics = now
        update += 1
        fps = (frames_now - frames0) / max(now - t0, 1e-9)
        rec = dict(update=update, frames=int(frames_now), fps=round(fps),
                   avg_return_50=round(float(np.mean(
                       list(returns_hist)[-50:])), 2) if returns_hist else 0.0,
                   ent=round(float(np.mean(ents)), 3) if ents else None,
                   vis_loss=vl and round(vl, 4),
                   grid_loss=gl and round(gl, 4),
                   vis_updates=vis_done, grid_updates=grid_done,
                   episodes=n_ep,
                   workers_alive=sum(p.is_alive() for p in actors.values()),
                   wall_s=round(elapsed_total()))
        print(json.dumps(rec), flush=True)
        metrics_f.write(json.dumps(rec) + '\n')
        metrics_f.flush()
      if now - last_resume > 600:
        last_resume = now
        save_resume()
      if now - last_ckpt > 3600:
        last_ckpt = now
        torch.save({'vision': vision.state_dict(), 'grid': grid.state_dict(),
                    'policy': policy.state_dict()},
                   os.path.join(args.out, f'ckpt_{int(frames_now):09d}.pt'))
        dump_gridcodes(f'{int(frames_now):09d}')
  except KeyboardInterrupt:
    print('interrupted; saving', flush=True)

  ctrl['stop'].value = True
  procs = list(actors.values()) + list(trainers.values())
  deadline = time.time() + 60
  for p in procs:
    p.join(timeout=max(0.1, deadline - time.time()))
  for p in procs:
    if p.is_alive():
      p.terminate()
  save_resume()
  torch.save({'vision': vision.state_dict(), 'grid': grid.state_dict(),
              'policy': policy.state_dict()},
             os.path.join(args.out, 'ckpt_final.pt'))
  dump_gridcodes('final')
  metrics_f.close()
  print('done', flush=True)


if __name__ == '__main__':
  main()
