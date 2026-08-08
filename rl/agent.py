"""Replay buffer and the two supervised trainers of the RL agent.

Per the paper: a shared replay of recent experience feeds (a) the vision
module (minibatch 32 frames -> place/HD code cross-entropy) and (b) the
grid network (minibatch 10 sequences of 100 consecutive steps, visual
codes masked to zero 95% of the time, velocity noise sigma=0.01, and the
supervised-experiment optimization recipe: RMSProp 1e-5, momentum 0.9,
elementwise grad clip 1e-5, weight decay 1e-5 on bottleneck/head weights).
Scaled down: replay holds 2M frames (paper: 20M) in host RAM.
"""

import numpy as np
import torch


def soft_ce(logits, target):
  return -(target * torch.log_softmax(logits, dim=-1)).sum(-1)


class Replay:
  """Per-env contiguous ring buffers so consecutive sequences can be drawn."""

  def __init__(self, n_envs, cap_per_env, frame_hw=84):
    self.n, self.cap = n_envs, cap_per_env
    self.frames = np.zeros((n_envs, cap_per_env, frame_hw, frame_hw, 3),
                           np.uint8)
    self.vels = np.zeros((n_envs, cap_per_env, 3), np.float32)
    self.poss = np.zeros((n_envs, cap_per_env, 2), np.float32)
    self.hds = np.zeros((n_envs, cap_per_env), np.float32)
    self.eps = np.full((n_envs, cap_per_env), -1, np.int64)
    self.ptr = 0
    self.count = 0

  def add(self, obs, episode_ids):
    """obs: batched dict from the vec env; episode_ids [n] int64."""
    p = self.ptr
    self.frames[:, p] = obs['rgb']
    self.vels[:, p] = obs['vel']
    self.poss[:, p] = obs['pos']
    self.hds[:, p] = obs['hd']
    self.eps[:, p] = episode_ids
    self.ptr = (p + 1) % self.cap
    self.count = min(self.count + 1, self.cap)

  def sample_frames(self, batch, rng):
    e = rng.integers(0, self.n, batch)
    t = rng.integers(0, self.count, batch)
    return self.frames[e, t], self.poss[e, t], self.hds[e, t]

  def sample_sequences(self, batch, seqlen, rng, max_tries=200):
    """Windows of consecutive steps within a single episode."""
    out = []
    tries = 0
    while len(out) < batch and tries < max_tries * batch:
      tries += 1
      e = int(rng.integers(0, self.n))
      if self.count < self.cap:
        if self.count <= seqlen:
          continue
        s = int(rng.integers(0, self.count - seqlen))
      else:
        s = int(rng.integers(0, self.cap - seqlen))
        # Avoid the write head: the window must not straddle self.ptr.
        if s < self.ptr <= s + seqlen:
          continue
      ep = self.eps[e, s:s + seqlen]
      if ep[0] < 0 or not (ep == ep[0]).all():
        continue
      out.append((e, s))
    idx_e = np.array([e for e, _ in out])
    idx_s = np.array([s for _, s in out])
    gather = lambda arr: np.stack(
        [arr[e, s:s + seqlen] for e, s in zip(idx_e, idx_s)])
    return (gather(self.frames), gather(self.vels), gather(self.poss),
            gather(self.hds))


def vision_update(vision, opt, replay, pc_ens, hd_ens, rng, device,
                  batch=32):
  frames, pos, hd = replay.sample_frames(batch, rng)
  rgb = torch.as_tensor(frames, device=device).permute(0, 3, 1, 2).float() / 255.
  pos_t = torch.as_tensor(pos, device=device)
  hd_t = torch.as_tensor(hd, device=device).unsqueeze(-1)
  pc_logits, hd_logits = vision(rgb)
  loss = (soft_ce(pc_logits, pc_ens.get_targets(pos_t)) +
          soft_ce(hd_logits, hd_ens.get_targets(hd_t))).mean()
  opt.zero_grad(set_to_none=True)
  loss.backward()
  opt.step()
  return loss.item()


def grid_update(grid, vision, opt, replay, pc_ens, hd_ens, rng, device,
                batch=10, seqlen=100, mask_keep=0.05, vel_noise=0.01,
                l2=1e-4, clip=0.0):
  frames, vels, poss, hds = replay.sample_sequences(batch, seqlen, rng)
  if len(frames) < batch:
    return None
  B, T = frames.shape[:2]
  rgb = torch.as_tensor(frames, device=device).permute(0, 1, 4, 2, 3)
  rgb = rgb.reshape(B * T, 3, 84, 84).float() / 255.
  with torch.no_grad():
    pc_l, hd_l = vision(rgb)
    vis_pc = torch.softmax(pc_l, -1).reshape(B, T, -1)
    vis_hd = torch.softmax(hd_l, -1).reshape(B, T, -1)
  vel = torch.as_tensor(vels, device=device)
  vel = vel + vel_noise * torch.randn_like(vel)
  mask = (torch.rand(B, T, 1, device=device) < mask_keep).float()
  pos_t = torch.as_tensor(poss, device=device)
  hd_t = torch.as_tensor(hds, device=device).unsqueeze(-1)

  state = grid.zero_state(B, device)
  pc_logits, hd_logits = [], []
  for t in range(T):
    g, state = grid.step(vel[:, t], vis_pc[:, t], vis_hd[:, t],
                         mask[:, t], state)
    pl, hl = grid.heads(g)
    pc_logits.append(pl)
    hd_logits.append(hl)
  pc_logits = torch.stack(pc_logits, 1)
  hd_logits = torch.stack(hd_logits, 1)
  loss = (soft_ce(pc_logits, pc_ens.get_targets(pos_t)) +
          soft_ce(hd_logits, hd_ens.get_targets(hd_t))).mean()
  opt.zero_grad(set_to_none=True)
  loss.backward()
  for p in grid.regularized_params():
    p.grad.add_(p.data, alpha=l2)
  if clip > 0:
    for p in grid.parameters():
      if p.grad is not None:
        p.grad.clamp_(-clip, clip)
  opt.step()
  return loss.item()
