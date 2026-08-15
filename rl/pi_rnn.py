"""Frozen path-integration RNN as the agent's grid-code module.

Wraps a pretrained Sorscher-style RNN from the predictive-grid-cell-analysis
repo (encoder Linear(Np=512 -> Ng=4096, no bias), nn.RNN(2 -> 4096, relu,
no bias), decoder Linear(4096 -> 512, no bias)) so its hidden state can be
streamed one decision step at a time inside the RL loop and fed to the
policy in place of the GridModule's bottleneck code.

Conventions (must match the source repo exactly):
  - Input is the allocentric per-step displacement [dx, dy] in metres,
    in the RNN's native box [-1.1, 1.1] m. The RL loop is responsible for
    scaling env coordinates into this box (scale = box_half / env_half).
  - h0 = encoder(p0) where p0 is the DoG place-cell activation at the
    start position; h0 enters the recurrence UN-rectified, exactly as in
    the source model (model.g: RNN(v, encoder(p0)[None])). On reinit we
    take one recurrence step with zero velocity so the code handed to the
    policy is always a post-recurrence (rectified) state.
  - Place-cell centres are frozen by np.random.seed(0) in the source repo
    (place_cells.py) and are regenerated here with the identical RNG calls.
  - Ablation follows the source repo's lesion semantics
    (zero_unit_weights_in_place): zero encoder row, W_ih row, W_hh row and
    column, decoder column ("full" mode). "readout" mode instead leaves
    the dynamics intact and only masks the code the policy (and the goal
    snapshot) sees.
"""

import numpy as np
import torch
from torch import nn

BOX_HALF = 1.1        # source-repo box: [-1.1, 1.1] m
NP_PLACE = 512
SIGMA = 0.12          # place_cell_rf
SURROUND = 2.0        # surround_scale (DoG)


def make_place_cell_centers(n_cells=NP_PLACE, box_half=BOX_HALF):
  """Replicates place_cells.py: np.random.seed(0), x then y uniform draws."""
  np.random.seed(0)
  usx = np.random.uniform(-box_half, box_half, (n_cells,))
  usy = np.random.uniform(-box_half, box_half, (n_cells,))
  return np.vstack([usx, usy]).T.astype(np.float64)  # [Np, 2]


class PIRNN(nn.Module):
  """Streaming, frozen wrapper around the pretrained path-integration RNN."""

  def __init__(self, ckpt_path, device='cpu', ablate_units=None,
               ablation_mode='full', jump_thresh=0.25):
    super().__init__()
    sd = torch.load(ckpt_path, map_location='cpu')
    w_ih = sd['RNN.weight_ih_l0'].clone().float()    # [Ng, 2]
    w_hh = sd['RNN.weight_hh_l0'].clone().float()    # [Ng, Ng]
    enc = sd['encoder.weight'].clone().float()       # [Ng, Np]
    dec = sd['decoder.weight'].clone().float()       # [Np, Ng]
    self.Ng = w_hh.shape[0]
    self.Np = enc.shape[1]
    self.ablation_mode = ablation_mode
    self.jump_thresh = jump_thresh
    self.ckpt_path = str(ckpt_path)

    units = np.asarray(ablate_units, dtype=int) if ablate_units is not None \
        else np.zeros(0, dtype=int)
    self.ablated_units = units
    mask = torch.ones(self.Ng)
    if units.size:
      if ablation_mode == 'full':
        # Source-repo lesion: silence the unit everywhere (weight surgery).
        u = torch.as_tensor(units, dtype=torch.long)
        enc[u, :] = 0.0
        w_ih[u, :] = 0.0
        w_hh[u, :] = 0.0
        w_hh[:, u] = 0.0
        dec[:, u] = 0.0
      elif ablation_mode == 'readout':
        mask[units] = 0.0    # dynamics intact; policy view masked
      else:
        raise ValueError(f'unknown ablation_mode {ablation_mode!r}')

    self.register_buffer('w_ih', w_ih)
    self.register_buffer('w_hh', w_hh)
    self.register_buffer('encoder_w', enc)
    self.register_buffer('decoder_w', dec)
    self.register_buffer('readout_mask', mask)
    us = torch.as_tensor(make_place_cell_centers(self.Np), dtype=torch.float32)
    self.register_buffer('pc_us', us)                # [Np, 2]
    for p in self.parameters():
      p.requires_grad_(False)
    self.to(device)

  # ---------- place cells (DoG, softmax-normalised, as in source repo) ----
  def place_cell_activation(self, pos):
    """pos [B,2] (RNN-box metres) -> DoG place-cell activation [B, Np]."""
    d2 = ((pos[:, None, :] - self.pc_us[None, :, :]) ** 2).sum(-1)  # [B,Np]
    out = torch.softmax(-d2 / (2 * SIGMA ** 2), dim=-1)
    out = out - torch.softmax(-d2 / (2 * SURROUND * SIGMA ** 2), dim=-1)
    out = out + out.min(dim=-1, keepdim=True).values.abs()
    out = out / out.sum(dim=-1, keepdim=True)
    return out

  # ---------- streaming API ----------
  def init_state(self, pos):
    """h0 = encoder(place cells(pos)), UN-rectified [B, Ng]."""
    p0 = self.place_cell_activation(pos)
    return p0 @ self.encoder_w.t()

  def advance(self, h, v, reinit, pos):
    """One decision step.

    h [B,Ng] carried state (post-recurrence, except right after init_state);
    v [B,2] allocentric displacement this step (RNN-box metres);
    reinit [B] bool: env i teleported/(re)spawned -> re-derive its state from
    the place-cell code at pos and take a zero-velocity settle step;
    pos [B,2] current position (RNN-box metres).
    Returns (h_next [B,Ng], reinit_applied [B] bool).
    """
    # Belt and braces: a displacement far beyond the motion model means an
    # unflagged teleport; re-anchor rather than integrate the jump.
    jump = v.norm(dim=1) > self.jump_thresh
    do = reinit | jump
    if do.any():
      h = torch.where(do[:, None], self.init_state(pos), h)
      v = torch.where(do[:, None], torch.zeros_like(v), v)
    h_next = torch.relu(v @ self.w_ih.t() + h @ self.w_hh.t())
    return h_next, do

  def g_for_policy(self, h):
    return h * self.readout_mask[None, :]

  def decode_pos(self, h, k=3):
    """Top-k place-cell decode of position [B,2] (RNN-box metres)."""
    logits = h @ self.decoder_w.t()
    _, idx = torch.topk(logits, k=k, dim=-1)
    return self.pc_us[idx].mean(-2)
