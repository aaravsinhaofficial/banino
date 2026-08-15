"""Equivalence + stability tests for rl.pi_rnn against the source repo.

Run with the predictive-grid-cell-analysis repo available (host path or
/pgc inside the container; override with PGC_REPO):
  python3 -m rl.test_pi_rnn
"""

import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.pi_rnn import BOX_HALF, PIRNN, make_place_cell_centers  # noqa: E402

PGC = os.environ.get('PGC_REPO', '/pgc' if os.path.isdir('/pgc')
                     else '/home/ec2-user/predictive-grid-cell-analysis')
sys.path.insert(0, os.path.join(PGC, 'code'))
import model as pgc_model            # noqa: E402
import place_cells as pgc_pc         # noqa: E402

# Extract zero_unit_weights_in_place from its module source (importing the
# module pulls cv2/imageio, unavailable on the host).
import re                            # noqa: E402
import typing                        # noqa: E402
_src = open(os.path.join(
    PGC, 'code/multi_seed_predictive_analysis.py')).read()
_m = re.search(r'\ndef zero_unit_weights_in_place.*?(?=\ndef )', _src,
               re.DOTALL)
_ns = {'np': np, 'torch': torch, 'RNN': pgc_model.RNN,
       'Sequence': typing.Sequence}
exec(_m.group(0), _ns)  # noqa: S102 - trusted local repo source
zero_unit_weights_in_place = _ns['zero_unit_weights_in_place']

CKPT = os.path.join(
    PGC, 'Models/canonical_cohort_v1/seed_0',
    'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_'
    'lr_00001_weight_decay_1e-06/most_recent_model.pth')


def ref_options():
  o = types.SimpleNamespace()
  o.Np, o.Ng = 512, 4096
  o.place_cell_rf, o.surround_scale = 0.12, 2.0
  o.box_width = o.box_height = 2.2
  o.periodic, o.DoG = False, True
  o.device = 'cpu'
  o.activation = 'relu'
  o.sequence_length, o.weight_decay, o.velocity_dim = 20, 1e-6, 2
  return o


def build_ref(ckpt=CKPT):
  o = ref_options()
  pc = pgc_pc.PlaceCells(o)
  m = pgc_model.RNN(o, pc)
  m.load_state_dict(torch.load(ckpt, map_location='cpu'))
  m.eval()
  return m, pc


def main():
  torch.manual_seed(0)
  rng = np.random.default_rng(0)
  ref, ref_pc = build_ref()
  mine = PIRNN(CKPT, device='cpu')

  # --- 1) place-cell centres + activation ---
  assert np.allclose(make_place_cell_centers(),
                     ref_pc.us.numpy(), atol=0), 'centres differ'
  pos = torch.as_tensor(rng.uniform(-1.0, 1.0, (64, 2)), dtype=torch.float32)
  a_ref = ref_pc.get_activation(pos[:, None, :].double())[:, 0]
  a_my = mine.place_cell_activation(pos)
  d = (a_ref.float() - a_my).abs().max().item()
  assert d < 1e-5, f'place-cell activation mismatch {d}'
  print(f'PASS place cells (max dev {d:.2e})')

  # --- 2) stepwise recurrence == reference model.g over 40 steps ---
  T, B = 40, 16
  v = torch.as_tensor(rng.normal(0, 0.015, (T, B, 2)), dtype=torch.float32)
  p0 = mine.place_cell_activation(
      torch.as_tensor(rng.uniform(-0.9, 0.9, (B, 2)), dtype=torch.float32))
  g_ref = ref.g((v, p0))                       # [T, B, Ng]
  h = p0 @ mine.encoder_w.t()                  # un-rectified h0
  errs = []
  for t in range(T):
    h = torch.relu(v[t] @ mine.w_ih.t() + h @ mine.w_hh.t())
    errs.append((h - g_ref[t]).abs().max().item())
  assert max(errs) < 1e-3, f'stepwise mismatch {max(errs)}'
  print(f'PASS stepwise recurrence (max dev over {T} steps {max(errs):.2e})')

  # --- 2b) advance() with reinit reproduces the same t=1 state after a
  #         zero-velocity settle step ---
  start = torch.as_tensor(rng.uniform(-0.9, 0.9, (B, 2)), dtype=torch.float32)
  hz = torch.zeros(B, mine.Ng)
  h1, did = mine.advance(hz, torch.full((B, 2), 9.9),  # garbage v: must be
                         torch.ones(B, dtype=torch.bool), start)  # ignored
  p0s = mine.place_cell_activation(start)
  h1_ref = torch.relu((p0s @ mine.encoder_w.t()) @ mine.w_hh.t())
  assert did.all() and (h1 - h1_ref).abs().max() < 1e-5, 'reinit mismatch'
  print('PASS advance()/reinit semantics')

  # --- 3) full-mode ablation == source-repo weight surgery ---
  z = np.load(os.path.join(
      PGC, 'analysis_outputs/canonical_cohort_v1/seed_0',
      'steps_20_batch_200_RNN_4096_relu_rf_012_DoG_True_periodic_False_'
      'lr_00001_weight_decay_1e-06/pgc_rigor/pgc_classification.npz'))
  pred_units = np.where(z['labels'] == 1)[0]
  print(f'seed_0 predictive units: n={len(pred_units)}')
  ref_ab, _ = build_ref()
  zero_unit_weights_in_place(ref_ab, pred_units)
  mine_ab = PIRNN(CKPT, device='cpu', ablate_units=pred_units,
                  ablation_mode='full')
  g_ref = ref_ab.g((v, p0))
  h = p0 @ mine_ab.encoder_w.t()
  worst = 0.0
  for t in range(T):
    h = torch.relu(v[t] @ mine_ab.w_ih.t() + h @ mine_ab.w_hh.t())
    worst = max(worst, (h - g_ref[t]).abs().max().item())
  assert worst < 1e-3, f'ablated stepwise mismatch {worst}'
  assert (h[:, pred_units].abs().max() == 0), 'ablated units not silent'
  print(f'PASS full-mode ablation == zero_unit_weights_in_place '
        f'(max dev {worst:.2e})')

  # --- 4) long-horizon stability + decode error on FakeEnv-like motion ---
  for name, m in (('intact', mine), ('pred-ablated', mine_ab)):
    B2, steps = 32, 2000
    pos_np = rng.uniform(-0.9, 0.9, (B2, 2))
    hd = rng.uniform(-np.pi, np.pi, B2)
    h = None
    marks = {}
    p = torch.as_tensor(pos_np, dtype=torch.float32)
    h, _ = m.advance(torch.zeros(B2, m.Ng), torch.zeros(B2, 2),
                     torch.ones(B2, dtype=torch.bool), p)
    for t in range(1, steps + 1):
      hd += rng.normal(0, 0.3, B2)
      speed = np.clip(rng.normal(0.25, 0.08, B2), 0, 0.35) * (4 / 60.)
      step = np.stack([speed * np.cos(hd), speed * np.sin(hd)], 1)
      new = np.clip(pos_np + step, -1.02, 1.02)  # positions already in box coords
      v_t = torch.as_tensor(new - pos_np, dtype=torch.float32)
      pos_np = new
      p = torch.as_tensor(pos_np, dtype=torch.float32)
      h, _ = m.advance(h, v_t, torch.zeros(B2, dtype=torch.bool), p)
      if t in (20, 100, 300, 1000, 2000):
        err = 100 * (m.decode_pos(h) - p).norm(dim=1).mean().item()
        marks[t] = (round(err, 1), round(h.abs().max().item(), 1))
    print(f'{name}: decode err cm (h_max) by step: {marks}')
    assert np.isfinite(h.numpy()).all(), 'state blew up'
  print('ALL TESTS PASSED')


if __name__ == '__main__':
  main()
