"""Supervised grid-cell training (PyTorch port of train.py).

Faithful to the paper/TF1 configuration:
  batch 10, BPTT over 100 steps, RMSProp(lr=1e-5, momentum=0.9, decay=0.9,
  eps=1e-10), elementwise gradient clipping to [-1e-5, 1e-5] on all
  parameters (utils.clip_all_gradients with the default flag), L2 weight
  decay 1e-5 on bottleneck and output-head weights, 512-unit bottleneck
  with dropout 0.5, 3e5 gradient steps, place cells N=256 sigma=0.01 m,
  HD cells M=12 kappa=20, neurons seed 8341.

Additions the TF1 code lacked: model checkpoints, position-decoding error
(the paper's Fig. 1a metric), and periodic dumps of bottleneck activations
for the offline analysis pipeline (grid/border/HD scores, shuffle nulls,
scale clustering).

Run:  .venv/bin/python -m modern.train_supervised --seed 1 --device cuda:0 \
        --out runs/seed1
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from modern import targets as targets_lib
from modern.data import TrajectorySampler
from modern.model import GridNetwork

EVAL_SIZE = 4000
EVAL_SEED = 999
POOL_SIZE = 200_000


def soft_ce(logits, target):
  return -(target * torch.log_softmax(logits, dim=-1)).sum(-1)


def make_pool(seed, n):
  """Fixed trajectory pool, mirroring the original fixed 1M-record dataset."""
  s = TrajectorySampler(seed=seed)
  return s.batch(n)


def evaluate(model, pool, pc, hd, device, batch=250):
  """Returns decode errors (m) at the final step and mean over steps."""
  model.eval()
  init_pos, init_hd, ego_vel, target_pos, target_hd = pool
  errs_final, errs_mean, acts, poss, hds = [], [], [], [], []
  with torch.no_grad():
    for i in range(0, init_pos.shape[0], batch):
      sl = slice(i, i + batch)
      ip, ih, ev = (x[sl].to(device) for x in (init_pos, init_hd, ego_vel))
      tp = target_pos[sl].to(device)
      pcl, _, btl = model(pc.get_targets(ip.unsqueeze(1)).squeeze(1),
                          hd.get_targets(ih.unsqueeze(1)).squeeze(1)
                          if ih.dim() == 2 else hd.get_targets(ih), ev)
      post = torch.softmax(pcl, dim=-1)
      pred_argmax = pc.means[post.argmax(-1)]           # [B,T,2]
      err = (pred_argmax - tp).norm(dim=-1)             # [B,T]
      errs_final.append(err[:, -1].cpu())
      errs_mean.append(err.mean(1).cpu())
      acts.append(btl.cpu().to(torch.float16))
      poss.append(tp.cpu().to(torch.float16))
      hds.append(target_hd[sl].to(torch.float16))
  model.train()
  return (torch.cat(errs_final).mean().item(),
          torch.cat(errs_mean).mean().item(),
          torch.cat(acts).numpy(), torch.cat(poss).numpy(),
          torch.cat(hds).numpy())


def ratemaps_from(acts, poss, nbins=20, half=1.1):
  """Mean activation per spatial bin: [units, nbins, nbins]."""
  a = acts.reshape(-1, acts.shape[-1]).astype(np.float32)
  p = poss.reshape(-1, 2).astype(np.float32)
  ix = np.clip(((p[:, 0] + half) / (2 * half) * nbins).astype(int), 0, nbins - 1)
  iy = np.clip(((p[:, 1] + half) / (2 * half) * nbins).astype(int), 0, nbins - 1)
  flat = ix * nbins + iy
  sums = np.zeros((nbins * nbins, a.shape[1]), np.float64)
  counts = np.bincount(flat, minlength=nbins * nbins)[:, None] + 1e-9
  np.add.at(sums, flat, a)
  return (sums / counts).T.reshape(a.shape[1], nbins, nbins)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--device', default='cuda:0')
  ap.add_argument('--steps', type=int, default=300_000)
  ap.add_argument('--out', required=True)
  ap.add_argument('--batch', type=int, default=10)
  ap.add_argument('--eval_every', type=int, default=50_000)
  ap.add_argument('--dump_at', type=int, nargs='*',
                  default=[0, 200_000, 300_000])
  args = ap.parse_args()
  os.makedirs(args.out, exist_ok=True)
  dev = torch.device(args.device)
  torch.manual_seed(args.seed)

  pc = targets_lib.PlaceCellEnsemble(device=dev)
  hd = targets_lib.HeadDirectionCellEnsemble(device=dev)
  model = GridNetwork().to(dev)

  # Weight decay is added to the gradient BEFORE clipping (as in TF, where
  # the l2_regularizer term is part of the clipped gradient), so the
  # optimizer itself carries no weight_decay.
  opt = torch.optim.RMSprop(model.parameters(), lr=1e-5, momentum=0.9,
                            alpha=0.9, eps=1e-10)
  reg_params = model.regularized_params()

  print(f'[seed {args.seed}] generating trajectory pool ({POOL_SIZE})...',
        flush=True)
  pool = make_pool(args.seed, POOL_SIZE)
  eval_pool = make_pool(EVAL_SEED, EVAL_SIZE)
  rng = np.random.RandomState(args.seed + 1)

  metrics = []
  t0 = time.time()
  for step in range(args.steps + 1):
    if step % args.eval_every == 0 or step == args.steps:
      ef, em, acts, poss, hds = evaluate(model, eval_pool, pc, hd, dev)
      rm = ratemaps_from(acts, poss)
      np.savez_compressed(os.path.join(args.out, f'ratemaps_{step:06d}.npz'),
                          ratemaps=rm)
      if step in args.dump_at:
        np.savez_compressed(os.path.join(args.out, f'acts_{step:06d}.npz'),
                            acts=acts, poss=poss, hds=hds)
        torch.save(model.state_dict(),
                   os.path.join(args.out, f'ckpt_{step:06d}.pt'))
      m = dict(step=step, decode_err_final_m=ef, decode_err_mean_m=em,
               elapsed_s=round(time.time() - t0, 1))
      metrics.append(m)
      with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
      print(f'[seed {args.seed}] step {step} decode_err_final '
            f'{ef * 100:.1f} cm (mean over steps {em * 100:.1f} cm)',
            flush=True)
    if step == args.steps:
      break

    idx = rng.randint(0, POOL_SIZE, args.batch)
    ip, ih, ev, tp, th = (x[idx].to(dev) for x in pool)
    pc_init = pc.get_targets(ip.unsqueeze(1)).squeeze(1)
    hd_init = hd.get_targets(ih.unsqueeze(1)).squeeze(1) \
        if ih.dim() == 2 else hd.get_targets(ih)
    pc_tgt = pc.get_targets(tp)
    hd_tgt = hd.get_targets(th).squeeze(-2)

    pcl, hdl, _ = model(pc_init, hd_init, ev)
    loss = (soft_ce(pcl, pc_tgt) + soft_ce(hdl, hd_tgt)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for p in reg_params:
      p.grad.add_(p.data, alpha=1e-5)
    for p in model.parameters():
      if p.grad is not None:
        p.grad.clamp_(-1e-5, 1e-5)
    opt.step()

    if step % 5000 == 0:
      sps = step / max(time.time() - t0, 1e-9)
      print(f'[seed {args.seed}] step {step} loss {loss.item():.4f} '
            f'({sps:.0f} steps/s)', flush=True)


if __name__ == '__main__':
  main()
