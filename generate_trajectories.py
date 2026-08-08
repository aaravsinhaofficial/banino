# Copyright 2026 — Apache License 2.0 (see LICENSE).
"""Regenerates the training dataset for the grid-cells code.

The original dataset (gs://grid-cells-datasets/square_room_100steps_2.2m_1000000)
was deleted from GCS and no mirror or archive exists, so this script
re-simulates it: foraging-rodent trajectories in a square arena following the
motion model of Raudies & Hasselmo (2012), as described in the Methods of
Banino et al. (2018) and its Supplementary Methods Table 1 (parameters as
reconstructed by the community, e.g. github.com/soluslan/grid-cells-torch):

  - 15 s trajectories simulated at dt = 0.02 s (750 fine steps), then
    subsampled to 100 stored steps at 0.15 s spacing (the paper unrolls BPTT
    over 100 steps covering the 15 s trajectory);
  - forward speed ~ Rayleigh(scale 0.13 m/s);
  - angular velocity ~ Normal(0, 330 deg/s), applied as heading change per
    fine step;
  - within 0.03 m of a wall while heading towards it: speed reduced by a
    factor 0.25 and heading redirected parallel to the wall;
  - 2.2 m square arena centred on the origin.

TFRecords are written in the exact schema dataset_reader.py expects:

  init_pos   [2]       initial position (m)
  init_hd    [1]       initial head direction (rad, in [-pi, pi])
  ego_vel    [T, 3]    per stored step: (speed (m/s), sin/cos of the heading
                       change between consecutive stored steps)
  target_pos [T, 2]    position at each stored step (m)
  target_hd  [T, 1]    head direction at each stored step (rad)

This is a reconstruction, not the original data: DeepMind released only the
TFRecord *reader*, so the 3-component ego_vel composition is an inferred
convention and results can differ from the paper in detail.

Run (inside the Dockerfile environment):
  python generate_trajectories.py --root data --records_per_shard 10000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import multiprocessing
import os

import numpy as np
import tensorflow as tf

# Motion-model parameters (Banino et al. 2018 Supplementary Methods Table 1,
# after Raudies & Hasselmo 2012).
ARENA_SIZE = 2.2            # square side (m); coords in [-1.1, 1.1]
DURATION = 15.0             # trajectory duration (s)
DT = 0.02                   # fine simulation timestep (s)
SEQUENCE_LENGTH = 100       # stored steps per trajectory (0.15 s spacing)
SPEED_RAYLEIGH_SCALE = 0.13     # forward speed ~ Rayleigh(b) (m/s)
ROT_VEL_STD = np.deg2rad(330.0)  # angular velocity ~ N(0, sigma) (rad/s)
BORDER_REGION = 0.03        # wall-avoidance region (m)
WALL_SLOWDOWN = 0.25        # fractional speed reduction at the wall

NUM_SHARDS = 100            # dataset_reader expects 0000-of-0099 .. 0099-of-0099

_FINE_STEPS = int(round(DURATION / DT))              # 750
_SUBSAMPLE = _FINE_STEPS / float(SEQUENCE_LENGTH)    # 7.5 fine steps per stored


def _simulate(n, rng):
  """Vectorised simulation of n trajectories. Returns dict of float32 arrays."""
  half = ARENA_SIZE / 2.0
  pos = rng.uniform(-half, half, size=(n, 2))
  hd = rng.uniform(-np.pi, np.pi, size=n)  # unwrapped cumulative heading

  pos_hist = np.empty((n, _FINE_STEPS + 1, 2), dtype=np.float64)
  hd_hist = np.empty((n, _FINE_STEPS + 1), dtype=np.float64)
  pos_hist[:, 0] = pos
  hd_hist[:, 0] = hd

  for t in range(_FINE_STEPS):
    speed = rng.rayleigh(SPEED_RAYLEIGH_SCALE, size=n)
    turn = rng.normal(0.0, ROT_VEL_STD, size=n) * DT

    # Wall avoidance: when close to a wall and heading towards it, slow down
    # and redirect the heading parallel to the wall.
    hd_wrapped = np.mod(hd + np.pi, 2 * np.pi) - np.pi
    dists = np.stack([half - pos[:, 0], half - pos[:, 1],
                      half + pos[:, 0], half + pos[:, 1]], axis=1)
    nearest = np.argmin(dists, axis=1)
    d_wall = dists[np.arange(n), nearest]
    wall_normal = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])[nearest]
    hd_to_wall = np.mod(hd_wrapped - wall_normal + np.pi, 2 * np.pi) - np.pi
    near = (d_wall < BORDER_REGION) & (np.abs(hd_to_wall) < np.pi / 2)
    speed = np.where(near, speed * (1.0 - WALL_SLOWDOWN), speed)
    turn = turn + np.where(near,
                           np.sign(hd_to_wall) * (np.pi / 2 - np.abs(hd_to_wall)),
                           0.0)

    hd = hd + turn
    step = speed * DT
    pos = pos + np.stack([step * np.cos(hd), step * np.sin(hd)], axis=1)
    pos = np.clip(pos, -half, half)

    pos_hist[:, t + 1] = pos
    hd_hist[:, t + 1] = hd

  # Subsample to SEQUENCE_LENGTH stored steps at 0.15 s spacing, linearly
  # interpolating between fine steps (7.5 fine steps per stored step).
  fine_idx = np.arange(SEQUENCE_LENGTH + 1) * _SUBSAMPLE  # 0, 7.5, ..., 750
  i0 = np.floor(fine_idx).astype(int)
  i1 = np.minimum(i0 + 1, _FINE_STEPS)
  frac = fine_idx - i0
  sub_pos = (pos_hist[:, i0] * (1 - frac)[None, :, None] +
             pos_hist[:, i1] * frac[None, :, None])
  sub_hd = hd_hist[:, i0] * (1 - frac)[None, :] + hd_hist[:, i1] * frac[None, :]

  # Egocentric velocity per stored step: speed from the travelled distance,
  # angular component from the heading change between stored steps.
  step_dt = DURATION / SEQUENCE_LENGTH  # 0.15 s
  disp = np.diff(sub_pos, axis=1)
  speed = np.linalg.norm(disp, axis=-1) / step_dt
  dtheta = np.diff(sub_hd, axis=1)

  wrap = lambda a: np.mod(a + np.pi, 2 * np.pi) - np.pi
  return dict(
      init_pos=sub_pos[:, 0].astype(np.float32),
      init_hd=wrap(sub_hd[:, 0])[:, None].astype(np.float32),
      ego_vel=np.stack([speed, np.sin(dtheta), np.cos(dtheta)],
                       axis=-1).astype(np.float32),
      target_pos=sub_pos[:, 1:].astype(np.float32),
      target_hd=wrap(sub_hd[:, 1:])[..., None].astype(np.float32))


def _write_shard(shard, out_dir, records_per_shard, seed):
  rng = np.random.RandomState(seed + shard)
  data = _simulate(records_per_shard, rng)
  path = os.path.join(out_dir,
                      '{:04d}-of-{:04d}.tfrecord'.format(shard, NUM_SHARDS - 1))
  with tf.io.TFRecordWriter(path) as writer:
    for i in range(records_per_shard):
      feature = {
          k: tf.train.Feature(
              float_list=tf.train.FloatList(value=data[k][i].ravel()))
          for k in data
      }
      writer.write(tf.train.Example(
          features=tf.train.Features(feature=feature)).SerializeToString())
  return path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True,
                      help='Dataset root (train.py --task_root).')
  parser.add_argument('--records_per_shard', type=int, default=10000,
                      help='Trajectories per shard; 10000 matches the original '
                           '1M-trajectory dataset across 100 shards.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--workers', type=int,
                      default=min(NUM_SHARDS, multiprocessing.cpu_count()))
  args = parser.parse_args()

  out_dir = os.path.join(args.root, 'square_room_100steps_2.2m_1000000')
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  worker = functools.partial(_write_shard, out_dir=out_dir,
                             records_per_shard=args.records_per_shard,
                             seed=args.seed)
  pool = multiprocessing.Pool(args.workers)
  for path in pool.imap_unordered(worker, range(NUM_SHARDS)):
    print('wrote', path)
  pool.close()
  pool.join()


if __name__ == '__main__':
  main()
