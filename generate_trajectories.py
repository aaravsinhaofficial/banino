# Copyright 2026 — Apache License 2.0 (see LICENSE).
"""Regenerates the training dataset for the grid-cells code.

The original dataset (gs://grid-cells-datasets/square_room_100steps_2.2m_1000000)
is no longer downloadable, so this script re-simulates it: foraging-rodent
trajectories in a square arena following the motion model of Raudies & Hasselmo
(2012) with the parameters reported in the Methods of Banino et al. (2018),
written as TFRecords in the exact schema dataset_reader.py expects:

  init_pos   [2]       initial position (m), arena centred on the origin
  init_hd    [1]       initial head direction (rad, in [-pi, pi])
  ego_vel    [T, 3]    egocentric velocity input: (speed (m/s),
                       sin(angular change per step), cos(angular change per step))
  target_pos [T, 2]    position after each step (m)
  target_hd  [T, 1]    head direction after each step (rad)

This is a reconstruction, not the original data: the exact per-step input
convention of the original TFRecords is not documented, so results can differ
from the paper in detail even though grid-like units should still emerge.

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

# Motion-model parameters (Raudies & Hasselmo 2012, as used in Banino et al.).
ARENA_SIZE = 2.2          # square side (m); coords in [-1.1, 1.1]
SEQUENCE_LENGTH = 100     # steps per trajectory
DT = 0.02                 # simulation timestep (s)
SPEED_RAYLEIGH_SCALE = 0.13   # forward speed ~ Rayleigh(b) (m/s)
ROT_VEL_STD = np.deg2rad(330.0)  # angular velocity ~ N(0, sigma) (rad/s)
BORDER_REGION = 0.03      # wall-avoidance region (m)
WALL_SLOWDOWN = 0.25      # speed factor when heading into a nearby wall

NUM_SHARDS = 100          # dataset_reader expects 0000-of-0099 .. 0099-of-0099


def _simulate(n, rng):
  """Vectorised simulation of n trajectories. Returns dict of float32 arrays."""
  half = ARENA_SIZE / 2.0
  pos = rng.uniform(-half, half, size=(n, 2))
  hd = rng.uniform(-np.pi, np.pi, size=n)
  init_pos, init_hd = pos.copy(), hd.copy()

  ego_vel = np.zeros((n, SEQUENCE_LENGTH, 3))
  target_pos = np.zeros((n, SEQUENCE_LENGTH, 2))
  target_hd = np.zeros((n, SEQUENCE_LENGTH, 1))

  for t in range(SEQUENCE_LENGTH):
    speed = rng.rayleigh(SPEED_RAYLEIGH_SCALE, size=n)
    turn = rng.normal(0.0, ROT_VEL_STD, size=n) * DT

    # Wall avoidance: when close to a wall and heading towards it, slow down
    # and turn away (towards the perpendicular).
    dists = np.stack([half - pos[:, 0], half - pos[:, 1],
                      half + pos[:, 0], half + pos[:, 1]], axis=1)
    nearest = np.argmin(dists, axis=1)
    d_wall = dists[np.arange(n), nearest]
    wall_normal = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])[nearest]
    hd_to_wall = np.mod(hd - wall_normal + np.pi, 2 * np.pi) - np.pi
    near = (d_wall < BORDER_REGION) & (np.abs(hd_to_wall) < np.pi / 2)
    speed = np.where(near, speed * WALL_SLOWDOWN, speed)
    turn = turn + np.where(near,
                           np.sign(hd_to_wall) * (np.pi / 2 - np.abs(hd_to_wall)),
                           0.0)

    hd = np.mod(hd + turn + np.pi, 2 * np.pi) - np.pi
    step = speed * DT
    pos = pos + np.stack([step * np.cos(hd), step * np.sin(hd)], axis=1)
    pos = np.clip(pos, -half, half)

    ego_vel[:, t] = np.stack([speed, np.sin(turn), np.cos(turn)], axis=1)
    target_pos[:, t] = pos
    target_hd[:, t, 0] = hd

  return dict(init_pos=init_pos.astype(np.float32),
              init_hd=init_hd[:, None].astype(np.float32),
              ego_vel=ego_vel.astype(np.float32),
              target_pos=target_pos.astype(np.float32),
              target_hd=target_hd.astype(np.float32))


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
