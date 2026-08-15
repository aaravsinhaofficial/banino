"""Aggregate the async-A3C cells into a paper-vs-sync-vs-async table.

Reads each cell directory (as synced from S3) for:
  eval_scores.json   frozen-policy mean score over 100 episodes (SI protocol)
  metrics.jsonl      training curve (frames, fps, avg_return_50, entropy)
  grid_scores.json   grid-ness of the agent's grid module (rl.score_gridcells)
  goal_decode*.json  goal-vector decoding from the policy LSTM (rl.decode_goal)

Usage:
  .venv/bin/python -m rl.aggregate_a3c --rl rl_runs --out A3C_RESULTS.json \
      --md A3C_RESULTS.md
"""

import argparse
import glob
import json
import os

import numpy as np

# Paper benchmarks (SI Table 3 / Figs 2-3): mean score over 100 episodes.
PAPER = {
    'goal_maze': {'grid': 289.0, 'placecell': 238.0, 'human': 346.5},
    'goal_doors': {'grid': 284.3, 'placecell': 90.5},
    'agent_grid_pct': 21.4,     # % grid-like units in the agent's module
    'agent_top_grid_score': None,
}

# This repo's previous full-scale SYNCHRONOUS A2C run (REPORT.md), for the
# like-for-like async-vs-sync comparison this experiment is designed to make.
SYNC_A2C = {
    'goal_maze': {'grid': (7.6, 1.0), 'placecell': (7.1, 1.2),
                  'a3c': (6.1, 1.0)},
    'agent_grid_pct': 14.5,
    'agent_top_grid_score': 1.32,
    'goal_distance_r2': 0.55,
    'goal_direction_deg': 24.4,
}


def load_cell(d):
  cell = {'name': os.path.basename(d.rstrip('/'))}
  p = os.path.join(d, 'eval_scores.json')
  if os.path.exists(p):
    cell['eval'] = json.load(open(p))
  p = os.path.join(d, 'config.json')
  if os.path.exists(p):
    cell['config'] = json.load(open(p))
  p = os.path.join(d, 'metrics.jsonl')
  if os.path.exists(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if rows:
      cell['final_metrics'] = rows[-1]
      cell['frames'] = rows[-1].get('frames')
      tail = [r.get('avg_return_50') for r in rows[-20:]
              if r.get('avg_return_50') is not None]
      cell['train_return_tail_mean'] = round(float(np.mean(tail)), 2) if tail \
          else None
      cell['train_return_peak'] = round(
          max((r.get('avg_return_50') or 0) for r in rows), 2)
      # Coarse learning curve for the report figure.
      cell['curve'] = [(r.get('frames'), r.get('avg_return_50'))
                       for r in rows[::max(1, len(rows) // 200)]]
  p = os.path.join(d, 'grid_scores.json')
  if os.path.exists(p):
    cell['grid_scores'] = json.load(open(p))
  for p in sorted(glob.glob(os.path.join(d, 'goal_decode*.json'))):
    cell.setdefault('goal_decode', {})[os.path.basename(p)] = json.load(open(p))
  return cell


def fmt_score(cell):
  ev = cell.get('eval')
  if not ev:
    return 'n/a'
  return f"{ev['mean_score']:.1f} ± {ev['sem']:.1f}"


def load_chance(rl_dir, name='untrained_control'):
  """The untrained-network score under the same 100-episode protocol.

  Without it a score cannot be called learning: on the goal maze an
  untrained network scores ~7, which is where every agent in the previous
  full-scale run landed.
  """
  p = os.path.join(rl_dir, name, 'eval_scores.json')
  return json.load(open(p)) if os.path.exists(p) else None


def verdict(cell, chance):
  """Is this cell above the untrained baseline by more than 2 sigma?"""
  ev = cell.get('eval')
  if not ev or not chance:
    return 'no eval'
  diff = ev['mean_score'] - chance['mean_score']
  sigma = (ev['sem'] ** 2 + chance['sem'] ** 2) ** 0.5
  if sigma == 0:
    return 'n/a'
  z = diff / sigma
  if z >= 2:
    return f'LEARNED (+{diff:.1f}, {z:.1f} sigma)'
  if z <= -2:
    return f'below chance ({z:.1f} sigma)'
  return f'at chance ({z:+.1f} sigma)'


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--rl', default='rl_runs')
  ap.add_argument('--prefix', default='a3c_goal')
  ap.add_argument('--out', default='A3C_RESULTS.json')
  ap.add_argument('--md', default='A3C_RESULTS.md')
  args = ap.parse_args()

  dirs = sorted(d for d in glob.glob(os.path.join(args.rl, args.prefix + '*'))
                if os.path.isdir(d))
  cells = [load_cell(d) for d in dirs]
  chance = load_chance(args.rl)
  chance_sa = load_chance(args.rl, 'untrained_seekavoid')
  res = {'paper': PAPER, 'sync_a2c': SYNC_A2C, 'chance_goal_maze': chance,
         'chance_seekavoid': chance_sa,
         'async_a3c': {c['name']: c for c in cells}}
  with open(args.out, 'w') as f:
    json.dump(res, f, indent=1)

  lines = ['# Async A3C vs synchronous A2C vs paper', '']
  if chance:
    lines += [f"**Chance line (untrained network, 100 episodes): "
              f"{chance['mean_score']:.1f} ± {chance['sem']:.1f}** on the goal "
              'maze. Any score not clearly above this is not navigation.', '']
  lines += ['| cell | agent | workers | frames | eval score (100 ep) | '
            'vs chance | train return (tail) | peak |',
            '|---|---|---|---|---|---|---|---|']
  for c in cells:
    cfg = c.get('config', {})
    ch = chance_sa if 'seekavoid' in c['name'] else chance
    lines.append(
        f"| {c['name']} | {cfg.get('agent', '?')} | {cfg.get('workers', '?')} "
        f"| {c.get('frames', 0):,} | {fmt_score(c)} | {verdict(c, ch)} "
        f"| {c.get('train_return_tail_mean', 'n/a')} "
        f"| {c.get('train_return_peak', 'n/a')} |")
  lines += ['', '## Benchmarks', '',
            '| source | grid | place-cell | A3C-baseline |', '|---|---|---|---|',
            f"| Paper (goal maze) | {PAPER['goal_maze']['grid']} | "
            f"{PAPER['goal_maze']['placecell']} | — |",
            f"| Sync A2C, 1e9 frames | {SYNC_A2C['goal_maze']['grid'][0]} | "
            f"{SYNC_A2C['goal_maze']['placecell'][0]} | "
            f"{SYNC_A2C['goal_maze']['a3c'][0]} |"]
  if chance:
    lines.append(f"| **Untrained network (chance)** | "
                 f"{chance['mean_score']:.1f} | {chance['mean_score']:.1f} | "
                 f"{chance['mean_score']:.1f} |")
  by_agent = {}
  for c in cells:
    by_agent.setdefault(c.get('config', {}).get('agent'), []).append(c)
  row = []
  for a in ('grid', 'placecell', 'a3c'):
    got = [fmt_score(c) for c in by_agent.get(a, []) if c.get('eval')]
    row.append(' / '.join(got) if got else 'n/a')
  lines.append('| Async A3C (this run) | ' + ' | '.join(row) + ' |')

  for c in cells:
    if c.get('grid_scores') or c.get('goal_decode'):
      lines += ['', f"### {c['name']} representations"]
      gs = c.get('grid_scores')
      if gs:
        lines.append('- grid module: ' + json.dumps(gs)[:400])
      for k, v in (c.get('goal_decode') or {}).items():
        lines.append(f'- {k}: ' + json.dumps(v)[:400])
  md = '\n'.join(lines) + '\n'
  with open(args.md, 'w') as f:
    f.write(md)
  print(md)


if __name__ == '__main__':
  main()
