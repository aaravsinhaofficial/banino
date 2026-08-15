#!/bin/bash
# Pull the finished async-A3C cells from S3 and run the full analysis:
# grid-ness of each agent's grid module, goal-vector decoding from the policy
# LSTM, frozen-policy evals (final and peak checkpoint), and the
# paper-vs-chance-vs-ours table.
#
# Every per-cell setting (level, agent, arena geometry, velocity encoding) is
# read from that cell's own config.json — the cells run different levels, and
# analysing an arena cell on the maze level would silently produce garbage.
#
# Usage: bash rl/finalize_a3c.sh [cell ...]
set -u
export AWS_PROFILE=banino-repro AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/banino
BUCKET=s3://banino-repro-975050064729/rl_runs
DEFAULT_CELLS="a3c_goal_grid a3c_goal_place a3c_goal_a3c a3c_goal_grid_hi \
a3c_arena_grid a3c_arena_place a3c_arena_grid2 a3c_arena_place2"
if [ $# -gt 0 ]; then CELLS=("$@"); else read -r -a CELLS <<< "$DEFAULT_CELLS"; fi
DOCKER="docker run --rm --shm-size=8g -v /home/ec2-user/banino:/workspace -w /workspace dmlab-rl:cpu"

cfg_get() {  # cfg_get <cell> <key> <default>
  .venv/bin/python - "$1" "$2" "$3" <<'PY'
import json, sys
cell, key, dflt = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    v = json.load(open(f'rl_runs/{cell}/config.json')).get(key, dflt)
except Exception:
    v = dflt
print(dflt if v is None else v)
PY
}

for c in "${CELLS[@]}"; do
  echo "=== $c"
  if aws s3 cp "$BUCKET/$c/results.tar.gz" "/tmp/$c.tar.gz" --quiet 2>/dev/null; then
    tar xzf "/tmp/$c.tar.gz" -C rl_runs/ || { echo "  extract failed"; continue; }
  else
    echo "  no synced tarball"; continue
  fi
  sudo chown -R "$(id -u):$(id -g)" "rl_runs/$c" 2>/dev/null

  agent=$(cfg_get "$c" agent grid)
  level=$(cfg_get "$c" level contributed/dmlab30/explore_goal_locations_small)
  cells_n=$(cfg_get "$c" arena_cells 11)
  half=$(cfg_get "$c" arena_half_m 1.375)
  velenc=$(cfg_get "$c" vel_encoding raw)
  echo "  agent=$agent level=$level arena_cells=$cells_n half=$half"

  ckpt="rl_runs/$c/ckpt_final.pt"
  [ -f "$ckpt" ] || ckpt=$(ls -t rl_runs/$c/ckpt_*.pt 2>/dev/null | head -1)
  [ -f "$ckpt" ] || ckpt="rl_runs/$c/resume.pt"
  [ -f "$ckpt" ] || { echo "  no checkpoint"; continue; }
  echo "  ckpt=$ckpt"

  # Grid-ness of the agent's grid module (Fig 2g analog).
  if ls rl_runs/$c/gridcodes_*.npz >/dev/null 2>&1; then
    .venv/bin/python -m rl.score_gridcells "rl_runs/$c" --arena_half "$half" \
      > "rl_runs/$c/grid_scores.log" 2>&1 \
      && echo "  grid scores done" || echo "  grid scoring FAILED"
  fi

  # Goal-vector decoding from the policy LSTM (Fig 2j/k analog).
  $DOCKER python3 -m rl.decode_goal --ckpt "$ckpt" --level "$level" \
    --agent "$agent" --episodes 60 --n_envs 16 --arena_cells "$cells_n" \
    --vel_encoding "$velenc" --device cpu --out "rl_runs/$c" \
    > "rl_runs/$c/decode.log" 2>&1 \
    && echo "  goal decoding done" || echo "  goal decoding FAILED (see decode.log)"

  # Eval fallback if the instance died before running its own.
  if [ ! -f "rl_runs/$c/eval_scores.json" ]; then
    echo "  no on-box eval; running 100 episodes locally"
    $DOCKER python3 -m rl.eval_agent --ckpt "$ckpt" --level "$level" \
      --agent "$agent" --episodes 100 --n_envs 24 --arena_cells "$cells_n" \
      --vel_encoding "$velenc" --device cpu --out "rl_runs/$c" \
      > "rl_runs/$c/eval_local.log" 2>&1 \
      && echo "  local eval done" || echo "  local eval FAILED"
  fi

  # Peak-checkpoint eval: these runs swing widely, so the final checkpoint
  # alone understates what the configuration reached.
  peak=$(.venv/bin/python - "$c" <<'PY'
import glob, json, os, re, sys
c = sys.argv[1]
try:
    rows = [json.loads(l) for l in open(f'rl_runs/{c}/metrics.jsonl') if l.strip()]
except OSError:
    sys.exit()
rows = [r for r in rows if r.get('avg_return_50') is not None]
if not rows:
    sys.exit()
best = max(rows, key=lambda r: r['avg_return_50'])
cands = []
for p in glob.glob(f'rl_runs/{c}/ckpt_*.pt'):
    m = re.search(r'ckpt_(\d+)\.pt', os.path.basename(p))
    if m:
        cands.append((abs(int(m.group(1)) - best['frames']), p))
if cands:
    print(min(cands)[1])
PY
)
  if [ -n "$peak" ] && [ "$peak" != "$ckpt" ]; then
    echo "  peak ckpt: $peak"
    mkdir -p "rl_runs/$c/peak"
    $DOCKER python3 -m rl.eval_agent --ckpt "$peak" --level "$level" \
      --agent "$agent" --episodes 100 --n_envs 24 --arena_cells "$cells_n" \
      --vel_encoding "$velenc" --device cpu --out "rl_runs/$c/peak" \
      > "rl_runs/$c/eval_peak.log" 2>&1 \
      && echo "  peak eval done" || echo "  peak eval FAILED"
  fi
  sudo chown -R "$(id -u):$(id -g)" "rl_runs/$c" 2>/dev/null
done

.venv/bin/python -m rl.aggregate_a3c --rl rl_runs \
  --out A3C_RESULTS.json --md A3C_RESULTS.md
.venv/bin/python -m rl.plot_a3c --rl rl_runs --out report/fig_a3c.png
