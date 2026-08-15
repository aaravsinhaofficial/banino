#!/bin/bash
# Pull the finished async-A3C cells from S3 and run the full analysis:
# grid-ness of each agent's grid module, goal-vector decoding from the policy
# LSTM, and the paper-vs-sync-vs-async table.
# Usage: bash rl/finalize_a3c.sh [cell ...]
set -u
export AWS_PROFILE=banino-repro AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/banino
BUCKET=s3://banino-repro-975050064729/rl_runs
CELLS=("${@:-a3c_goal_grid a3c_goal_place a3c_goal_a3c a3c_goal_grid64}")
[ $# -gt 0 ] && CELLS=("$@")
LEVEL=contributed/dmlab30/explore_goal_locations_small
DOCKER="docker run --rm --shm-size=8g -v /home/ec2-user/banino:/workspace -w /workspace dmlab-rl:cpu"

for c in "${CELLS[@]}"; do
  echo "=== $c: downloading"
  aws s3 cp "$BUCKET/$c/results.tar.gz" "/tmp/$c.tar.gz" >/dev/null 2>&1 || {
    echo "  no synced tarball"; continue; }
  tar xzf "/tmp/$c.tar.gz" -C rl_runs/ || continue
  agent=$(python3 -c "import json;print(json.load(open('rl_runs/$c/config.json'))['agent'])" 2>/dev/null)
  ckpt=rl_runs/$c/ckpt_final.pt
  [ -f "$ckpt" ] || ckpt=$(ls -t rl_runs/$c/ckpt_*.pt 2>/dev/null | head -1)
  [ -f "$ckpt" ] || ckpt=rl_runs/$c/resume.pt
  echo "  agent=$agent ckpt=$ckpt"

  # Grid-ness of the agent's own grid module (Fig 2g analog).
  if ls rl_runs/$c/gridcodes_*.npz >/dev/null 2>&1; then
    .venv/bin/python -m rl.score_gridcells "rl_runs/$c" --arena_half 1.375 \
      > "rl_runs/$c/grid_scores.log" 2>&1 && echo "  grid scores done"
  fi

  # Goal-vector decoding from the policy LSTM (Fig 2j analog). Runs the
  # real env, so it needs the container.
  if [ -f "$ckpt" ]; then
    $DOCKER python3 -m rl.decode_goal --ckpt "$ckpt" --level "$LEVEL" \
      --agent "$agent" --episodes 60 --n_envs 16 --device cpu \
      --out "rl_runs/$c" > "rl_runs/$c/decode.log" 2>&1 \
      && echo "  goal decoding done" || echo "  goal decoding FAILED (see decode.log)"
  fi

  # Eval fallback: if the instance died before its on-box eval, run it here.
  if [ ! -f "rl_runs/$c/eval_scores.json" ] && [ -f "$ckpt" ]; then
    echo "  no eval_scores.json; running 100-episode eval locally"
    $DOCKER python3 -m rl.eval_agent --ckpt "$ckpt" --level "$LEVEL" \
      --agent "$agent" --episodes 100 --n_envs 24 --device cpu \
      --out "rl_runs/$c" > "rl_runs/$c/eval_local.log" 2>&1 \
      && echo "  local eval done" || echo "  local eval FAILED"
  fi

  # Peak-checkpoint eval. Several cells (notably the arena grid agent) peak
  # mid-training and then regress, so the final checkpoint understates what
  # the architecture reached. Evaluate the hourly checkpoint nearest the
  # best training return as well, and report both.
  peak=$(.venv/bin/python - "$c" <<'PY'
import glob, json, os, re, sys
c = sys.argv[1]
rows = [json.loads(l) for l in open(f'rl_runs/{c}/metrics.jsonl') if l.strip()]
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
    echo "  peak checkpoint: $peak"
    $DOCKER python3 -m rl.eval_agent --ckpt "$peak" --level "$LEVEL" \
      --agent "$agent" --episodes 100 --n_envs 24 --device cpu \
      --out "rl_runs/$c/peak" > "rl_runs/$c/eval_peak.log" 2>&1 \
      && echo "  peak eval done" || echo "  peak eval FAILED"
  fi
done

.venv/bin/python -m rl.aggregate_a3c --rl rl_runs --prefix a3c_goal \
  --out A3C_RESULTS.json --md A3C_RESULTS.md
