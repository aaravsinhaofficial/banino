#!/bin/bash
# Pull just the metrics/config of each running cell from its synced S3
# tarball into rl_runs/<cell>/, for mid-run plotting. Does not touch
# checkpoints (finalize_a3c.sh does the full download at the end).
set -u
export AWS_PROFILE=banino-repro AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/banino
BUCKET=s3://banino-repro-975050064729/rl_runs
CELLS=(a3c_goal_grid a3c_goal_place a3c_goal_a3c a3c_goal_grid_hi
       a3c_seekavoid a3c_arena_grid a3c_arena_place)
[ $# -gt 0 ] && CELLS=("$@")
for c in "${CELLS[@]}"; do
  mkdir -p "rl_runs/$c"
  aws s3 cp "$BUCKET/$c/results.tar.gz" - 2>/dev/null \
    | tar xz -C rl_runs/ "$c/metrics.jsonl" "$c/config.json" 2>/dev/null \
    && echo "$c: $(wc -l < "rl_runs/$c/metrics.jsonl") points" \
    || echo "$c: not synced yet"
done
