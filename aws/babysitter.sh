#!/bin/bash
# Keeps the six full-scale AWS cells alive: every 10 minutes, relaunch any
# cell with no pending/running instance (user-data resumes from synced S3
# state). Stops babysitting a cell once its synced metrics reach the frame
# target. Run inside tmux: bash aws/babysitter.sh >> rl_runs/babysitter.log 2>&1
set -u
export AWS_PROFILE=banino-repro
cd /home/ec2-user/banino
TARGET=1000000000
declare -A SPEC=(
  [fs_goal_grid]="grid $TARGET contributed/dmlab30/explore_goal_locations_small"
  [fs_goal_place]="placecell $TARGET contributed/dmlab30/explore_goal_locations_small"
  [fs_goal_a3c]="a3c $TARGET contributed/dmlab30/explore_goal_locations_small"
  [fs_doors_grid]="grid $TARGET contributed/dmlab30/explore_obstructed_goals_small"
  [fs_doors_place]="placecell $TARGET contributed/dmlab30/explore_obstructed_goals_small"
  [fs_doors_a3c]="a3c $TARGET contributed/dmlab30/explore_obstructed_goals_small"
)
done_cell() {  # true once synced metrics show >= TARGET frames
  aws s3 cp "s3://banino-repro-975050064729/rl_runs/$1/results.tar.gz" - 2>/dev/null \
    | tar xzO "$1/metrics.jsonl" 2>/dev/null | tail -1 \
    | grep -o '"frames": [0-9]*' | grep -o '[0-9]*' \
    | awk -v t=$TARGET '{exit !($1 >= t)}'
}
while true; do
  for name in "${!SPEC[@]}"; do
    read -r agent frames level <<< "${SPEC[$name]}"
    state=$(aws ec2 describe-instances \
      --filters "Name=tag:Name,Values=banino-$name" \
                "Name=instance-state-name,Values=pending,running" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
    if [ -z "$state" ]; then
      if done_cell "$name"; then
        echo "$(date -u +%H:%M) $name: complete"
      else
        id=$(bash aws/provision_baseline.sh "$name" "$agent" "$frames" "$level")
        echo "$(date -u +%H:%M) $name: relaunched $id"
      fi
    fi
  done
  sleep 600
done
