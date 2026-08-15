#!/bin/bash
# Keeps the async-A3C cells alive until the run deadline: every 10 minutes,
# relaunch any cell with no pending/running instance (user-data resumes from
# synced S3 state and computes its own remaining train window from the
# shared train-until epoch). A cell is finished once its synced tarball
# contains eval_scores.json — after the train-until time, a relaunch skips
# training and just runs the eval, so late crashes still produce a score.
# Run inside tmux:
#   bash aws/babysitter_a3c.sh <train_until_epoch> \
#     "name:agent:level:workers:itype" ... >> rl_runs/babysitter_a3c.log 2>&1
set -u
export AWS_PROFILE=banino-repro AWS_DEFAULT_REGION=us-east-1
cd /home/ec2-user/banino
UNTIL=${1:?train-until epoch}; shift
CELLS=("$@")
declare -A RELAUNCHES

finished() {  # true once the synced tarball holds an eval result
  aws s3 cp "s3://banino-repro-975050064729/rl_runs/$1/results.tar.gz" - \
    2>/dev/null | tar tz 2>/dev/null | grep -q "^$1/eval_scores.json"
}

while true; do
  now=$(date +%s)
  if [ "$now" -gt $((UNTIL + 9000)) ]; then
    echo "$(date -u +%H:%M) deadline + eval window passed; exiting"
    exit 0
  fi
  all_done=1
  for cell in "${CELLS[@]}"; do
    IFS=: read -r name agent level workers itype <<< "$cell"
    if finished "$name"; then
      echo "$(date -u +%H:%M) $name: finished"
      continue
    fi
    all_done=0
    state=$(aws ec2 describe-instances \
      --filters "Name=tag:Name,Values=banino-$name" \
                "Name=instance-state-name,Values=pending,running" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
    if [ -z "$state" ]; then
      n=${RELAUNCHES[$name]:-0}
      if [ "$n" -ge 15 ]; then
        echo "$(date -u +%H:%M) $name: relaunch limit reached; giving up"
        continue
      fi
      RELAUNCHES[$name]=$((n + 1))
      id=$(bash aws/provision_a3c.sh "$name" "$agent" "$UNTIL" "$level" \
             "$workers" "$itype" ondemand 2>&1 | tail -1)
      echo "$(date -u +%H:%M) $name: relaunched ($id)"
    fi
  done
  [ "$all_done" = 1 ] && { echo "$(date -u +%H:%M) all cells finished"; exit 0; }
  sleep 600
done
