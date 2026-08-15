#!/bin/bash
# EC2 user-data for one async-A3C cell: pull artifacts via presigned URLs,
# resume any synced state for this run name, train until the global
# train-until timestamp, run the SI's 100-episode frozen-policy eval,
# upload, self-terminate. No AWS credentials on the box.
set -x
# Hard cost failsafe: power off even if everything below wedges
# (instance-initiated-shutdown-behavior is terminate).
shutdown -h +__FAILSAFE_MIN__
dnf install -y docker
systemctl start docker
cd /root
curl -fsSL "__IMAGE_URL__" | docker load
curl -fsSL "__REPO_URL__" -o banino-repo.tar.gz
tar xzf banino-repo.tar.gz   # creates /root/banino
OUT=banino/rl_runs/__NAME__
mkdir -p "$OUT"
# Relaunch resume: restore previously synced state if it exists.
if curl -fsSL "__GET_RESULTS_URL__" -o prev.tar.gz; then
  tar xzf prev.tar.gz -C banino/rl_runs/ && echo "resumed from synced state"
fi
upload() { tar czf /root/results.tar.gz -C banino/rl_runs __NAME__ && \
  curl -fsS -X PUT -T /root/results.tar.gz "__PUT_URL__"; }
( while true; do sleep 900; upload; done ) &
MAXS=$(( __TRAIN_UNTIL_EPOCH__ - $(date +%s) ))
if [ "$MAXS" -gt 300 ]; then
  docker run --rm --shm-size=__SHM__ --ulimit nofile=65535:65535 \
    -v /root/banino:/workspace -w /workspace \
    dmlab-rl:cpu python3 -m rl.train_a3c \
    --level "__LEVEL__" __EXTRA__ \
    --out rl_runs/__NAME__ --agent __AGENT__ --frames __FRAMES__ \
    --workers __WORKERS__ --replay_total __REPLAY_TOTAL__ \
    --max_seconds "$MAXS" --resume \
    >> "$OUT/train.log" 2>&1
fi
upload
# Frozen-policy benchmark eval (SI protocol: mean score over 100 episodes).
CKPT="rl_runs/__NAME__/ckpt_final.pt"
[ -f "banino/$CKPT" ] || CKPT="rl_runs/__NAME__/resume.pt"
if [ -f "banino/$CKPT" ]; then
  docker run --rm --shm-size=8g -v /root/banino:/workspace -w /workspace \
    dmlab-rl:cpu python3 -m rl.eval_agent --ckpt "$CKPT" \
    --level "__LEVEL__" --agent __AGENT__ --episodes 100 --n_envs 16 \
    --device cpu --out rl_runs/__NAME__ \
    >> "$OUT/eval.log" 2>&1
fi
upload
shutdown -h now
