#!/bin/bash
# EC2 user-data: pull artifacts via presigned URLs, resume any previous state
# for this run name, train one RL cell in docker, upload results every 15
# minutes and at exit, then self-terminate. No AWS credentials on the box.
set -x
dnf install -y docker
systemctl start docker
cd /root
curl -fsSL "__IMAGE_URL__" | docker load
curl -fsSL "__REPO_URL__" -o banino-repo.tar.gz
tar xzf banino-repo.tar.gz   # creates /root/banino
OUT=banino/rl_runs/__NAME__
mkdir -p "$OUT"
# Spot-interruption resume: restore previously synced state if it exists.
if curl -fsSL "__GET_RESULTS_URL__" -o prev.tar.gz; then
  tar xzf prev.tar.gz -C banino/rl_runs/ && echo "resumed from synced state"
fi
upload() { tar czf /root/results.tar.gz -C banino/rl_runs __NAME__ && \
  curl -fsS -X PUT -T /root/results.tar.gz "__PUT_URL__"; }
( while true; do sleep 900; upload; done ) &
docker run --rm --shm-size=8g -v /root/banino:/workspace -w /workspace \
  dmlab-rl:cpu python3 -m rl.train_rl \
  --level "__LEVEL__" __EXTRA__ \
  --out rl_runs/__NAME__ --agent __AGENT__ --frames __FRAMES__ \
  --device cpu --resume \
  >> "$OUT/train.log" 2>&1
upload
shutdown -h now
