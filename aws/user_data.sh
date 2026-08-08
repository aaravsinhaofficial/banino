#!/bin/bash
# EC2 user-data: pull artifacts via presigned URLs, run one RL baseline in
# docker, upload results tarball to a reusable presigned PUT URL every 15
# minutes and at exit, then self-terminate. No AWS credentials on the box.
set -x
dnf install -y docker
systemctl start docker
cd /root
curl -fsSL "__IMAGE_URL__" | docker load
curl -fsSL "__REPO_URL__" -o banino-repo.tar.gz
mkdir -p banino && tar xzf banino-repo.tar.gz -C banino
OUT=banino/rl_runs/__NAME__
mkdir -p "$OUT"
upload() { tar czf /root/results.tar.gz -C banino/rl_runs __NAME__ && \
  curl -fsS -X PUT -T /root/results.tar.gz "__PUT_URL__"; }
( while true; do sleep 900; upload; done ) &
docker run --rm --shm-size=8g -v /root/banino:/workspace -w /workspace \
  dmlab-rl:latest python3 -m rl.train_rl \
  --level contributed/dmlab30/explore_goal_locations_small \
  --out rl_runs/__NAME__ --agent __AGENT__ --frames __FRAMES__ --device cpu \
  > "$OUT/train.log" 2>&1
upload
shutdown -h now
