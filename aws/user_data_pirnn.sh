#!/bin/bash
# EC2 user-data: one pirnn RL cell (frozen path-integration RNN as grid
# code). Pull artifacts via presigned URLs (image + repo + RNN checkpoint),
# resume synced state if present, train in docker, sync results every 10
# minutes, hard-stop after 200 minutes as a cost backstop. No credentials
# on the instance.
set -x
dnf install -y docker
systemctl start docker
cd /root
curl -fsSL "__IMAGE_URL__" | docker load
curl -fsSL "__REPO_URL__" -o banino-repo.tar.gz
tar xzf banino-repo.tar.gz   # creates /root/banino
mkdir -p pgc
curl -fsSL "__PGC_URL__" -o pgc/rnn.pth
OUT=banino/rl_runs/__NAME__
mkdir -p "$OUT"
if curl -fsSL "__GET_RESULTS_URL__" -o prev.tar.gz; then
  tar xzf prev.tar.gz -C banino/rl_runs/ && echo "resumed from synced state"
fi
upload() { tar czf /root/results.tar.gz -C banino/rl_runs __NAME__ && \
  curl -fsS -X PUT -T /root/results.tar.gz "__PUT_URL__"; }
( while true; do sleep 600; upload; done ) &
shutdown -h +200   # hard cost/deadline backstop
docker run --rm --shm-size=8g -v /root/banino:/workspace -v /root/pgc:/pgc \
  -w /workspace dmlab-rl:cpu python3 -m rl.train_rl \
  --level "__LEVEL__" __EXTRA__ \
  --pirnn_ckpt /pgc/rnn.pth \
  --out rl_runs/__NAME__ --agent pirnn --frames __FRAMES__ \
  --device cpu --resume \
  >> "$OUT/train.log" 2>&1
upload
shutdown -h now
