#!/bin/bash
# Launch one async-A3C RL cell on EC2 (default on-demand: these runs have a
# hard deadline, and the whole fleet costs ~$9/h).
# Usage: ./provision_a3c.sh <name> <agent> <train_until_epoch> \
#          [level] [workers] [itype] [market] [frames] [extra-args]
# Requires artifacts in S3 (docker image + repo tarball); tagged
# Project=banino-repro, self-terminating, resume-on-boot from synced state,
# no credentials on the instance (presigned URLs).
set -euo pipefail
NAME=${1:?name}; AGENT=${2:?agent}; UNTIL=${3:?train-until epoch seconds}
LEVEL=${4:-contributed/dmlab30/explore_goal_locations_small}
WORKERS=${5:-32}
ITYPE=${6:-c7i.16xlarge}
MARKET=${7:-ondemand}          # ondemand | spot
FRAMES=${8:-1000000000}
EXTRA=${9:-}
PROFILE=banino-repro; REGION=us-east-1
PY=/home/ec2-user/banino/.venv/bin/python
cd /home/ec2-user/banino

NOW=$(date +%s)
# Failsafe poweroff: train window + 100 min for boot/eval/upload slack.
FAILSAFE_MIN=$(( (UNTIL - NOW) / 60 + 100 ))
[ "$FAILSAFE_MIN" -lt 110 ] && FAILSAFE_MIN=110
# /dev/shm must hold the actor-owned replay (~21.2 KB/step) plus slack.
SHM_GB=$(( WORKERS * 46875 * 22 / 1000000000 + 6 ))

AMI=$(aws ec2 describe-images --profile $PROFILE --region $REGION \
  --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023*-kernel-*-x86_64' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
IMAGE_URL=$($PY aws/presign.py get dmlab-rl.tar.gz)
REPO_URL=$($PY aws/presign.py get banino-repo-a3c.tar.gz)
PUT_URL=$($PY aws/presign.py put "rl_runs/$NAME/results.tar.gz")
GET_RESULTS_URL=$($PY aws/presign.py get "rl_runs/$NAME/results.tar.gz")

USER_DATA=$(IMAGE_URL="$IMAGE_URL" REPO_URL="$REPO_URL" PUT_URL="$PUT_URL" \
  GET_RESULTS_URL="$GET_RESULTS_URL" AGENT="$AGENT" NAME="$NAME" \
  FRAMES="$FRAMES" LEVEL="$LEVEL" EXTRA="$EXTRA" WORKERS="$WORKERS" \
  TRAIN_UNTIL_EPOCH="$UNTIL" FAILSAFE_MIN="$FAILSAFE_MIN" \
  SHM="${SHM_GB}g" $PY - <<'EOF'
import base64, os
s = open('aws/user_data_a3c.sh').read()
for k in ['IMAGE_URL', 'REPO_URL', 'PUT_URL', 'GET_RESULTS_URL', 'AGENT',
          'NAME', 'FRAMES', 'LEVEL', 'EXTRA', 'WORKERS',
          'TRAIN_UNTIL_EPOCH', 'FAILSAFE_MIN', 'SHM']:
  s = s.replace('__%s__' % k, os.environ[k])
print(base64.b64encode(s.encode()).decode())
EOF
)

MARKET_ARGS=()
if [ "$MARKET" = spot ]; then
  MARKET_ARGS=(--instance-market-options
    'MarketType=spot,SpotOptions={SpotInstanceType=one-time}')
fi

aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id "$AMI" --instance-type "$ITYPE" \
  "${MARKET_ARGS[@]}" \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=banino-repro},{Key=Name,Value=banino-$NAME}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text
