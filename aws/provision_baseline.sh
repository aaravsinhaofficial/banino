#!/bin/bash
# Launch one RL cell on an EC2 spot instance.
# Usage: ./provision_baseline.sh <name> <agent> <frames> [level] [extra-args]
# Requires artifacts in S3 (docker image + repo tarball). Spot, tagged
# Project=banino-repro, self-terminating, resume-on-boot from synced state,
# no credentials on the instance (presigned URLs).
set -euo pipefail
NAME=${1:?name}; AGENT=${2:?agent}; FRAMES=${3:-1000000000}
LEVEL=${4:-contributed/dmlab30/explore_goal_locations_small}
EXTRA=${5:-}
PROFILE=banino-repro; REGION=us-east-1
PY=/home/ec2-user/banino/.venv/bin/python
ITYPE=c7i.16xlarge          # 64 vCPU; spot ~= $1.1/h in us-east-1
# ssm:GetParameter is denied for this user; resolve the AMI via EC2 instead.
AMI=$(aws ec2 describe-images --profile $PROFILE --region $REGION \
  --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023*-kernel-*-x86_64' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
IMAGE_URL=$($PY aws/presign.py get dmlab-rl.tar.gz)
REPO_URL=$($PY aws/presign.py get banino-repo.tar.gz)
PUT_URL=$($PY aws/presign.py put "rl_runs/$NAME/results.tar.gz")
GET_RESULTS_URL=$($PY aws/presign.py get "rl_runs/$NAME/results.tar.gz")

# Substitution via python: presigned URLs contain sed metacharacters.
USER_DATA=$(IMAGE_URL="$IMAGE_URL" REPO_URL="$REPO_URL" PUT_URL="$PUT_URL" \
  GET_RESULTS_URL="$GET_RESULTS_URL" AGENT="$AGENT" NAME="$NAME" \
  FRAMES="$FRAMES" LEVEL="$LEVEL" EXTRA="$EXTRA" $PY - <<'EOF'
import base64, os
s = open('aws/user_data.sh').read()
for k in ['IMAGE_URL', 'REPO_URL', 'PUT_URL', 'GET_RESULTS_URL', 'AGENT',
          'NAME', 'FRAMES', 'LEVEL', 'EXTRA']:
  s = s.replace('__%s__' % k, os.environ[k])
print(base64.b64encode(s.encode()).decode())
EOF
)

aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id "$AMI" --instance-type $ITYPE \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=banino-repro},{Key=Name,Value=banino-$NAME}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text
