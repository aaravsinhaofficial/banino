#!/bin/bash
# Launch one pirnn RL cell on an ON-DEMAND EC2 instance (deadline runs:
# no spot churn). Usage:
#   ./provision_pirnn.sh <name> <pgc_s3_key> <frames> [level] [extra-args]
# Artifacts required in S3: dmlab-rl.tar.gz, banino-repo-pirnn.tar.gz,
# <pgc_s3_key> (an RNN state_dict). Tagged Project=banino-repro.
set -euo pipefail
NAME=${1:?name}; PGC_KEY=${2:?pgc s3 key}; FRAMES=${3:-400000000}
LEVEL=${4:-contributed/dmlab30/explore_obstructed_goals_small}
EXTRA=${5:-}
PROFILE=banino-repro; REGION=us-east-1
PY=/home/ec2-user/banino/.venv/bin/python
ITYPE=c7i.16xlarge          # 64 vCPU on-demand ~$3.4/h
AMI=$(aws ec2 describe-images --profile $PROFILE --region $REGION \
  --owners amazon \
  --filters 'Name=name,Values=al2023-ami-2023*-kernel-*-x86_64' \
            'Name=state,Values=available' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
IMAGE_URL=$($PY aws/presign.py get dmlab-rl.tar.gz)
REPO_URL=$($PY aws/presign.py get banino-repo-pirnn.tar.gz)
PGC_URL=$($PY aws/presign.py get "$PGC_KEY")
PUT_URL=$($PY aws/presign.py put "rl_runs/$NAME/results.tar.gz")
GET_RESULTS_URL=$($PY aws/presign.py get "rl_runs/$NAME/results.tar.gz")

USER_DATA=$(IMAGE_URL="$IMAGE_URL" REPO_URL="$REPO_URL" PGC_URL="$PGC_URL" \
  PUT_URL="$PUT_URL" GET_RESULTS_URL="$GET_RESULTS_URL" NAME="$NAME" \
  FRAMES="$FRAMES" LEVEL="$LEVEL" EXTRA="$EXTRA" $PY - <<'EOF'
import base64, os
s = open('aws/user_data_pirnn.sh').read()
for k in ['IMAGE_URL', 'REPO_URL', 'PGC_URL', 'PUT_URL', 'GET_RESULTS_URL',
          'NAME', 'FRAMES', 'LEVEL', 'EXTRA']:
  s = s.replace('__%s__' % k, os.environ[k])
print(base64.b64encode(s.encode()).decode())
EOF
)

aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id "$AMI" --instance-type $ITYPE \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=banino-repro},{Key=Name,Value=banino-$NAME}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text
