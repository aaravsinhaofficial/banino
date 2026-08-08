#!/bin/bash
# Launch one RL baseline run on an EC2 spot instance.
# Usage: ./provision_baseline.sh <name> <agent:placecell|a3c|grid> [frames]
# Requires artifacts in S3 (aws/push_artifacts.sh). Tagged Project=banino-repro;
# spot, self-terminating, no credentials on instance (presigned URLs).
set -euo pipefail
NAME=${1:?name}; AGENT=${2:?agent}; FRAMES=${3:-20000000}
PROFILE=banino-repro; REGION=us-east-1
PY=/home/ec2-user/banino/.venv/bin/python
ITYPE=c7i.16xlarge          # 64 vCPU; spot ~= $1.1/h in us-east-1
AMI=$(aws ssm get-parameter --profile $PROFILE --region $REGION \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameter.Value' --output text)
IMAGE_URL=$($PY aws/presign.py get dmlab-rl.tar.gz)
REPO_URL=$($PY aws/presign.py get banino-repo.tar.gz)
PUT_URL=$($PY aws/presign.py put "rl_runs/$NAME/results.tar.gz")

USER_DATA=$(sed -e "s|__IMAGE_URL__|$IMAGE_URL|" -e "s|__REPO_URL__|$REPO_URL|" \
  -e "s|__PUT_URL__|$PUT_URL|" -e "s|__AGENT__|$AGENT|" -e "s|__NAME__|$NAME|" \
  -e "s|__FRAMES__|$FRAMES|" aws/user_data.sh | base64 -w0)

aws ec2 run-instances --profile $PROFILE --region $REGION \
  --image-id "$AMI" --instance-type $ITYPE \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=banino-repro},{Key=Name,Value=banino-$NAME}]" \
  --user-data "$USER_DATA" \
  --query 'Instances[0].InstanceId' --output text
