#!/bin/bash
# Launch the five-arm DeepLPF v2 ablation, one GPU instance per arm.
#
# Each arm runs identical code and differs only in --fixes, so the baseline is
# the same code path as every treatment. See docs/V2_ABLATION.md.
#
# Prerequisites: an S3 bucket holding code.tgz (git archive of the revision to
# run) and data.tgz (adobe5k_dpe_data + adobe5k_dpe), an EC2 key pair, a
# security group allowing SSH from your address, and an instance profile with
# S3 read access.
#
# Everything is configurable by environment variable; nothing is hardcoded to
# one account.

set -euo pipefail

: "${BUCKET:?set BUCKET, e.g. s3://my-bucket}"
: "${KEY_NAME:?set KEY_NAME, an EC2 key pair name}"
: "${SECURITY_GROUP:?set SECURITY_GROUP, e.g. sg-0123456789abcdef0}"
: "${SUBNETS:?set SUBNETS, space-separated subnet ids to spread arms across}"
INSTANCE_PROFILE="${INSTANCE_PROFILE:-EC2Role}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-default}"
EPOCHS="${EPOCHS:-500}"
# One seed for every arm: arms must differ by their flag alone, not by luck.
SEED="${SEED:-42}"
HARD_CAP="${HARD_CAP:-48}"
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch (Ubuntu 24.04).
AMI="${AMI:?set AMI, a Deep Learning AMI id for your region}"

HERE=$(cd "$(dirname "$0")" && pwd)
TMPL=$HERE/arm_userdata.sh.tmpl
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# arm name : --fixes value
ARMS=(
  "baseline:none"
  "wiring:wiring"
  "ellipse:ellipse"
  "ste:ste"
  "msssim:msssim"
  "fusion:fusion"
)

read -ra SUBNET_ARR <<< "$SUBNETS"
i=0
for entry in "${ARMS[@]}"; do
  NAME="${entry%%:*}"
  SPEC="${entry##*:}"
  SUBNET="${SUBNET_ARR[$((i % ${#SUBNET_ARR[@]}))]}"
  i=$((i + 1))

  sed -e "s|__ARM_NAME__|$NAME|" \
      -e "s|__FIX_SPEC__|$SPEC|" \
      -e "s|__EPOCHS__|$EPOCHS|" \
      -e "s|__HARD_CAP__|$HARD_CAP|" \
      -e "s|__BUCKET__|$BUCKET|" \
      -e "s|__SEED__|$SEED|" \
      "$TMPL" > "$WORKDIR/ud_$NAME.sh"

  ID=$(aws ec2 run-instances --region "$REGION" --profile "$PROFILE" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" --associate-public-ip-address \
    --iam-instance-profile "Name=$INSTANCE_PROFILE" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=deeplpf-arm-$NAME},{Key=Project,Value=deeplpf-v2-ablation},{Key=Arm,Value=$NAME}]" \
    --user-data "file://$WORKDIR/ud_$NAME.sh" \
    --query 'Instances[0].InstanceId' --output text)
  printf "%-9s --fixes=%-9s %s (%s)\n" "$NAME" "$SPEC" "$ID" "$SUBNET"
done
