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

# The split lists to stage: a directory in the repo holding images_*.txt.
SPLIT_DIR="${SPLIT_DIR:-adobe5k_dpe}"
# The code archive in $BUCKET to run. Build one with
#   git archive --format=tar.gz -o code.tgz HEAD && aws s3 cp code.tgz $BUCKET/
CODE_TGZ="${CODE_TGZ:-code.tgz}"

# arm name : --fixes value [: extra main.py flags]. Override with ARMS_SPEC,
# one entry per line; the extra-flags field may itself contain spaces, which is
# why the split is on newlines and not on whitespace.
ARMS=(
  "baseline:none"
  "wiring:wiring"
  "ellipse:ellipse"
  "ste:ste"
  "msssim:msssim"
  "fusion:fusion"
)

# Not mapfile: macOS ships bash 3.2, which does not have it.
if [ -n "${ARMS_SPEC:-}" ]; then
  ARMS=()
  while IFS= read -r line; do
    [ -n "$line" ] && ARMS+=("$line")
  done <<< "$ARMS_SPEC"
fi

read -ra SUBNET_ARR <<< "$SUBNETS"
i=0
for entry in "${ARMS[@]}"; do
  NAME="${entry%%:*}"
  rest="${entry#*:}"
  SPEC="${rest%%:*}"
  # Empty for arms with no extra flags, i.e. entries with only two fields.
  EXTRA=""
  [ "$rest" != "$SPEC" ] && EXTRA="${rest#*:}"
  SUBNET="${SUBNET_ARR[$((i % ${#SUBNET_ARR[@]}))]}"
  i=$((i + 1))

  sed -e "s|__ARM_NAME__|$NAME|" \
      -e "s|__FIX_SPEC__|$SPEC|" \
      -e "s|__EPOCHS__|$EPOCHS|" \
      -e "s|__HARD_CAP__|$HARD_CAP|" \
      -e "s|__BUCKET__|$BUCKET|" \
      -e "s|__SEED__|$SEED|" \
      -e "s|__EXTRA_ARGS__|$EXTRA|" \
      -e "s|__SPLIT_DIR__|$SPLIT_DIR|" \
      -e "s|__CODE_TGZ__|$CODE_TGZ|" \
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
  printf "%-12s --fixes=%-28s %-12s %s (%s)\n" "$NAME" "$SPEC" "$EXTRA" "$ID" "$SUBNET"
done
