#!/bin/bash
# Status of every DeepLPF ablation arm: instance state, epoch, loss, best PSNR.
# Instances are found by tag, so no instance ids or addresses are hardcoded.

REGION="${REGION:-us-east-1}"
PROFILE="${PROFILE:-default}"
KEY="${KEY:-$HOME/.ssh/founders-workbench.pem}"
P="--region $REGION --profile $PROFILE"

# Only live instances. Terminated ones linger in describe-instances for about
# an hour, so after a couple of relaunches the unfiltered listing shows several
# generations at once and the current run is impossible to pick out.
printf "%-9s %-20s %s\n" ARM INSTANCE PROGRESS
aws ec2 describe-instances $P \
  --filters "Name=tag:Project,Values=deeplpf-v2-ablation" \
            "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[].Instances[].[Tags[?Key==`Arm`].Value|[0],InstanceId,State.Name,PublicIpAddress]' \
  --output text 2>/dev/null | sort | while read -r arm id state ip; do
  if [ "$state" != "running" ]; then
    printf "%-9s %-20s %s\n" "$arm" "$id" "$state"
    continue
  fi
  # -n matters: without it ssh consumes the loop's stdin and the loop stops.
  info=$(ssh -n -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 \
             -o BatchMode=yes -i "$KEY" ubuntu@"$ip" '
    D=/opt/dlami/nvme/deeplpf/out
    E=$(grep -hc "train loss" $D/log_*/deep_lpf.log 2>/dev/null)
    L=$(grep -h "train loss" $D/log_*/deep_lpf.log 2>/dev/null | tail -1 | sed "s/.*loss: //" | cut -c1-8)
    V=$(ls $D/log_*/ 2>/dev/null | grep -oE "validpsnr_[0-9.]+" | sed "s/validpsnr_//" | sort -g | tail -1)
    F=$(grep -h "fixes enabled" $D/log_*/deep_lpf.log 2>/dev/null | sed "s/.*enabled: //" | head -1)
    S=$(grep -h "Seed:" $D/log_*/deep_lpf.log 2>/dev/null | sed "s/.*Seed: //" | head -1)
    G=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)
    echo "ep=${E:-0} loss=${L:-?} psnr=${V:-none} gpu=${G:-?} [${F:-?}, seed ${S:-?}]"' 2>/dev/null)
  printf "%-9s %-20s %s\n" "$arm" "$id" "${info:-booting}"
done

N=$(aws ec2 describe-instances $P \
      --filters "Name=tag:Project,Values=deeplpf-v2-ablation" "Name=instance-state-name,Values=running" \
      --query 'length(Reservations[].Instances[])' --output text)
# g5.xlarge on-demand, us-east-1.
echo "running: $N   burn: \$$(echo "$N" | awk '{printf "%.2f", $1 * 1.006}')/hr"
