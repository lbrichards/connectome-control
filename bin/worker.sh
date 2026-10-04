#!/bin/zsh
# Pull-based worker. Runs on ANY machine (including the coordinator).
# Loops: claim a job from the coordinator's queue over SSH, run it locally,
# push the result back, mark done. Exits when the queue stays empty.
#
# usage: worker.sh <coordinator-ssh-alias> [nproc]
#   nproc is informational; run one worker.sh per desired slot.
set -u
COORD=$1
ME=$(hostname -s)
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
EMPTY=0
while true; do
  JOB_JSON=$(ssh -o BatchMode=yes "$COORD" "QUEUE_DIR=\$HOME/cc-queue zsh ~/cc-queue/claim.sh $ME" 2>/tmp/claim_name.$$)
  if [ $? -ne 0 ] || [ -z "$JOB_JSON" ]; then
    EMPTY=$((EMPTY+1))
    [ $EMPTY -ge 3 ] && break
    sleep 5
    continue
  fi
  EMPTY=0
  FNAME=$(cat /tmp/claim_name.$$)
  echo "$JOB_JSON" > /tmp/job.$$.json
  JID=$(python3 -c "import json,sys;print(json.load(open('/tmp/job.$$.json'))['id'])")
  echo "[$ME] running $JID"
  .venv/bin/python -m connectome_control.jobs /tmp/job.$$.json
  if [ $? -eq 0 ] && [ -f "results/$JID.json" ]; then
    scp -q "results/$JID.json" "results/$JID.pt" "${COORD}:cc-queue/results/"
    ssh -o BatchMode=yes "$COORD" "mv ~/cc-queue/claimed/$ME.$FNAME ~/cc-queue/done/ 2>/dev/null"
  else
    # return the job to the queue for someone else
    ssh -o BatchMode=yes "$COORD" "mv ~/cc-queue/claimed/$ME.$FNAME ~/cc-queue/pending/$FNAME 2>/dev/null"
    echo "[$ME] $JID FAILED; returned to queue"
    sleep 10
  fi
  rm -f /tmp/job.$$.json /tmp/claim_name.$$
done
echo "[$ME] queue empty; worker exiting"
