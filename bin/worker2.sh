#!/bin/zsh
# Overnight worker: claims from the SQLite queue via qctl over SSH, runs the
# job locally with a 60 s heartbeat, syncs the result, marks done/fail.
# usage: worker2.sh <coordinator>   (one process per worker slot)
set -u
COORD=$1
ME=$(hostname -s)
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
QPY='python3 ~/projects/connectome-control/queue/qctl.py'
EMPTY=0
while true; do
  JOB=$(ssh -o BatchMode=yes "$COORD" "$QPY claim $ME" 2>/dev/null)
  RC=$?
  if [ $RC -eq 3 ]; then echo "[$ME] queue HALTED; exiting"; break; fi
  if [ $RC -ne 0 ] || [ -z "$JOB" ]; then
    EMPTY=$((EMPTY+1)); [ $EMPTY -ge 5 ] && break
    sleep 20; continue
  fi
  EMPTY=0
  echo "$JOB" > /tmp/ccjob.$$.json
  JID=$(python3 -c "import json;print(json.load(open('/tmp/ccjob.$$.json'))['id'])")
  echo "[$ME] $(date +%H:%M) running $JID"
  # heartbeat in the background for the duration of the job
  ( while true; do sleep 60
      ssh -o BatchMode=yes "$COORD" "$QPY heartbeat $JID" >/dev/null 2>&1 || true
    done ) & HB=$!
  .venv/bin/python -m connectome_control.jobs /tmp/ccjob.$$.json \
      > "results_logs_$JID.log" 2>&1
  RC=$?
  kill $HB 2>/dev/null; wait $HB 2>/dev/null
  mkdir -p results logs && mv -f "results_logs_$JID.log" "logs/$JID.log"
  if [ $RC -eq 0 ] && [ -f "results/$JID.json" ]; then
    if scp -q "results/$JID.json" "results/$JID.pt" "logs/$JID.log" \
          "${COORD}:cc-queue/results/" ; then
      ssh -o BatchMode=yes "$COORD" "$QPY done $JID" >/dev/null
    else
      # sync failed; leave local result, try again on next contact
      ssh -o BatchMode=yes "$COORD" "$QPY fail $JID 'result sync failed'" >/dev/null
    fi
  else
    scp -q "logs/$JID.log" "${COORD}:cc-queue/results/" 2>/dev/null
    ssh -o BatchMode=yes "$COORD" "$QPY fail $JID 'job exit $RC'" >/dev/null
    echo "[$ME] $JID failed (exit $RC)"
  fi
  rm -f /tmp/ccjob.$$.json
done
echo "[$ME] worker exiting"
