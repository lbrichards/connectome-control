#!/bin/zsh
# Batch-2 chain: wait for relay assets -> commit/deploy -> pilot gate ->
# load full tiers -> start scheduler + workers + pilot. Halts on gate fail.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while kill -0 $(cat /tmp/relay_pid) 2>/dev/null; do sleep 30; done
grep -q "relay assets complete" relay_assets.log || {
  echo "[batch2] RELAY ASSET GATES FAILED"; tail -5 relay_assets.log; exit 1; }
git add -A; git -c user.name="Larry Richards" -c user.email="larry@source1.jp" \
  commit -q -m "batch2: relay-distillation tier 1, quiet-hold gate, calibrated tier-2 target

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"; git push -q
for H in mbp mba; do ./bin/deploy.sh $H || exit 1; done
echo "[batch2] pilot gate..."
$PY queue/batch2.py pilot > batch2_pilot.log 2>&1
RC=$?
cat batch2_pilot.log
[ $RC -ne 0 ] && { echo "[batch2] PILOT GATE FAILED -- halting"; exit 1; }
TARGET=$(grep -o "load [0-9.]*" batch2_pilot.log | awk '{print $2}')
echo "[batch2] loading full tiers with target $TARGET"
$PY queue/batch2.py load "$TARGET"
rm -f ~/cc-queue/TIER4_EMITTED ~/cc-queue/FIRST_PASS_DONE ~/cc-queue/HALT
nohup caffeinate -is $PY queue/scheduler.py > /dev/null 2>&1 &
mkdir -p logs
P_LOCAL=$(( $(sysctl -n hw.perflevel0.physicalcpu) - 1 ))
for i in $(seq 1 $P_LOCAL); do
  nohup caffeinate -is ./bin/worker2.sh macmini > logs/b2_local_$i.log 2>&1 &
done
for H in mbp mba; do
  P=$(ssh $H sysctl -n hw.perflevel0.physicalcpu)
  case $H in mba) P=$(( (P+1)/2 ));; esac
  for i in $(seq 1 $P); do
    ssh $H "cd ~/projects/connectome-control && mkdir -p logs && nohup caffeinate -is ./bin/worker2.sh macmini > logs/b2_$i.log 2>&1 & disown"
  done
  echo "[batch2] $H: $P workers"
done
nohup $PY queue/pilot.py > /dev/null 2>&1 &
echo "[batch2] LAUNCHED"
