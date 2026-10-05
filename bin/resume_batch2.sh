#!/bin/zsh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
git add -A; git -c user.name="Larry Richards" -c user.email="larry@source1.jp" \
  commit -q -m "web engine + parity green; batch2 infra; deploy hardened

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null; git push -q 2>/dev/null
for H in mbp mba; do ./bin/deploy.sh $H || exit 1; done
echo "[resume] pilot gate (8 jobs, local parallel)..."
$PY queue/batch2.py pilot > batch2_pilot.log 2>&1
RC=$?
cat batch2_pilot.log
[ $RC -ne 0 ] && { echo "[resume] PILOT GATE FAILED -- halted"; exit 1; }
TARGET=$(grep -o "load [0-9.]*" batch2_pilot.log | awk '{print $2}')
$PY queue/batch2.py load "$TARGET"
rm -f ~/cc-queue/TIER4_EMITTED ~/cc-queue/FIRST_PASS_DONE ~/cc-queue/HALT
nohup caffeinate -is $PY queue/scheduler.py > /dev/null 2>&1 &
mkdir -p logs
for i in 1 2 3; do nohup caffeinate -is ./bin/worker2.sh macmini > logs/b2_local_$i.log 2>&1 & done
for H in mbp mba; do
  P=$(ssh $H sysctl -n hw.perflevel0.physicalcpu)
  case $H in mba) P=$(( (P+1)/2 ));; esac
  for i in $(seq 1 $P); do
    ssh $H "cd ~/projects/connectome-control && mkdir -p logs && nohup caffeinate -is ./bin/worker2.sh macmini > logs/b2_$i.log 2>&1 & disown"
  done
done
nohup $PY queue/pilot.py > /dev/null 2>&1 &
echo "[resume] BATCH 2 LAUNCHED (target $TARGET)"
