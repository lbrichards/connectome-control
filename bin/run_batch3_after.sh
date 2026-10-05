#!/bin/zsh
# Waits for batch 2 to fully drain, then: build quality relays (calibrated),
# deploy assets, load batch 3, restart fleet. Honors HALT.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while true; do
  STATE=$($PY queue/qctl.py status 2>/dev/null)
  echo "$STATE" | grep -qE '"(pending|running)"' || break
  sleep 300
done
[ -f ~/cc-queue/HALT ] && { echo "[batch3] HALT present; not starting"; exit 1; }
echo "[batch3] batch 2 drained; building quality relays..."
$PY -u -m connectome_control.relay quality > relay_quality.log 2>&1 || {
  echo "[batch3] quality relay build FAILED"; exit 1; }
git add -A; git -c user.name="Larry Richards" -c user.email="larry@source1.jp" \
  commit -q -m "batch3: teacher-quality sweep assets (q91/q80/q67)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" 2>/dev/null; git push -q 2>/dev/null
for H in $(cat ~/cc-queue/hosts.txt); do ./bin/deploy.sh $H || exit 1; done
$PY queue/batch3.py /tmp/b3jobs.jsonl
$PY queue/qctl.py load /tmp/b3jobs.jsonl
rm -f ~/cc-queue/TIER4_EMITTED ~/cc-queue/FIRST_PASS_DONE
pgrep -f "queue/scheduler.py" >/dev/null || \
  nohup caffeinate -is $PY queue/scheduler.py > /dev/null 2>&1 &
for i in 1 2 3; do nohup caffeinate -is ./bin/worker2.sh macmini > logs/b3_local_$i.log 2>&1 & done
for H in $(cat ~/cc-queue/hosts.txt); do
  P=$(ssh $H sysctl -n hw.perflevel0.physicalcpu)
  case $H in mba) P=$(( (P+1)/2 ));; esac
  for i in $(seq 1 $P); do
    ssh $H "cd ~/projects/connectome-control && nohup caffeinate -is ./bin/worker2.sh macmini > logs/b3_$i.log 2>&1 & disown"
  done
done
echo "[batch3] LAUNCHED (120 jobs, 3 teacher-quality levels)"
