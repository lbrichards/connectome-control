#!/bin/zsh
# Entry-criteria check + launch. Refuses to start unless everything passes.
# usage: launch_overnight.sh mbp mba   (run on macmini)
set -u
cd "$(dirname "$0")/.."
HOSTS=("$@")
PASS=1
echo "== ENTRY CRITERIA =="
./bin/preflight.sh "${HOSTS[@]}" || PASS=0
.venv/bin/python -m connectome_control.datasets verify || PASS=0
for H in "${HOSTS[@]}"; do
  ssh "$H" 'cd ~/projects/connectome-control && .venv/bin/python -m connectome_control.datasets verify' || PASS=0
  ssh "$H" 'df -h / | tail -1'
done
[ $PASS -eq 1 ] || { echo "ENTRY CRITERIA FAILED -- not launching"; exit 1; }
echo "== LAUNCH =="
mkdir -p ~/cc-queue/results
.venv/bin/python queue/qctl.py init
.venv/bin/python queue/make_jobs.py /tmp/jobs.jsonl
.venv/bin/python queue/qctl.py load /tmp/jobs.jsonl
nohup caffeinate -is .venv/bin/python queue/scheduler.py > /dev/null 2>&1 &
echo "scheduler pid $!"
# workers: perf cores; macmini -1 (scheduler), mba half
start_workers() {  # host n
  local H=$1 N=$2
  for i in $(seq 1 $N); do
    if [ "$H" = "local" ]; then
      nohup caffeinate -is ./bin/worker2.sh macmini > logs/worker_local_$i.log 2>&1 &
    else
      ssh "$H" "cd ~/projects/connectome-control && mkdir -p logs && nohup caffeinate -is ./bin/worker2.sh macmini > logs/worker_$i.log 2>&1 & disown" 
    fi
  done
  echo "$H: $N workers"
}
mkdir -p logs
P_LOCAL=$(( $(sysctl -n hw.perflevel0.physicalcpu) - 1 ))
start_workers local $P_LOCAL
for H in "${HOSTS[@]}"; do
  P=$(ssh "$H" sysctl -n hw.perflevel0.physicalcpu)
  case "$H" in mba) P=$(( (P+1)/2 ));; esac
  start_workers "$H" "$P"
done
echo "launched; status: ~/cc-queue/status.txt"
