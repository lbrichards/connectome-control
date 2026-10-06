#!/bin/zsh
# Fleet keeper: every 5 min, while undone jobs exist, ensure each host runs
# its expected worker count; restart the shortfall and append an ALERT to
# ~/cc-queue/ALERTS (the scheduler folds the tail into status.txt).
# usage: keeper.sh   (run detached on the coordinator)
set -u
cd "$(dirname "$0")/.."
Q=~/cc-queue
typeset -A EXPECT
EXPECT=(local 3 mbp 8 mba 2 imac 4)

count_workers() {  # host
  local H=$1
  if [ "$H" = "local" ]; then
    ps ax -o command= | grep "worker2.sh" | grep -v -e caffeinate -e grep | wc -l | tr -d ' '
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$H" \
      'ps ax -o command= | grep "worker2.sh" | grep -v -e caffeinate -e grep | wc -l' \
      2>/dev/null | tr -d ' ' || echo "?"
  fi
}

start_workers() {  # host n
  local H=$1 N=$2
  [ "$N" -le 0 ] && return
  if [ "$H" = "local" ]; then
    for i in $(seq 1 $N); do
      nohup caffeinate -is ./bin/worker2.sh macmini \
        > logs/workerK_$(date +%s)_$i.log 2>&1 &
    done
  else
    ssh "$H" "cd ~/projects/connectome-control && mkdir -p logs && \
      for i in \$(seq 1 $N); do nohup caffeinate -is ./bin/worker2.sh macmini \
      > logs/workerK_\$(date +%s)_\$i.log 2>&1 & disown; done"
  fi
}

while true; do
  LEFT=$(sqlite3 $Q/queue.db \
    "SELECT COUNT(*) FROM jobs WHERE state IN ('pending','running')" \
    2>/dev/null || echo 0)
  if [ "$LEFT" = "0" ]; then sleep 300; continue; fi
  for H in ${(k)EXPECT}; do
    N=$(count_workers "$H")
    E=${EXPECT[$H]}
    if [ "$N" != "?" ] && [ "$N" -lt "$E" ]; then
      start_workers "$H" $((E - N))
      echo "$(date '+%F %H:%M') ALERT $H: workers $N/$E, restarted $((E-N))" \
        >> $Q/ALERTS
    elif [ "$N" = "?" ]; then
      echo "$(date '+%F %H:%M') ALERT $H: UNREACHABLE" >> $Q/ALERTS
    fi
  done
  sleep 300
done
