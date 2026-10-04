#!/bin/zsh
# Full bring-up when worker hosts come online: deploy -> preflight ->
# cross-host smoke -> determinism check -> dry run -> launch overnight.
# usage: bringup.sh mbp mba
set -eu
cd "$(dirname "$0")/.."
HOSTS=("$@")
for H in "${HOSTS[@]}"; do ./bin/deploy.sh "$H"; done
./bin/preflight.sh "${HOSTS[@]}"

echo "== cross-host smoke (one tiny job per host, via jobs runner) =="
cat > /tmp/smokejob.json <<'EOS'
{"id":"xsmoke","type":"bc","kind":"worm","tseed":0,"demo_seed":900,
 "steps":200,"dagger":0,"eval_ep":10,"eval_steps":300,"tier":0,"cost":1}
EOS
rm -f /tmp/xsha.*
.venv/bin/python -m connectome_control.jobs /tmp/smokejob.json
python3 -c "import json;print(json.load(open('results/xsmoke.json'))['manifest']['weights_sha256'])" > /tmp/xsha.local
for H in "${HOSTS[@]}"; do
  scp -q /tmp/smokejob.json "$H:/tmp/"
  ssh "$H" 'cd ~/projects/connectome-control && .venv/bin/python -m connectome_control.jobs /tmp/smokejob.json && python3 -c "import json;print(json.load(open(\"results/xsmoke.json\"))[\"manifest\"][\"weights_sha256\"])"' > /tmp/xsha.$H
done
echo "== determinism check (weights sha must match across hosts) =="
OK=1
REF=$(cat /tmp/xsha.local); echo "local: $REF"
for H in "${HOSTS[@]}"; do
  V=$(tail -1 /tmp/xsha.$H); echo "$H:    $V"
  [ "$V" = "$REF" ] || OK=0
done
[ $OK -eq 1 ] && echo "DETERMINISM: EXACT MATCH (tolerance: bitwise)" \
  || { echo "DETERMINISM FAILED"; exit 1; }

echo "== dry run: one real tier-1-sized job per host =="
i=0
for H in local "${HOSTS[@]}"; do
  cat > /tmp/dry_$i.json <<EOS
{"id":"dry_$H","type":"bc","kind":"worm","tseed":9$i,"demo_seed":900,
 "steps":3000,"dagger":1,"dagger_ep":32,"dagger_steps":500,
 "eval_ep":30,"eval_steps":1000,"tier":0,"cost":1}
EOS
  if [ "$H" = "local" ]; then
    .venv/bin/python -m connectome_control.jobs /tmp/dry_$i.json &
  else
    scp -q /tmp/dry_$i.json "$H:/tmp/"
    ssh "$H" "cd ~/projects/connectome-control && .venv/bin/python -m connectome_control.jobs /tmp/dry_$i.json && scp -q results/dry_$H.json macmini:cc-queue/results/" &
  fi
  i=$((i+1))
done
wait
echo "dry runs complete"
./bin/launch_overnight.sh "${HOSTS[@]}"
