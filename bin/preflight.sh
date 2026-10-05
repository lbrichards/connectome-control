#!/bin/zsh
# Pre-flight across all hosts: SSH reachable, same commit, same Python /
# torch / numpy versions, CPU backend, single-thread settings honoured.
# usage: preflight.sh host1 host2 ...   (run from the repo on macmini)
set -u
REPO_NAME=connectome-control
LOCAL_COMMIT=$(git rev-parse HEAD)
echo "coordinator commit: $LOCAL_COMMIT"
FAIL=0
for H in "$@"; do
  echo "--- $H ---"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$H" true 2>/dev/null; then
    echo "  SSH: UNREACHABLE"; FAIL=1; continue
  fi
  RC=$(ssh "$H" "cd ~/projects/$REPO_NAME 2>/dev/null && { git rev-parse HEAD 2>/dev/null || cat COMMIT_STAMP; }" 2>/dev/null)
  if [ "$RC" = "$LOCAL_COMMIT" ]; then echo "  commit: match"
  else echo "  commit: MISMATCH ($RC)"; FAIL=1; fi
  ssh "$H" "cd ~/projects/$REPO_NAME && .venv/bin/python - <<'EOF'
import sys, torch, numpy, platform
print(f'  python {sys.version.split()[0]}  torch {torch.__version__}  numpy {numpy.__version__}')
print(f'  backend: cpu (mps available={torch.backends.mps.is_available()}, NOT used)')
print(f'  arch: {platform.machine()}')
EOF" || FAIL=1
done
[ $FAIL -eq 0 ] && echo "PREFLIGHT PASS" || echo "PREFLIGHT FAIL"
exit $FAIL
