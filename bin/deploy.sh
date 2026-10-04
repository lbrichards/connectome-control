#!/bin/zsh
# Deploy this repo to a worker host: rsync source at current commit, build
# venv from the lockfile, install the .pth. usage: deploy.sh <host>
set -eu
H=$1
REPO=$(cd "$(dirname "$0")/.." && pwd)
ssh "$H" "mkdir -p ~/projects/connectome-control"
rsync -a --delete --exclude .venv --exclude results --exclude __pycache__ \
      "$REPO/" "$H:projects/connectome-control/"
ssh "$H" 'cd ~/projects/connectome-control
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv venv --python 3.12 .venv >/dev/null 2>&1 || true
  uv pip sync --python .venv/bin/python requirements.lock
  echo "$HOME/projects/connectome-control/src" > .venv/lib/python3.12/site-packages/connectome_control.pth
  git init -q 2>/dev/null; git add -A 2>/dev/null
  echo "deployed: $(hostname -s)"'
