#!/bin/zsh
# Runs ON the coordinator (macmini). Atomically claim one pending job for
# the named worker host. Prints the job JSON to stdout, or exits 1 if the
# queue is empty. Atomicity: mv within one filesystem; losers of a race
# get a failed mv and retry on the next loop.
set -u
Q=${QUEUE_DIR:-$HOME/cc-queue}
HOST=$1
cd "$Q" || exit 2
for f in $(ls pending 2>/dev/null); do
  if mv "pending/$f" "claimed/$HOST.$f" 2>/dev/null; then
    cat "claimed/$HOST.$f"
    echo "$f" >&2
    exit 0
  fi
done
exit 1
