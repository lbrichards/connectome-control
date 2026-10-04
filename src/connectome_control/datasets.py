"""Teacher demonstration datasets: generated ONCE on the coordinator,
checksummed, rsynced to workers. Workers never regenerate them.

One file per demo seed: data/demos/demos_s<seed>.npz  (O, A float32).
SHA256SUMS sits beside them and is verified on every host.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

from .jobs import collect_demos

DEMO_DIR = os.path.join(os.path.dirname(__file__), "data", "demos")


def path_for(seed):
    return f"{DEMO_DIR}/demos_s{seed}.npz"


def generate(seeds, n_ep=320, seq=350):
    os.makedirs(DEMO_DIR, exist_ok=True)
    sums = []
    for sd in seeds:
        p = path_for(sd)
        if not os.path.exists(p):
            O, A = collect_demos(n_ep, seq, sd)
            np.savez_compressed(p, O=O, A=A)
            print(f"  demos_s{sd}: {len(O)} eps", flush=True)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        sums.append(f"{h}  demos_s{sd}.npz")
    open(f"{DEMO_DIR}/SHA256SUMS", "w").write("\n".join(sums) + "\n")
    print(f"{len(seeds)} datasets + SHA256SUMS")


def verify():
    ok = True
    for line in open(f"{DEMO_DIR}/SHA256SUMS"):
        h, name = line.split()
        p = f"{DEMO_DIR}/{name}"
        if not os.path.exists(p):
            print(f"MISSING {name}"); ok = False; continue
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != h:
            print(f"CHECKSUM MISMATCH {name}"); ok = False
    print("datasets OK" if ok else "DATASET VERIFY FAILED")
    return ok


def load(seed):
    d = np.load(path_for(seed))
    return d["O"], d["A"]


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate([int(x) for x in sys.argv[2:]])
    elif sys.argv[1] == "verify":
        sys.exit(0 if verify() else 1)
