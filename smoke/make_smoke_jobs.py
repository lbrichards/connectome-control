"""Create the smoke-test job set in the coordinator queue.

Six tiny end-to-end jobs (demo collection -> BC -> delayed eval -> manifest):
worm/shuffle0/dense78 x 2 seeds, at reduced budget so each runs in a couple
of minutes. These validate the pipeline, not the science.
"""

import json
import os

Q = os.path.expanduser("~/cc-queue")
for d in ("pending", "claimed", "done", "results"):
    os.makedirs(f"{Q}/{d}", exist_ok=True)

SMOKE = dict(steps=300, n_demos=24, seq=300, eval_ep=20, eval_steps=500)
i = 0
for kind in ("worm", "shuffle0", "dense78"):
    for tseed in (0, 1):
        job = {"id": f"smoke_{kind}_t{tseed}", "kind": kind, "tseed": tseed,
               **SMOKE}
        with open(f"{Q}/pending/{i:03d}_{job['id']}.json", "w") as f:
            json.dump(job, f)
        i += 1
print(f"{i} smoke jobs -> {Q}/pending")
