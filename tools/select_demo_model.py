"""Demo-model selection (rule of record, demo-side only — the research
training protocol is untouched by demo considerations).

Demo-grade criteria, evaluated from the demo's start distribution
(= the training start distribution):
  1. median time-to-first-sustained-catch <= 5 s
  2. quiet-hold >= 80%
  3. conversion on catchable approaches >= 50%
Selection: the MEDIAN-by-held seed if it passes all three; otherwise the
BEST-by-held seed among passers, labelled "best of 20 seeds".
Writes web/demo_selection.json consumed by the exporter.
"""
import glob, json, sys
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")
from connectome_control.protocol import rollout, hang_starts, metrics
from connectome_control.jobs import _build_net

rows = []
for f in sorted(glob.glob("/Users/macmini/cc-queue/results/b2t1_worm_*.json")):
    r = json.load(open(f)); seed = r["job"]["tseed"]
    ck = torch.load(f.replace(".json", ".pt"), weights_only=False)
    net = _build_net("worm", seed); net.load_state_dict(ck["state"])
    TH, alive, exit_t = rollout(net, ck["norm"],
                                hang_starts(50, np.random.default_rng(42)), 1500)
    m = metrics(TH, alive, exit_t)
    rows.append(dict(seed=seed, pt=f.replace(".json", ".pt"),
                     held=r["metrics"]["held"],
                     quiet=r["metrics"]["quiet_hold"],
                     convc=r["metrics"]["conversion_catchable"] or 0,
                     ttc=m["time_to_catch_median_s"]))
rows.sort(key=lambda x: x["held"])
ok = lambda x: (x["ttc"] is not None and x["ttc"] <= 5.0
                and x["quiet"] >= 0.80 and x["convc"] >= 0.50)
med = rows[len(rows) // 2]
if ok(med):
    pick, how = med, "median of 20 seeds"
else:
    passers = [x for x in rows if ok(x)]
    pick = max(passers, key=lambda x: x["held"])
    how = "best of 20 seeds (median failed demo-grade)"
label = (f"v4 C. elegans, seed {pick['seed']}, {how} "
         f"({pick['held']*100:.0f}% held, {pick['ttc']:.1f}s to catch, "
         f"{pick['convc']*100:.0f}% conversion)")
json.dump({"pt": pick["pt"], "seed": pick["seed"], "label": label,
           "criteria": "ttc<=5s & quiet>=80% & convC>=50%",
           "metrics": {k: pick[k] for k in ("held", "quiet", "convc", "ttc")}},
          open("web/demo_selection.json", "w"), indent=1)
print("SELECTED:", label)
