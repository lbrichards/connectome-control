"""Demo-model selection (usage: select_demo_model.py [worm|dense78]) (rule of record, demo-side only — the research
training protocol is untouched by demo considerations).

Demo-grade criteria, evaluated from the demo's start distribution
(= the training start distribution):
  1. median time-to-first-sustained-catch <= 5 s
  2. quiet-hold >= 80%
  3. conversion on catchable approaches >= 50%
Selection (rule of record since 2026-10-06): the BEST-by-held seed among
demo-grade passers, for EVERY arm, labelled "best of 20 seeds" — a
showcase-vs-showcase comparison across tabs.
Writes web/demo_selection.json consumed by the exporter.
"""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/src")
from connectome_control.protocol import rollout, hang_starts, metrics
from connectome_control.jobs import _build_net

ARM = sys.argv[1] if len(sys.argv) > 1 else "worm"
rows = []
for f in sorted(glob.glob(os.path.expanduser(f"~/cc-queue/results/b2t1_{ARM}_*.json"))):
    r = json.load(open(f)); seed = r["job"]["tseed"]
    ck = torch.load(f.replace(".json", ".pt"), weights_only=False)
    net = _build_net(ARM, seed); net.load_state_dict(ck["state"])
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
passers = [x for x in rows if ok(x)]
pick = max(passers, key=lambda x: x["held"])
how = "best of 20 seeds"
NAME = {"worm": "v4 C. elegans", "dense78": "v4 dense-78"}[ARM]
label = (f"{NAME}, seed {pick['seed']}, {how} "
         f"({pick['held']*100:.0f}% held, {pick['ttc']:.1f}s to catch, "
         f"{pick['convc']*100:.0f}% conversion)")
json.dump({"pt": pick["pt"], "seed": pick["seed"], "label": label,
           "criteria": "ttc<=5s & quiet>=80% & convC>=50%",
           "metrics": {k: pick[k] for k in ("held", "quiet", "convc", "ttc")}},
          open(f"web/demo_selection{'' if ARM=='worm' else '_dense'}.json", "w"), indent=1)
print("SELECTED:", label)
