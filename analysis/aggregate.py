"""Collect result JSONs (coordinator ~/cc-queue/results) into one table."""
import glob, json, os, sys
rows = []
src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/cc-queue/results")
for f in sorted(glob.glob(f"{src}/*.json")):
    r = json.load(open(f))
    m, man = r["metrics"], r["manifest"]
    rows.append((r["job"]["id"], r["job"]["kind"], r["job"]["tseed"],
                 m["held"], m["off_track"], m.get("conversion_catchable"),
                 r["final_mse"], man["machine"], man["weights_sha256"][:12]))
print(f"{'id':<22}{'kind':<10}{'seed':>5}{'held':>7}{'off':>6}{'convC':>7}"
      f"{'mse':>8}  {'machine':<10}{'weights':<14}")
for r in rows:
    cc = f"{r[5]*100:.0f}%" if r[5] is not None else "-"
    print(f"{r[0]:<22}{r[1]:<10}{r[2]:>5}{r[3]*100:>6.0f}%{r[4]*100:>5.0f}%"
          f"{cc:>7}{r[6]:>8}  {r[7]:<10}{r[8]:<14}")
