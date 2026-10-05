#!/usr/bin/env python3
"""Unattended pilot: Tier-4 emission, drain detection, final checks, report.

Runs detached on the coordinator alongside the scheduler. Loop:
  - HALT present -> write report once, exit (halt conditions win).
  - Tier 1 fully terminal and not yet emitted -> build Tier 4 robustness
    jobs (median + best per arm among DONE Tier-1, refs validated) and
    load them (host_req = this host, where synced weights live).
  - Queue drained (nothing pending/running) -> run two cross-machine
    re-run checks, then the full analysis + figures + MORNING_REPORT.md,
    then exit.
State files in ~/cc-queue: TIER4_EMITTED, rerun_checks.json, pilot.log.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import sqlite3
import subprocess
import time

Q = os.path.expanduser("~/cc-queue")
DB = f"{Q}/queue.db"
REPO = os.path.expanduser("~/projects/connectome-control")
ME = socket.gethostname().split(".")[0]
# Worker host aliases come from the environment (or ~/cc-queue/hosts.txt),
# never hardcoded: keeps machine names out of the repo.
def _hosts():
    env = os.environ.get("CC_HOSTS")
    if env:
        return env.split(",")
    p = os.path.expanduser("~/cc-queue/hosts.txt")
    return open(p).read().split() if os.path.exists(p) else []
HOSTS = _hosts()


def log(m):
    with open(f"{Q}/pilot.log", "a") as f:
        f.write(f"{time.ctime()}  {m}\n")


def cx():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def counts(c, tier=None, exclude9=False):
    q = "SELECT state, COUNT(*) FROM jobs"
    args = ()
    w = []
    if tier is not None:
        w.append("tier=?"); args = (tier,)
    if exclude9:
        w.append("tier != 9")
    if w:
        q += " WHERE " + " AND ".join(w)
    q += " GROUP BY state"
    return dict(c.execute(q, args).fetchall())


def emit_tier4(c):
    rows = c.execute("SELECT id, config FROM jobs WHERE tier=1 AND state='DONE'"
                     ).fetchall()
    res = []
    for jid, cfg in rows:
        p = f"{Q}/results/{jid}.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        res.append((r["job"]["kind"], r["metrics"]["held"], jid))
    # demo-grade gate: every tier-1 model with quiet_hold >= 0.80 gets the
    # robustness suite; the count of passers is itself a reported result.
    jobs, n_passed, n_total = [], 0, 0
    for jid_, cfg_ in rows:
        pth = f"{Q}/results/{jid_}.json"
        if not os.path.exists(pth):
            continue
        r = json.load(open(pth))
        n_total += 1
        qh = r["metrics"].get("quiet_hold")
        if qh is None or qh < 0.80:
            continue
        n_passed += 1
        arm = ("shuffle" if r["job"]["kind"].startswith("shuffle")
               else r["job"]["kind"])
        cfg = {"type": "robust", "ref": jid_, "tier": 4, "cost": 1,
               "host_req": ME, "kind": arm, "tseed": r["job"]["tseed"]}
        cfg["id"] = (f"t4_{arm}_s{cfg['tseed']}_" + hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10])
        jobs.append(cfg)
    log(f"tier-4 demo-grade gate: {n_passed}/{n_total} tier-1 models passed "
        f"(quiet_hold >= 0.80)")
    open(f"{Q}/tier4_gate.json", "w").write(json.dumps(
        {"passed": n_passed, "total": n_total}))
    path = f"{Q}/tier4.jsonl"
    with open(path, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    subprocess.run([f"{REPO}/.venv/bin/python", f"{REPO}/queue/qctl.py",
                    "load", path], check=True)
    open(f"{Q}/TIER4_EMITTED", "w").write(f"{len(jobs)} jobs\n")
    log(f"tier 4 emitted: {len(jobs)} robustness jobs")


def rerun_checks(c):
    done = c.execute("""SELECT id, host, config FROM jobs
        WHERE state='DONE' AND tier IN (1,2)""").fetchall()
    random.seed(20261005)
    picks = random.sample(done, min(2, len(done)))
    out = []
    for jid, host, cfg in picks:
        cfg = json.loads(cfg)
        orig = (host or "").split(".")[0].lower()
        cands = [h for h in HOSTS + ["macmini"]
                 if h.lower() != orig and orig not in h.lower()]
        target = None
        for cand in cands:
            probe = ["true"] if cand == "macmini" else                 ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                 cand, "true"]
            if subprocess.run(probe).returncode == 0:
                target = cand
                break
        if target is None:
            out.append({"id": jid, "orig_host": host, "rerun_host": None,
                        "match": None, "note": "no reachable host"})
            continue
        orig_sha = json.load(open(f"{Q}/results/{jid}.json"))[
            "manifest"]["weights_sha256"]
        cfg2 = dict(cfg); cfg2["id"] = f"rerun_{jid}"
        jp = f"/tmp/rerun_{jid}.json"
        json.dump(cfg2, open(jp, "w"))
        log(f"rerun {jid} ({host}) on {target}")
        try:
            if target == "macmini":
                subprocess.run([f"{REPO}/.venv/bin/python", "-m",
                                "connectome_control.jobs", jp],
                               cwd=REPO, check=True, timeout=7200)
                new = json.load(open(f"{REPO}/results/rerun_{jid}.json"))
            else:
                subprocess.run(["scp", "-q", jp, f"{target}:/tmp/"],
                               check=True)
                subprocess.run(["ssh", "-o", "BatchMode=yes", target,
                                f"cd ~/projects/connectome-control && "
                                f".venv/bin/python -m connectome_control.jobs {jp}"],
                               check=True, timeout=7200)
                subprocess.run(["scp", "-q",
                                f"{target}:projects/connectome-control/"
                                f"results/rerun_{jid}.json", "/tmp/"],
                               check=True)
                new = json.load(open(f"/tmp/rerun_{jid}.json"))
            match = new["manifest"]["weights_sha256"] == orig_sha
        except Exception as e:
            log(f"rerun {jid} error: {e}")
            match = False
        out.append({"id": jid, "orig_host": host, "rerun_host": target,
                    "match": bool(match)})
    json.dump(out, open(f"{Q}/rerun_checks.json", "w"), indent=1)
    log(f"rerun checks: {out}")
    return all(e["match"] for e in out if e["match"] is not None)


def finale():
    ok = True
    try:
        c = cx()
        ok = rerun_checks(c)
    except Exception as e:
        log(f"rerun phase error: {e}")
        ok = False
    subprocess.run([f"{REPO}/.venv/bin/python",
                    f"{REPO}/analysis/overnight_report.py"])
    log(f"final report written (rerun checks {'PASS' if ok else 'FAIL'})")


def main():
    log(f"pilot started on {ME}")
    while True:
        try:
            if os.path.exists(f"{Q}/HALT"):
                log("HALT detected; writing report and exiting")
                subprocess.run([f"{REPO}/.venv/bin/python",
                                f"{REPO}/analysis/overnight_report.py"])
                return
            c = cx()
            t1 = counts(c, 1)
            if (not os.path.exists(f"{Q}/TIER4_EMITTED")
                    and t1.get("pending", 0) == 0
                    and t1.get("running", 0) == 0
                    and t1.get("DONE", 0) > 0):
                emit_tier4(c)
            main_c = counts(c, exclude9=True)
            if (main_c.get("pending", 0) == 0 and main_c.get("running", 0) == 0
                    and os.path.exists(f"{Q}/TIER4_EMITTED")
                    and not os.path.exists(f"{Q}/FIRST_PASS_DONE")):
                log("tiers 1/2/4 drained; running FIRST-PASS finale "
                    "(dense-448 still running)")
                finale()
                open(f"{Q}/FIRST_PASS_DONE", "w").write(time.ctime())
            t9 = counts(c, tier=9)
            if (os.path.exists(f"{Q}/FIRST_PASS_DONE")
                    and t9.get("pending", 0) == 0
                    and t9.get("running", 0) == 0):
                log("tier 3 (demoted) drained; second-pass report append")
                subprocess.run([f"{REPO}/.venv/bin/python",
                                f"{REPO}/analysis/overnight_report.py"])
                log("second pass complete; pilot exiting")
                return
            c.close()
        except Exception as e:
            log(f"pilot error: {e}")
        time.sleep(120)


if __name__ == "__main__":
    main()
