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
HOSTS = ["mbp", "mba"]          # rerun candidates besides coordinator


def log(m):
    with open(f"{Q}/pilot.log", "a") as f:
        f.write(f"{time.ctime()}  {m}\n")


def cx():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def counts(c, tier=None):
    q = "SELECT state, COUNT(*) FROM jobs"
    args = ()
    if tier is not None:
        q += " WHERE tier=?"; args = (tier,)
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
    arms = {}
    for kind, held, jid in res:
        arm = "shuffle" if kind.startswith("shuffle") else kind
        arms.setdefault(arm, []).append((held, jid))
    jobs = []
    for arm, lst in arms.items():
        lst.sort()
        for tag, pick in (("median", lst[len(lst)//2]), ("best", lst[-1])):
            ref = pick[1]
            cfg = {"type": "robust", "ref": ref, "tier": 4, "cost": 1,
                   "host_req": ME, "kind": arm, "tseed": 0}
            cfg["id"] = (f"t4_{arm}_{tag}_" + hashlib.sha256(
                json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10])
            jobs.append(cfg)
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
        orig = (host or "").split(".")[0].replace("Larrys-MacBook-Pro","mbp").replace("mini","macmini")
        cands = [h for h in HOSTS + ["macmini"] if h != orig]
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
            allc = counts(c)
            if allc.get("pending", 0) == 0 and allc.get("running", 0) == 0 \
                    and os.path.exists(f"{Q}/TIER4_EMITTED"):
                log("queue drained; running finale")
                finale()
                return
            c.close()
        except Exception as e:
            log(f"pilot error: {e}")
        time.sleep(120)


if __name__ == "__main__":
    main()
