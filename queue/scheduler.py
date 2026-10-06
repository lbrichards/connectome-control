#!/usr/bin/env python3
"""Long-lived scheduler daemon (coordinator). Run detached under caffeinate.

Every SWEEP_S: expire stale leases back to pending (attempts preserved;
MAX_ATTEMPTS enforced at claim/fail). Every STATUS_S: write status.json /
status.txt with per-tier counts, per-host throughput, crude ETA. Halt
conditions: >10% of completed jobs FAILED, or same error on 2+ hosts.
"""

import collections
import json
import os
import sqlite3
import time

Q = os.path.expanduser("~/cc-queue")
DB = f"{Q}/queue.db"
SWEEP_S, STATUS_S = 30, 300


def cx():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def sweep(c):
    n = c.execute("""UPDATE jobs SET state='pending', host=NULL
                     WHERE state='running' AND lease_until < ?""",
                  (time.time(),)).rowcount
    c.commit()
    if n:
        log(f"lease sweep: returned {n} job(s) to pending")


def halt_check(c):
    done = c.execute("SELECT COUNT(*) FROM jobs WHERE state='DONE'").fetchone()[0]
    failed = c.execute("SELECT COUNT(*) FROM jobs WHERE state='FAILED'").fetchone()[0]
    if done + failed >= 10 and failed / max(1, done + failed) > 0.10:
        return f"failure rate {failed}/{done+failed} > 10%"
    errs = c.execute("""SELECT error, COUNT(DISTINCT host) FROM jobs
                        WHERE state='FAILED' AND error IS NOT NULL
                        GROUP BY substr(error,1,80)""").fetchall()
    for e, nh in errs:
        if nh >= 2:
            return f"same error on {nh} hosts: {e[:80]}"
    return None


def status(c):
    tiers = collections.defaultdict(lambda: collections.Counter())
    for t, s, n in c.execute(
            "SELECT tier,state,COUNT(*) FROM jobs GROUP BY tier,state"):
        tiers[t][s] = n
    hosts = {}
    hour_ago = time.time() - 3600
    for h, n, mn in c.execute("""SELECT host, COUNT(*),
            AVG((done_at-claimed_at)/60.0) FROM jobs
            WHERE state='DONE' AND done_at > ? GROUP BY host""", (hour_ago,)):
        hosts[h] = {"done_last_hour": n, "avg_min": round(mn or 0, 1)}
    pend = c.execute(
        "SELECT COUNT(*) FROM jobs WHERE state='pending'").fetchone()[0]
    run = c.execute(
        "SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0]
    rate = sum(h["done_last_hour"] for h in hosts.values())
    eta_h = round((pend + run) / rate, 1) if rate else None
    s = {"time": time.ctime(), "tiers": {k: dict(v) for k, v in tiers.items()},
         "hosts": hosts, "pending": pend, "running": run, "eta_hours": eta_h,
         "halted": os.path.exists(f"{Q}/HALT")}
    alerts = []
    if os.path.exists(f"{Q}/ALERTS"):
        alerts = open(f"{Q}/ALERTS").read().splitlines()[-5:]
    s["alerts"] = alerts
    json.dump(s, open(f"{Q}/status.json", "w"), indent=1)
    with open(f"{Q}/status.txt", "w") as f:
        f.write(json.dumps(s, indent=1))
        if alerts:
            f.write("\n\nRECENT ALERTS (keeper):\n" + "\n".join(alerts) + "\n")
    return s


def log(msg):
    with open(f"{Q}/scheduler.log", "a") as f:
        f.write(f"{time.ctime()}  {msg}\n")


def main():
    log("scheduler started")
    last_status = 0
    while True:
        try:
            c = cx()
            sweep(c)
            reason = halt_check(c)
            if reason and not os.path.exists(f"{Q}/HALT"):
                open(f"{Q}/HALT", "w").write(f"{time.ctime()}: AUTO: {reason}\n")
                log(f"AUTO-HALT: {reason}")
            if time.time() - last_status > STATUS_S:
                s = status(c)
                log(f"status: pending={s['pending']} running={s['running']} "
                    f"eta_h={s['eta_hours']}")
                last_status = time.time()
            c.close()
        except Exception as e:
            log(f"scheduler error: {e}")
        time.sleep(SWEEP_S)


if __name__ == "__main__":
    main()
