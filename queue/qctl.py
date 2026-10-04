#!/usr/bin/env python3
"""Queue control (runs ONLY on the coordinator; SQLite = atomic claims).

Commands:
  init                      create db
  load <jobs.jsonl>         insert jobs (idempotent by id; skips DONE)
  claim <host>              atomically claim best pending job -> prints JSON
  heartbeat <id>            renew lease
  done <id>                 validate synced result, mark DONE
  fail <id> <msg>           record failure (retry up to 2x, then FAILED)
  status                    print one-line counts
  halt <reason> | unhalt    stop/resume claims
DB: ~/cc-queue/queue.db  Results: ~/cc-queue/results/
"""

import json
import os
import sqlite3
import sys
import time

Q = os.path.expanduser("~/cc-queue")
DB = f"{Q}/queue.db"
LEASE_S = 180          # renewed by 60 s heartbeats
MAX_ATTEMPTS = 3


def cx():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    os.makedirs(Q, exist_ok=True)
    c = cx()
    c.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, tier INT, cost INT, config TEXT,
        state TEXT DEFAULT 'pending', host TEXT, host_req TEXT,
        lease_until REAL, attempts INT DEFAULT 0, error TEXT,
        claimed_at REAL, done_at REAL)""")
    c.commit()
    print("db ready")


def load(path):
    c = cx()
    n = 0
    for line in open(path):
        j = json.loads(line)
        cur = c.execute("SELECT state FROM jobs WHERE id=?", (j["id"],)).fetchone()
        if cur and cur[0] == "DONE":
            continue
        c.execute("""INSERT INTO jobs(id,tier,cost,config,host_req)
                     VALUES(?,?,?,?,?)
                     ON CONFLICT(id) DO UPDATE SET
                       tier=excluded.tier, cost=excluded.cost,
                       config=excluded.config, host_req=excluded.host_req""",
                  (j["id"], j["tier"], j.get("cost", 1), json.dumps(j),
                   j.get("host_req")))
        n += 1
    c.commit()
    print(f"loaded {n}")


def claim(host):
    if os.path.exists(f"{Q}/HALT"):
        sys.exit(3)
    c = cx()
    c.execute("BEGIN IMMEDIATE")
    row = c.execute("""SELECT id, config FROM jobs
        WHERE state='pending' AND attempts < ?
          AND (host_req IS NULL OR host_req=?)
        ORDER BY tier ASC, cost DESC, id ASC LIMIT 1""",
        (MAX_ATTEMPTS, host)).fetchone()
    if not row:
        c.execute("COMMIT")
        sys.exit(1)
    jid, cfg = row
    c.execute("""UPDATE jobs SET state='running', host=?,
                 lease_until=?, attempts=attempts+1, claimed_at=?
                 WHERE id=?""",
              (host, time.time() + LEASE_S, time.time(), jid))
    c.execute("COMMIT")
    print(cfg)


def heartbeat(jid):
    c = cx()
    c.execute("UPDATE jobs SET lease_until=? WHERE id=? AND state='running'",
              (time.time() + LEASE_S, jid))
    c.commit()


def done(jid):
    # validate the synced result before marking DONE
    p = f"{Q}/results/{jid}.json"
    try:
        r = json.load(open(p))
        man = r["manifest"]
        assert r["metrics"] is not None
        for k in ("protocol", "git", "machine", "weights_sha256"):
            assert man[k]
    except Exception as e:
        fail(jid, f"result validation failed: {e}")
        return
    c = cx()
    c.execute("UPDATE jobs SET state='DONE', done_at=? WHERE id=?",
              (time.time(), jid))
    c.commit()
    print("DONE")


def fail(jid, msg):
    c = cx()
    att = c.execute("SELECT attempts FROM jobs WHERE id=?", (jid,)).fetchone()
    state = "FAILED" if att and att[0] >= MAX_ATTEMPTS else "pending"
    c.execute("UPDATE jobs SET state=?, error=?, host=NULL WHERE id=?",
              (state, msg[:500], jid))
    c.commit()
    print(state)


def status():
    c = cx()
    rows = c.execute("""SELECT tier, state, COUNT(*) FROM jobs
                        GROUP BY tier, state ORDER BY tier""").fetchall()
    print(json.dumps(rows))


def halt(reason):
    open(f"{Q}/HALT", "w").write(f"{time.ctime()}: {reason}\n")
    print("halted")


def unhalt():
    try:
        os.remove(f"{Q}/HALT")
    except FileNotFoundError:
        pass
    print("resumed")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"init": init,
     "load": lambda: load(sys.argv[2]),
     "claim": lambda: claim(sys.argv[2]),
     "heartbeat": lambda: heartbeat(sys.argv[2]),
     "done": lambda: done(sys.argv[2]),
     "fail": lambda: fail(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""),
     "status": status,
     "halt": lambda: halt(sys.argv[2] if len(sys.argv) > 2 else "manual"),
     "unhalt": unhalt}[cmd]() if cmd != "init" else init()
