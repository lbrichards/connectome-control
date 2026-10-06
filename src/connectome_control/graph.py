"""Load the C. elegans connectome as adjacency masks, plus topology controls.

Data: OpenWorm c302 `herm_full_edgelist.csv` (White et al. 1986 lineage).
Columns: Source, Target, Weight, Type  where Type is 'chemical' or 'electrical'
and Weight is a synapse count.

Chemical synapses are directed; electrical (gap) junctions are symmetric.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data" / "herm_full_edgelist.csv"
URL = "https://raw.githubusercontent.com/openworm/c302/master/c302/data/herm_full_edgelist.csv"

# Amphid chemosensory neurons -- the worm's main chemical/odour input.
SENSORY = [
    f"{c}{s}" for c in
    ("ASE", "ASH", "ASI", "ASJ", "ASK", "ASG", "ADF", "ADL", "AWA", "AWB", "AWC")
    for s in ("L", "R")
]

# Body wall muscles: 95 cells, named by dorsal/ventral and left/right.
MUSCLE_RE = re.compile(r"^[dv]BWM[LR]\d+$")


@dataclass
class Connectome:
    names: list[str]
    chem: np.ndarray   # (N, N) directed synapse counts, chem[i, j] = j -> i
    gap: np.ndarray    # (N, N) symmetric gap-junction counts
    sensory: np.ndarray
    muscle: np.ndarray

    @property
    def n(self) -> int:
        return len(self.names)

    def summary(self) -> str:
        return (
            f"{self.n} nodes | chemical edges {int((self.chem > 0).sum())} "
            f"| gap edges {int((self.gap > 0).sum())} "
            f"| sensory {len(self.sensory)} | muscle {len(self.muscle)} "
            f"| density {(self.chem > 0).mean():.4f}"
        )


def load(path: Path = DATA) -> Connectome:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; download from {URL}")

    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(
                (r["Source"].strip(), r["Target"].strip(),
                 float(r["Weight"]), r["Type"].strip().lower())
            )

    names = sorted({n for src, tgt, _, _ in rows for n in (src, tgt)})
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)

    chem = np.zeros((n, n), dtype=np.float32)
    gap = np.zeros((n, n), dtype=np.float32)

    for src, tgt, w, kind in rows:
        i, j = idx[tgt], idx[src]          # row = postsynaptic, col = presynaptic
        if kind.startswith("chem"):
            chem[i, j] += w
        else:
            gap[i, j] += w
            gap[j, i] += w                 # force symmetry regardless of listing

    sensory = np.array([idx[s] for s in SENSORY if s in idx], dtype=np.int64)
    muscle = np.array([i for i, nm in enumerate(names) if MUSCLE_RE.match(nm)],
                      dtype=np.int64)

    return Connectome(names, chem, gap, sensory, muscle)


def dorsal_ventral(c: Connectome) -> tuple[np.ndarray, np.ndarray]:
    """Split body wall muscles into dorsal and ventral groups."""
    d = np.array([i for i in c.muscle if c.names[i].startswith("d")], dtype=np.int64)
    v = np.array([i for i in c.muscle if c.names[i].startswith("v")], dtype=np.int64)
    return d, v


# ---------------------------------------------------------------- controls

def rewire_degree_preserving(adj: np.ndarray, rng: np.random.Generator,
                             symmetric: bool = False) -> np.ndarray:
    """Configuration-model shuffle: same in/out degree per node, edges reassigned.

    Weights ride along with the edges so the weight distribution is preserved too.
    """
    mask = adj > 0
    if symmetric:
        iu = np.triu_indices_from(adj, k=1)
        present = mask[iu]
        w = adj[iu][present]
        deg = mask.sum(1)
        out = np.zeros_like(adj)
        # Chung-Lu style resampling against the degree sequence.
        p = deg / max(deg.sum(), 1)
        n_edges = int(present.sum())
        placed = 0
        guard = 0
        while placed < n_edges and guard < 200 * n_edges:
            i, j = rng.choice(len(adj), size=2, p=p)
            guard += 1
            if i == j or out[i, j] > 0:
                continue
            out[i, j] = out[j, i] = w[placed]
            placed += 1
        return out

    rows, cols = np.nonzero(mask)
    w = adj[rows, cols]
    # Preserve out-degree exactly by permuting targets within the edge list.
    perm = rng.permutation(len(rows))
    out = np.zeros_like(adj)
    for k, p in enumerate(perm):
        i, j = rows[p], cols[k]
        if i == j:
            continue
        out[i, j] = max(out[i, j], w[p])
    return out


def erdos_renyi(adj: np.ndarray, rng: np.random.Generator,
                symmetric: bool = False) -> np.ndarray:
    """Random graph at matched edge count, weights resampled from the real ones."""
    n = len(adj)
    mask = adj > 0
    n_edges = int(mask.sum() // (2 if symmetric else 1))
    w = adj[mask]
    out = np.zeros_like(adj)
    placed = 0
    while placed < n_edges:
        i, j = rng.integers(0, n, size=2)
        if i == j or out[i, j] > 0:
            continue
        val = w[rng.integers(0, len(w))]
        out[i, j] = val
        if symmetric:
            out[j, i] = val
        placed += 1
    return out


if __name__ == "__main__":
    c = load()
    print(c.summary())
    d, v = dorsal_ventral(c)
    print(f"dorsal muscles {len(d)}, ventral muscles {len(v)}")
    print("sensory:", [c.names[i] for i in c.sensory][:8], "...")


# ===== typed shuffle (v4 control) =====



def node_classes(names):
    return np.array([nm[0].isupper() for nm in names])      # True = neuron


def _swap_rewire_directed(edges, rng, passes=10, attempts=None):
    """edges: list of [i, j, w] with i=post, j=pre. Swap targets i between
    same-class edge pairs. Preserves in/out degree exactly."""
    E = [list(e) for e in edges]
    n = len(E)
    existing = set((e[0], e[1]) for e in E)
    swaps = 0
    for _ in range(attempts if attempts is not None else passes * n):
        a, b = rng.integers(0, n, 2)
        if a == b: continue
        ia, ja, _ = E[a]; ib, jb, _ = E[b]
        if ia == ib or ja == jb: continue
        if ia == jb or ib == ja: continue                  # would self-loop
        if (ia, jb) in existing or (ib, ja) in existing: continue
        existing.discard((ia, ja)); existing.discard((ib, jb))
        E[a][1], E[b][1] = jb, ja
        existing.add((ia, jb)); existing.add((ib, ja))
        swaps += 1
    return E, swaps


def _swap_rewire_symmetric(pairs, rng, passes=10, attempts=None):
    """pairs: list of [i, j, w], i<j, symmetric edges. Swap partners within
    class. Preserves each node's gap degree exactly."""
    E = [list(e) for e in pairs]
    n = len(E)
    key = lambda i, j: (min(i, j), max(i, j))
    existing = set(key(e[0], e[1]) for e in E)
    swaps = 0
    for _ in range(attempts if attempts is not None else passes * n):
        a, b = rng.integers(0, n, 2)
        if a == b: continue
        ia, ja, _ = E[a]; ib, jb, _ = E[b]
        if len({ia, ja, ib, jb}) < 4: continue
        # swap: (ia-jb), (ib-ja)
        if key(ia, jb) in existing or key(ib, ja) in existing: continue
        existing.discard(key(ia, ja)); existing.discard(key(ib, jb))
        E[a][1], E[b][1] = jb, ja
        existing.add(key(ia, jb)); existing.add(key(ib, ja))
        swaps += 1
    return E, swaps


def typed_shuffle(chem, gap, is_neuron, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    n = len(is_neuron)

    # ---- chemical: classes by (pre class, post class); chem[i,j] = j -> i
    out_chem = np.zeros_like(chem)
    ii, jj = np.nonzero(chem)
    cls = (is_neuron[jj].astype(int) * 2 + is_neuron[ii].astype(int))
    for cl in np.unique(cls):
        sel = cls == cl
        edges = [[int(i), int(j), float(chem[i, j])]
                 for i, j in zip(ii[sel], jj[sel])]
        E, swaps = _swap_rewire_directed(edges, rng)
        for i, j, w in E:
            out_chem[i, j] = w
        if verbose:
            tag = {3: "N->N", 2: "N->E", 1: "E->N", 0: "E->E"}[int(cl)]
            print(f"  chem {tag}: {len(E)} edges, {swaps} swaps")

    # ---- gap: unordered classes {NN, NE, EE}; self-junctions (diagonal)
    # carried through unchanged -- they are node-local and have no partner
    out_gap = np.zeros_like(gap)
    np.fill_diagonal(out_gap, np.diag(gap))
    iu = np.triu_indices(n, k=1)
    mask = gap[iu] > 0
    pi, pj = iu[0][mask], iu[1][mask]
    gcls = is_neuron[pi].astype(int) + is_neuron[pj].astype(int)
    for cl in np.unique(gcls):
        sel = gcls == cl
        pairs = [[int(a), int(b), float(gap[a, b])]
                 for a, b in zip(pi[sel], pj[sel])]
        E, swaps = _swap_rewire_symmetric(pairs, rng)
        for a, b, w in E:
            out_gap[a, b] = w; out_gap[b, a] = w
        if verbose:
            tag = {2: "NN", 1: "NE", 0: "EE"}[int(cl)]
            print(f"  gap {tag}: {len(E)} edges, {swaps} swaps")

    return out_chem, out_gap


def verify(chem, gap, chem2, gap2, is_neuron):
    ok = True
    for name, a, b in (("chem", chem, chem2), ("gap", gap, gap2)):
        if int((a > 0).sum()) != int((b > 0).sum()):
            print(f"  !! {name} edge count changed"); ok = False
        for axis, deg in (("in", 1), ("out", 0)):
            if not np.array_equal((a > 0).sum(axis=deg), (b > 0).sum(axis=deg)):
                print(f"  !! {name} {axis}-degree not preserved"); ok = False
    # class composition
    ii, jj = np.nonzero(chem); ii2, jj2 = np.nonzero(chem2)
    c1 = np.bincount(is_neuron[jj].astype(int)*2 + is_neuron[ii].astype(int), minlength=4)
    c2 = np.bincount(is_neuron[jj2].astype(int)*2 + is_neuron[ii2].astype(int), minlength=4)
    if not np.array_equal(c1, c2):
        print("  !! chem per-class counts changed"); ok = False
    frac_moved = 1 - np.mean((chem > 0) == (chem2 > 0))
    print(f"  degree sequences preserved: {ok} | adjacency entries changed: "
          f"{frac_moved*100:.1f}% of matrix")
    return ok




# ===== partial typed rewiring (Task A) =====

def partial_rewire(chem, gap, is_neuron, target_f, seed, verbose=False,
                   restrict=None, input_idx=None, target_edges=None):
    """Rewire a FRACTION of worm edges by the same typed, degree-preserving
    double-edge swaps as typed_shuffle, stopping once the realized fraction
    of ORIGINAL edges no longer present reaches target_f. Swap attempts are
    spread round-robin across classes in small increments (2% of class size)
    so the stop is precise; swaps may re-create destroyed originals, so f is
    always measured against the original edge sets, never counted.
    Returns (chem2, gap2, realized_f). Deterministic in (target_f, seed).

    restrict (targeted-rewiring extension): 'input_out' confines swaps to
    chemical edges whose PRESYNAPTIC node is in input_idx (the injected
    sensory set); 'not_input_out' confines swaps to all other chemical
    edges. Gap junctions are untouched under either restriction so the two
    variants differ only in WHICH chem edges move. target_edges, if given,
    stops at an absolute count of original edges removed instead of a
    fraction (both measured against the full original edge set)."""
    rng = np.random.default_rng(seed)
    n = len(is_neuron)
    in_set = set(map(int, input_idx)) if input_idx is not None else set()

    ii, jj = np.nonzero(chem)
    cls = (is_neuron[jj].astype(int) * 2 + is_neuron[ii].astype(int))
    chem_classes, frozen_chem = [], []
    for cl in np.unique(cls):
        sel = cls == cl
        E = [[int(i), int(j), float(chem[i, j])]
             for i, j in zip(ii[sel], jj[sel])]
        if restrict == "input_out":
            chem_classes.append([e for e in E if e[1] in in_set])
            frozen_chem.append([e for e in E if e[1] not in in_set])
        elif restrict == "not_input_out":
            chem_classes.append([e for e in E if e[1] not in in_set])
            frozen_chem.append([e for e in E if e[1] in in_set])
        else:
            chem_classes.append(E)
    iu = np.triu_indices(n, k=1)
    mask = gap[iu] > 0
    pi, pj = iu[0][mask], iu[1][mask]
    gcls = is_neuron[pi].astype(int) + is_neuron[pj].astype(int)
    gap_classes = []
    for cl in np.unique(gcls):
        sel = gcls == cl
        gap_classes.append([[int(a), int(b), float(gap[a, b])]
                            for a, b in zip(pi[sel], pj[sel])])

    if restrict is not None:
        gap_frozen = True
        orig_c = set((i, j) for E in chem_classes + frozen_chem
                     for i, j, _ in E)
    else:
        gap_frozen = False
        orig_c = set((i, j) for E in chem_classes for i, j, _ in E)
    orig_g = set((min(a, b), max(a, b))
                 for E in gap_classes for a, b, _ in E)
    total = len(orig_c) + len(orig_g)

    def removed():
        kept = sum((i, j) in orig_c for E in chem_classes for i, j, _ in E)
        kept += sum(len(E) for E in frozen_chem) if restrict else 0
        kept += sum((min(a, b), max(a, b)) in orig_g
                    for E in gap_classes for a, b, _ in E)
        return total - kept

    goal = target_edges if target_edges is not None \
        else int(round(target_f * total))
    rm = 0
    step = 0.005 if restrict else 0.02      # finer steps near a tight goal
    for _ in range(2000 if restrict else 600):
        if rm >= goal:
            break
        for k, E in enumerate(chem_classes):
            if not E:
                continue
            E2, _ = _swap_rewire_directed(
                E, rng, attempts=max(1, int(step * len(E))))
            chem_classes[k] = E2
            if restrict:
                rm = removed()
                if rm >= goal:
                    break
        if not gap_frozen:
            for k, E in enumerate(gap_classes):
                E2, _ = _swap_rewire_symmetric(
                    E, rng, attempts=max(1, int(step * len(E))))
                gap_classes[k] = E2
        rm = removed()
    f = rm / total
    if verbose:
        print(f"  partial_rewire goal {goal} edges -> removed {rm} "
              f"(f={f:.3f})")

    out_chem = np.zeros_like(chem)
    for E in chem_classes + (frozen_chem if restrict else []):
        for i, j, w in E:
            out_chem[i, j] = w
    out_gap = np.zeros_like(gap)
    np.fill_diagonal(out_gap, np.diag(gap))
    for E in gap_classes:
        for a, b, w in E:
            out_gap[a, b] = w; out_gap[b, a] = w
    return out_chem, out_gap, f


# ------------------------------------------------------- graph arms by name

def build_graph(kind: str):
    """Graph spec -> (chem, gap, input_idx, n). Kinds: worm | shuffle<g> |
    rw<pct>g<g> (partial rewire, e.g. rw25g3) | dense78 | dense448."""
    c = load()
    if kind == "worm":
        return c.chem, c.gap, c.sensory, c.n
    if kind.startswith("rwS") or kind.startswith("rwX"):
        # targeted rewiring, matched budget K=200 edges (pre-registered;
        # within-pool swap recreation caps removals at ~220 of the 316
        # input-out edges, so 200 is the largest budget both variants
        # reach): rwS<g> = swaps confined to input-sensory out-edges;
        # rwX<g> = swaps confined to all other chem edges; gap untouched.
        g = int(kind[3:].lstrip("g"))
        mode = "input_out" if kind[2] == "S" else "not_input_out"
        ch, gp, _ = partial_rewire(c.chem, c.gap, node_classes(c.names),
                                   0.0, seed=g, restrict=mode,
                                   input_idx=c.sensory, target_edges=200)
        return ch, gp, c.sensory, c.n
    if kind.startswith("rw"):
        pct, g = kind[2:].split("g")
        ch, gp, _ = partial_rewire(c.chem, c.gap, node_classes(c.names),
                                   int(pct) / 100.0, seed=int(g))
        return ch, gp, c.sensory, c.n
    if kind.startswith("shuffle"):
        g = int(kind[7:])
        ch, gp = typed_shuffle(c.chem, c.gap, node_classes(c.names),
                               seed=g, verbose=False)
        return ch, gp, c.sensory, c.n
    if kind == "dense448":
        n = c.n
        return (np.ones((n, n), np.float32), np.zeros((n, n), np.float32),
                c.sensory, n)
    if kind == "dense78":
        n = 78
        return (np.ones((n, n), np.float32), np.zeros((n, n), np.float32),
                np.arange(22, dtype=np.int64), n)
    raise ValueError(f"unknown graph kind {kind}")
