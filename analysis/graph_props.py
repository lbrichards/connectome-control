"""Task G: compute the 10 pre-registered graph properties (+2 registered
side metrics) for every graph. Definitions exactly as in the README
pre-registration. Caches to ~/cc-queue/taskG_props.json.

usage: graph_props.py            (all graphs: 90 shuffles, 32 rw, worm)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from connectome_control.graph import load, build_graph, node_classes

Q = os.path.expanduser("~/cc-queue")
OUT = f"{Q}/taskG_props.json"


def props_for(kind, c, isn):
    chem_w, gap_w, in_idx, n = build_graph(kind)
    A = (chem_w > 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    Af = A.astype(np.float64)
    U = (((chem_w + chem_w.T + gap_w) > 0).astype(np.int8))
    np.fill_diagonal(U, 0)

    out = {}
    E = int(A.sum())
    # 1. reciprocity (overall; and neuron->neuron side metric)
    out["reciprocity"] = float((A & A.T).sum() / E)
    nn = np.outer(isn, isn)
    Enn = int((A & nn).sum())
    out["reciprocity_nn"] = float(((A & A.T) & nn).sum() / Enn)
    # 2. motifs
    A2 = Af @ Af
    out["ffl"] = float((A2 * Af).sum())
    out["cycle3"] = float(np.trace(A2 @ Af) / 3.0)
    # 3. near-input cycles (within 2 directed hops downstream of inputs)
    # NOTE adjacency convention: chem[i,j] = j -> i (row = post), so the
    # FLOW-direction matrix for traversal is Ad = A.T (Ad[i,j] = i -> j).
    Ad = Af.T
    e = np.zeros(n); e[in_idx] = 1
    reach = (e + e @ Ad + (e @ Ad) @ Ad) > 0
    sub = Af[np.ix_(reach, reach)]
    c2 = float((sub * sub.T).sum() / 2.0)
    c3 = float(np.trace(sub @ sub @ sub) / 3.0)
    out["near_input_cycles"] = c2 + c3
    # 4. mean shortest path inputs -> effectors (directed, unweighted)
    S = csr_matrix(A.T)                      # flow direction (see above)
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(S, method="D", directed=True, unweighted=True,
                      indices=np.asarray(in_idx))
    eff = ~isn
    d = D[:, eff]
    finite = np.isfinite(d)
    out["path_in_eff"] = float(d[finite].mean())
    out["path_unreachable"] = int((~finite).sum())
    # 5. 2-hop reach
    out["reach2"] = int(reach.sum())
    # 6. largest SCC
    ncomp, lab = connected_components(S, directed=True, connection="strong")
    out["lscc"] = int(np.bincount(lab).max())
    # 7. spectral radius of count-weighted chem+gap
    W = chem_w.astype(np.float64) + gap_w.astype(np.float64)
    out["spectral_radius"] = float(np.abs(np.linalg.eigvals(W)).max())
    # 8. clustering (union)
    Gu = nx.from_numpy_array(U)
    out["clustering"] = float(nx.average_clustering(Gu))
    # 9. modularity (Louvain, seed 0, unweighted union)
    from networkx.algorithms.community import louvain_communities, modularity
    comms = louvain_communities(Gu, weight=None, seed=0)
    out["modularity"] = float(modularity(Gu, comms, weight=None))
    # 10. rich club at k*=77 (union, unnormalized)
    deg = U.sum(1)
    rich = deg >= 77
    nr = int(rich.sum())
    out["rich_club_77"] = (float(U[np.ix_(rich, rich)].sum() /
                                 (nr * (nr - 1))) if nr > 1 else 0.0)
    return out


def main():
    c = load()
    isn = node_classes(c.names)
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    kinds = (["worm"] + [f"shuffle{g}" for g in range(60)]
             + [f"shuffle{g}" for g in range(200, 230)]
             + [f"rw{p}g{g}" for p in (10, 25, 50, 75) for g in range(8)])
    for k in kinds:
        if k in done:
            continue
        done[k] = props_for(k, c, isn)
        json.dump(done, open(OUT, "w"))
        print(k, "ok", flush=True)
    print("PROPS DONE", len(done), flush=True)


if __name__ == "__main__":
    main()
