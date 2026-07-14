"""Conflict analysis & visualization for MERIT-style dataset partitioning.

Consumes per-dataset gradient vectors (from `estimate_conflict.py`, run on the
server) and produces the paper's method artifacts (docs/experiment_design.md §5.2,
RQ5): the cosine-conflict matrix C, PCA embedding, the conflict-aware partition,
and how well that partition recovers anatomy (ARI/NMI), plus heatmap/scatter plots.

Pure-numpy metrics (ARI/NMI implemented here so no sklearn dependency on the
server); matplotlib is imported lazily only for plotting.
"""
import json
import os
from math import comb, log

import numpy as np

from ai_bone.merit.split import pca_conflict_split


# ----- gradient conflict structure -------------------------------------------
def cosine_matrix(grads):
    """grads: {name: 1-D vector}. Returns (names, C) with C[i,j] = cos(g_i, g_j)."""
    names = list(grads)
    G = np.stack([np.asarray(grads[n], dtype=float) for n in names])
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    return names, Gn @ Gn.T


def pca_embed(C, r=2):
    """Column-centered PCA on the cosine matrix C (MERIT step 2) → (T, r) scores."""
    Cc = C - C.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Cc, full_matrices=False)
    r = min(r, U.shape[1])
    return U[:, :r] * S[:r]


# ----- partition vs anatomy agreement (pure numpy) ---------------------------
def _contingency(a, b):
    la, lb = sorted(set(a)), sorted(set(b))
    ia = {x: i for i, x in enumerate(la)}
    ib = {x: i for i, x in enumerate(lb)}
    m = np.zeros((len(la), len(lb)), dtype=float)
    for x, y in zip(a, b):
        m[ia[x], ib[y]] += 1
    return m


def adjusted_rand_index(a, b):
    m = _contingency(a, b)
    n = int(m.sum())
    if n < 2:
        return 1.0
    sc = sum(comb(int(x), 2) for x in m.sum(1))
    sk = sum(comb(int(x), 2) for x in m.sum(0))
    s = sum(comb(int(v), 2) for v in m.ravel())
    exp = sc * sk / comb(n, 2)
    mx = 0.5 * (sc + sk)
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def normalized_mutual_info(a, b):
    m = _contingency(a, b)
    n = m.sum()
    if n == 0:
        return 1.0
    pij = m / n
    pi, pj = pij.sum(1), pij.sum(0)
    mi = sum(pij[i, j] * log(pij[i, j] / (pi[i] * pj[j]))
             for i in range(m.shape[0]) for j in range(m.shape[1]) if pij[i, j] > 0)
    ha = -sum(p * log(p) for p in pi if p > 0)
    hb = -sum(p * log(p) for p in pj if p > 0)
    den = 0.5 * (ha + hb)
    return 1.0 if den == 0 else mi / den


def agreement(partition, region_of):
    """partition: {gid: [name,...]}; region_of: {name: region}. Returns {ari, nmi}
    between the discovered partition and the anatomical grouping."""
    plabel = {n: g for g, ns in partition.items() for n in ns}
    names = list(plabel)
    a = [plabel[n] for n in names]
    b = [region_of[n] for n in names]
    return {"ari": adjusted_rand_index(a, b), "nmi": normalized_mutual_info(a, b)}


# ----- anatomical region of each dataset (coarse) ----------------------------
_COARSE = {"skull": "skull", "cervical": "spine", "thoracic": "spine",
           "lumbar": "spine", "sacrum": "spine", "ribs": "ribs",
           "sternum": "thorax", "pelvis": "pelvis"}


def _label_region(name):
    from ai_bone.eval import bone_groups as bg
    from ai_bone import taxonomy_v1 as tx
    lid = tx.NAME_TO_ID[name]
    for region, ids in bg.REGION_GROUPS.items():
        if lid in ids:
            return _COARSE[region]
    return "other"


def infer_regions(dataset_names=None):
    """Map each registered dataset → its dominant coarse region, from its
    label_map present_labels."""
    from ai_bone.datasets.registry import DATASETS
    from ai_bone.label_map import load_label_map
    out = {}
    for name in (dataset_names or DATASETS):
        if name not in DATASETS:
            continue
        lm = load_label_map(DATASETS[name].label_map_path)
        regions = [_label_region(n) for n in lm.present_labels]
        out[name] = max(set(regions), key=regions.count) if regions else "other"
    return out


# ----- plots (matplotlib, lazy) ----------------------------------------------
def plot_conflict_matrix(names, C, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.5 * len(names) + 2, 0.5 * len(names) + 2))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("dataset gradient cosine conflict")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def plot_pca_embedding(embed, names, partition, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gid = {n: g for g, ns in partition.items() for n in ns}
    fig, ax = plt.subplots(figsize=(7, 6))
    xs, ys = embed[:, 0], embed[:, 1] if embed.shape[1] > 1 else np.zeros(len(names))
    groups = sorted(set(gid.values()))
    for g in groups:
        idx = [i for i, n in enumerate(names) if gid.get(n) == g]
        ax.scatter(xs[idx], ys[idx], label=f"group {g}", s=60)
    for i, n in enumerate(names):
        ax.annotate(n, (xs[i], ys[i]), fontsize=7)
    ax.set_xlabel("PC1 (conflict)"); ax.set_ylabel("PC2")
    ax.legend(); ax.set_title("conflict PCA embedding + partition")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


def load_grads(npz_path):
    """Load {name: vector} from an .npz saved by estimate_conflict.py."""
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MERIT conflict analysis + plots.")
    ap.add_argument("--grads", required=True, help=".npz of {dataset: gradient vector}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=2, help="number of partitions (2^r)")
    ap.add_argument("--regions", default=None, help="optional {dataset: region} JSON")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    grads = load_grads(args.grads)
    names, C = cosine_matrix(grads)
    embed = pca_embed(C, r=2)
    partition = pca_conflict_split(grads, k=args.k)
    regions = (json.loads(open(args.regions, encoding="utf-8").read())
               if args.regions else infer_regions(names))
    agr = agreement(partition, regions)

    np.save(os.path.join(args.out_dir, "conflict_matrix.npy"), C)
    with open(os.path.join(args.out_dir, "partition.json"), "w", encoding="utf-8") as f:
        json.dump({str(g): ns for g, ns in partition.items()}, f, indent=2)
    with open(os.path.join(args.out_dir, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump({"agreement_vs_anatomy": agr, "regions": regions}, f, indent=2)
    plot_conflict_matrix(names, C, os.path.join(args.out_dir, "conflict_matrix.png"))
    plot_pca_embedding(embed, names, partition, os.path.join(args.out_dir, "pca_embedding.png"))
    print("partition:", {g: ns for g, ns in partition.items()})
    print("agreement vs anatomy:", agr)


if __name__ == "__main__":
    main()
