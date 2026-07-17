"""MERIT branch orchestration — turn a dataset partition into trainable branches.

Three partition strategies (the paper's core split comparison, C3):
  - `assign_conflict`  : gradient-conflict PCA split (MERIT). Needs per-dataset
    gradient vectors from `estimate_conflict.py` (GPU-derived).
  - `assign_anatomy`   : anatomical-prior grouping (the user's idea) — spine
    datasets together, ribs, pelvis, whole-body. NO gradients needed (runs now).
  - `assign_random`    : random balanced split (baseline). Deterministic by seed.

Given a partition {branch: [datasets]} and the base CV splits (from make_splits),
`branch_splits_as_folds` builds a splits list where **entry b = branch b**: feed it
to nnU-Net as `fold=b` to train branch b on ONLY its datasets' cases (val
restricted the same way). `branch_weights` gives the case counts for the
token/case-weighted merge. The merged model is evaluated on the common held-out
test set (which is absent from every branch by construction).

Pure logic (numpy only for the conflict path), unit-tested in ct_env.
"""
import json
import os
import random

# Anatomical region of each source dataset (for the anatomy-prior baseline).
DATASET_REGION = {
    "totalseg": "wholebody",
    "ctpelvic1k": "pelvis",
    "ribseg": "ribs",
    "ctspine1k": "spine",
    "verse": "spine",
    "spinemets": "spine",
}


def assign_conflict(grad_vectors, k=2):
    """{dataset: grad vector} → {branch: [datasets]} via MERIT PCA conflict split."""
    from ai_bone.merit.split import pca_conflict_split
    return pca_conflict_split(grad_vectors, k=k)


def assign_anatomy(dataset_names, region_of=None):
    """Group datasets by anatomical region → {branch: [datasets]} (branches ordered
    by region name). The user's 'similar bones together' baseline; no GPU needed."""
    region_of = region_of or DATASET_REGION
    by_region = {}
    for d in dataset_names:
        by_region.setdefault(region_of.get(d, "other"), []).append(d)
    return {i: sorted(by_region[r]) for i, r in enumerate(sorted(by_region))}


def assign_random(dataset_names, k=2, seed=1337):
    """Random balanced partition of datasets into k branches (deterministic)."""
    names = sorted(dataset_names)
    random.Random(seed).shuffle(names)
    part = {i: [] for i in range(k)}
    for i, n in enumerate(names):
        part[i % k].append(n)
    return {b: sorted(v) for b, v in part.items() if v}


def branch_splits_as_folds(case_to_dataset, partition, base_splits, base_fold=0):
    """Build a splits_final.json-style list where entry b = branch b's train/val,
    filtered from base_splits[base_fold] to that branch's datasets. Train nnU-Net
    `fold=b` to get branch b's model. Branch order = sorted(partition)."""
    fold = base_splits[base_fold]
    out = []
    for b in sorted(partition):
        dset = set(partition[b])
        out.append({
            "train": [c for c in fold["train"] if case_to_dataset.get(c) in dset],
            "val": [c for c in fold["val"] if case_to_dataset.get(c) in dset],
        })
    return out


def branch_weights(branch_folds):
    """[{train,val}, ...] (branch-as-fold list) → [n_train per branch] for the
    case-weighted merge (MERIT token weighting → case weighting)."""
    return [len(b["train"]) for b in branch_folds]


def build(strategy, case_to_dataset, base_splits, k=2, grad_vectors=None,
          base_fold=0, seed=1337):
    """One call: partition + branch-as-fold splits + merge weights.
    strategy ∈ {'conflict','anatomy','random'}. Returns a dict artifact."""
    datasets = sorted(set(case_to_dataset.values()))
    if strategy == "conflict":
        if grad_vectors is None:
            raise ValueError("conflict strategy needs grad_vectors (from estimate_conflict)")
        partition = assign_conflict(grad_vectors, k=k)
    elif strategy == "anatomy":
        partition = assign_anatomy(datasets)
    elif strategy == "random":
        partition = assign_random(datasets, k=k, seed=seed)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    folds = branch_splits_as_folds(case_to_dataset, partition, base_splits, base_fold)
    return {
        "strategy": strategy,
        "partition": {str(b): v for b, v in sorted(partition.items())},
        "branch_folds": folds,                 # feed to nnU-Net as splits, fold=b
        "merge_weights": branch_weights(folds),
        "base_fold": base_fold,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build MERIT branch artifact (GPU-free).")
    ap.add_argument("--strategy", required=True, choices=["conflict", "anatomy", "random"])
    ap.add_argument("--raw", required=True, help="merged raw dir (case_datasets.json)")
    ap.add_argument("--splits", required=True, help="base splits_final.json (from make_splits)")
    ap.add_argument("--out", required=True, help="output artifact json (+ a splits file)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--grad-vectors", default=None,
                    help="npz of {dataset: vector} for the conflict strategy")
    ap.add_argument("--base-fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    from ai_bone.train.make_splits import load_case_datasets
    c2d = load_case_datasets(args.raw)
    base_splits = json.loads(open(args.splits, encoding="utf-8").read())
    gv = None
    if args.grad_vectors:
        import numpy as np
        z = np.load(args.grad_vectors)
        gv = {k: z[k] for k in z.files}
    art = build(args.strategy, c2d, base_splits, k=args.k, grad_vectors=gv,
                base_fold=args.base_fold, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1)
    # also drop the branch-as-fold splits next to it, ready for nnU-Net
    sp = os.path.splitext(args.out)[0] + "_splits.json"
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(art["branch_folds"], f, indent=1)
    print(f"{args.strategy}: {len(art['branch_folds'])} branches, "
          f"partition={art['partition']}, weights={art['merge_weights']} → {args.out}")


if __name__ == "__main__":
    main()
