"""Author nnU-Net cross-validation splits for the merged unified-bone dataset.

Why not nnU-Net's default? The default makes a random 5-fold over ALL 3255 cases,
which (a) leaks each source dataset across folds without control and (b) has no
held-out test set — so numbers aren't comparable to public benchmarks and the
pathological domain (Spine-Mets) can leak into training.

This builds, deterministically from a seed:
  1. a held-out TEST set = every case of the `domain_holdout` datasets (default
     Spine-Mets → out-of-distribution / pathological eval) PLUS a `test_frac`
     slice of every other dataset (in-distribution held-out);
  2. a stratified K-fold over the remaining train pool, so each fold's validation
     set has proportional representation from every source dataset.

Outputs nnU-Net's `splits_final.json` (list of {"train":[...], "val":[...]}) — the
union of all folds is the train pool; test cases appear in NO fold, so nnU-Net
never trains/validates on them. Also writes `test_ids.json` for our evaluation.

Pure logic (stdlib random), unit-tested in ct_env.
"""
import json
import os
import random


def _ds_seed(seed, ds):
    """Deterministic per-dataset seed (avoid builtin hash — PYTHONHASHSEED varies)."""
    return seed + sum(ord(c) for c in ds)


def stratified_kfold(ids_by_group, k, seed):
    """ids_by_group: {group: [ids]} → list of k {"train","val"} dicts, each group's
    members spread round-robin across folds so every fold's val is stratified."""
    folds_val = [[] for _ in range(k)]
    for g in sorted(ids_by_group):
        ids = list(ids_by_group[g])
        random.Random(_ds_seed(seed, g)).shuffle(ids)
        for i, cid in enumerate(ids):
            folds_val[i % k].append(cid)
    all_ids = [c for g in ids_by_group for c in ids_by_group[g]]
    splits = []
    for i in range(k):
        val = sorted(folds_val[i])
        val_set = set(val)
        train = sorted(c for c in all_ids if c not in val_set)
        splits.append({"train": train, "val": val})
    return splits


def make_splits(case_to_dataset, n_folds=5, test_frac=0.15, domain_holdout=("spinemets",),
                seed=1337):
    """case_to_dataset: {case_id: dataset_name} → (splits list, sorted test ids)."""
    domain = set(domain_holdout)
    by_ds = {}
    for cid, ds in case_to_dataset.items():
        by_ds.setdefault(ds, []).append(cid)

    test, trainpool_by_ds = [], {}
    for ds in sorted(by_ds):
        ids = sorted(by_ds[ds])
        if ds in domain:
            test.extend(ids)                      # whole dataset → domain-shift test
            continue
        random.Random(_ds_seed(seed, ds)).shuffle(ids)
        n_test = int(round(len(ids) * test_frac))
        test.extend(ids[:n_test])
        trainpool_by_ds[ds] = ids[n_test:]
    splits = stratified_kfold(trainpool_by_ds, n_folds, seed)
    return splits, sorted(test)


def load_case_datasets(raw_dataset_dir):
    """Read case→dataset from merge_raw's case_datasets.json; fall back to the
    `<dataset>__<id>` case-id prefix if the sidecar is absent."""
    import glob
    p = os.path.join(raw_dataset_dir, "case_datasets.json")
    if os.path.exists(p):
        return json.loads(open(p, encoding="utf-8").read())
    out = {}
    for f in glob.glob(os.path.join(raw_dataset_dir, "labelsTr", "*.nii.gz")):
        cid = os.path.basename(f)[: -len(".nii.gz")]
        out[cid] = cid.split("__", 1)[0] if "__" in cid else "unknown"
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Author nnU-Net splits_final.json (GPU-free).")
    ap.add_argument("--raw", required=True, help="merged raw dataset dir (has case_datasets.json)")
    ap.add_argument("--preprocessed", required=True,
                    help="preprocessed dataset dir to write splits_final.json into")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--domain-holdout", nargs="*", default=["spinemets"])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    c2d = load_case_datasets(args.raw)
    splits, test = make_splits(c2d, args.folds, args.test_frac,
                               tuple(args.domain_holdout), args.seed)
    os.makedirs(args.preprocessed, exist_ok=True)
    with open(os.path.join(args.preprocessed, "splits_final.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)
    with open(os.path.join(args.preprocessed, "test_ids.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=1)
    trainpool = len(splits[0]["train"]) + len(splits[0]["val"])
    print(f"folds={args.folds} trainpool={trainpool} test={len(test)} "
          f"(domain_holdout={args.domain_holdout}, test_frac={args.test_frac})")


if __name__ == "__main__":
    main()
