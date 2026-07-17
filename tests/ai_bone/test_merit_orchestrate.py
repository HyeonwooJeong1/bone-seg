import numpy as np
from ai_bone.merit.orchestrate import (
    assign_anatomy, assign_random, assign_conflict,
    branch_splits_as_folds, branch_weights, build,
)

DATASETS = ["totalseg", "ctpelvic1k", "ribseg", "ctspine1k", "verse", "spinemets"]


def _case_to_dataset():
    c2d = {}
    for d in DATASETS:
        for i in range(6):
            c2d[f"{d}__c{i}"] = d
    return c2d


def _base_splits(c2d):
    ids = sorted(c2d)
    val = ids[::5]
    train = [c for c in ids if c not in set(val)]
    return [{"train": train, "val": val}]


def test_anatomy_groups_spine_together():
    part = assign_anatomy(DATASETS)
    groups = [set(v) for v in part.values()]
    assert {"ctspine1k", "verse", "spinemets"} in groups   # spine datasets one branch
    assert {"ribseg"} in groups and {"ctpelvic1k"} in groups
    assert sum(len(v) for v in part.values()) == len(DATASETS)   # every dataset once


def test_random_is_balanced_and_deterministic():
    a = assign_random(DATASETS, k=2, seed=5)
    b = assign_random(DATASETS, k=2, seed=5)
    assert a == b                                          # deterministic
    sizes = sorted(len(v) for v in a.values())
    assert sizes == [3, 3]                                 # balanced 6→3/3
    allds = sorted(d for v in a.values() for d in v)
    assert allds == sorted(DATASETS)                       # partition (no dup/loss)


def test_conflict_split_partitions_by_gradient():
    # two clearly-opposed gradient clusters → split separates them
    gv = {"a": np.array([1.0, 0.0]), "b": np.array([0.9, 0.1]),
          "c": np.array([-1.0, 0.0]), "d": np.array([-0.9, -0.1])}
    part = assign_conflict(gv, k=2)
    groups = [set(v) for v in part.values()]
    assert {"a", "b"} in groups and {"c", "d"} in groups


def test_branch_folds_restrict_to_branch_datasets():
    c2d = _case_to_dataset()
    base = _base_splits(c2d)
    part = assign_anatomy(DATASETS)
    folds = branch_splits_as_folds(c2d, part, base)
    assert len(folds) == len(part)                         # one entry per branch
    for b, bid in enumerate(sorted(part)):                 # fold b ↔ branch sorted(part)[b]
        datasets = set(part[bid])
        cases = folds[b]["train"] + folds[b]["val"]
        assert all(c2d[c] in datasets for c in cases)      # only branch datasets
    # every train-pool case appears in exactly one branch (partition of the pool)
    pooled = [c for f in folds for c in f["train"] + f["val"]]
    base_pool = base[0]["train"] + base[0]["val"]
    assert sorted(pooled) == sorted(base_pool)


def test_build_artifact_and_weights():
    c2d = _case_to_dataset()
    base = _base_splits(c2d)
    art = build("anatomy", c2d, base)
    assert art["strategy"] == "anatomy"
    assert len(art["branch_folds"]) == len(art["partition"])
    assert art["merge_weights"] == branch_weights(art["branch_folds"])
    assert sum(art["merge_weights"]) == len(base[0]["train"])   # weights = train counts


def test_build_conflict_requires_grad_vectors():
    c2d = _case_to_dataset(); base = _base_splits(c2d)
    try:
        build("conflict", c2d, base)
        assert False, "should require grad_vectors"
    except ValueError:
        pass
