from ai_bone.train.make_splits import make_splits, stratified_kfold


def _toy(n_per):
    c2d = {}
    for ds, n in n_per.items():
        for i in range(n):
            c2d[f"{ds}__c{i}"] = ds
    return c2d


def test_domain_holdout_fully_in_test_never_in_folds():
    c2d = _toy({"totalseg": 40, "verse": 20, "spinemets": 10})
    splits, test = make_splits(c2d, n_folds=5, test_frac=0.0,
                               domain_holdout=("spinemets",), seed=7)
    sm = {c for c in c2d if c.startswith("spinemets__")}
    assert sm <= set(test)                                   # all Spine-Mets held out
    infolds = {c for s in splits for c in s["train"] + s["val"]}
    assert not (sm & infolds)                                # none leak into folds


def test_test_frac_holds_out_per_dataset():
    c2d = _toy({"totalseg": 100, "verse": 100})
    splits, test = make_splits(c2d, n_folds=5, test_frac=0.2,
                               domain_holdout=(), seed=7)
    ts = [c for c in test if c.startswith("totalseg__")]
    vs = [c for c in test if c.startswith("verse__")]
    assert len(ts) == 20 and len(vs) == 20                   # 20% each


def test_folds_partition_trainpool_exactly_once():
    c2d = _toy({"totalseg": 37, "verse": 23})
    splits, test = make_splits(c2d, n_folds=5, test_frac=0.1,
                               domain_holdout=(), seed=3)
    trainpool = {c for c in c2d} - set(test)
    val_union = [c for s in splits for c in s["val"]]
    assert sorted(val_union) == sorted(trainpool)            # covers pool
    assert len(val_union) == len(set(val_union))             # each val exactly once
    for s in splits:                                         # train = pool minus val
        assert set(s["train"]) == trainpool - set(s["val"])


def test_stratified_each_fold_has_every_dataset():
    c2d = _toy({"totalseg": 50, "verse": 50, "ribseg": 50})
    splits, _ = make_splits(c2d, n_folds=5, test_frac=0.0, domain_holdout=(), seed=1)
    for s in splits:
        ds_in_val = {c.split("__")[0] for c in s["val"]}
        assert ds_in_val == {"totalseg", "verse", "ribseg"}  # every dataset represented


def test_deterministic_same_seed():
    c2d = _toy({"totalseg": 30, "verse": 30, "spinemets": 8})
    a = make_splits(c2d, seed=99)
    b = make_splits(c2d, seed=99)
    assert a == b                                            # reproducible
