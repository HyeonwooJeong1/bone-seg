import numpy as np
from ai_bone.merit.split import pca_conflict_split

def test_splits_opposing_groups():
    # 두 무리: +방향 3개, -방향 3개 → 서로 다른 파티션
    g = {}
    for i in range(3): g[f"pos{i}"] = np.array([1.0,0.0]) + 0.01*i
    for i in range(3): g[f"neg{i}"] = np.array([-1.0,0.0]) + 0.01*i
    part = pca_conflict_split(g, k=2)
    groups = list(part.values())
    # pos끼리, neg끼리 같은 그룹
    gid = {name: p for p, names in part.items() for name in names}
    assert gid["pos0"]==gid["pos1"]==gid["pos2"]
    assert gid["neg0"]==gid["neg1"]==gid["neg2"]
    assert gid["pos0"]!=gid["neg0"]

def test_all_datasets_assigned():
    g={f"d{i}":np.random.default_rng(i).normal(size=5) for i in range(7)}
    part=pca_conflict_split(g,k=2)
    assigned=[n for names in part.values() for n in names]
    assert sorted(assigned)==sorted(g)
