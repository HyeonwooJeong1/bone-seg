import numpy as np
from ai_bone.merit.merge import weighted_average, ties_merge

def test_weighted_average_convex():
    a={"w":np.array([0.0,0.0])}; b={"w":np.array([2.0,4.0])}
    out=weighted_average([a,b],[0.25,0.75])
    assert np.allclose(out["w"], [1.5,3.0])

def test_weights_normalized():
    a={"w":np.array([1.0])}; b={"w":np.array([3.0])}
    out=weighted_average([a,b],[1,1])
    assert np.allclose(out["w"], [2.0])

def test_ties_preserves_shape_and_sign():
    base={"w":np.array([0.0,0.0,0.0])}
    m1={"w":np.array([1.0,0.0,-1.0])}
    m2={"w":np.array([1.0,0.0, 0.0])}
    out=ties_merge(base,[m1,m2],[1,1],density=1.0)
    assert out["w"].shape==(3,)
    assert out["w"][0] > 0            # 부호 합의(+) 유지
