import numpy as np
from ai_bone.merit.estimate_conflict import make_projection, reduce_grad, _random_patch

def test_projection_and_reduce_shapes():
    proj = make_projection(1000, 64, seed=1)
    assert proj.shape == (64, 1000)
    g = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    assert reduce_grad(g, proj).shape == (64,)

def test_projection_is_deterministic():
    assert np.array_equal(make_projection(50, 8, seed=3), make_projection(50, 8, seed=3))

def test_random_patch_crop_and_pad():
    rng = np.random.default_rng(0)
    data = np.zeros((2, 10, 10, 10)); seg = np.zeros((1, 10, 10, 10))
    d, s = _random_patch(data, seg, (4, 4, 4), rng)
    assert d.shape == (2, 4, 4, 4) and s.shape == (1, 4, 4, 4)
    # smaller-than-patch → zero-padded up to patch
    data2 = np.zeros((2, 3, 3, 3)); seg2 = np.zeros((1, 3, 3, 3))
    d2, s2 = _random_patch(data2, seg2, (4, 4, 4), rng)
    assert d2.shape == (2, 4, 4, 4) and s2.shape == (1, 4, 4, 4)
