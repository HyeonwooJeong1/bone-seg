import numpy as np
from ai_bone.train.marginal_loss import collapse_to_present, present_mask_from_ids


def test_collapse_absorbs_nonpresent_into_background():
    prob = np.array([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])   # (C=3, N=2)
    present = [True, True, False]                           # class 2 not annotated
    out = collapse_to_present(prob, present)
    assert np.allclose(out[2], 0.0)                         # non-present zeroed
    assert np.allclose(out[0], [0.7, 0.4])                  # bg absorbs class 2
    assert np.allclose(out[1], [0.3, 0.6])                  # present kept
    assert np.allclose(out.sum(0), prob.sum(0))             # probability conserved


def test_collapse_all_present_is_identity():
    prob = np.array([[0.2], [0.3], [0.5]])
    out = collapse_to_present(prob, [True, True, True])
    assert np.allclose(out, prob)


def test_present_mask_from_ids():
    m = present_mask_from_ids([2, 5], 8)
    assert m[0] and m[2] and m[5]                           # bg always on
    assert not m[1] and not m[3]
    assert present_mask_from_ids([], 4).all()              # empty → fully annotated
