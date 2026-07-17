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


# --- vectorized / ignore-aware torch loss (ct_env has torch) ---

def test_vectorized_collapse_matches_reference():
    import torch
    from ai_bone.train.marginal_loss import _collapse_probs
    torch.manual_seed(0)
    B, C = 3, 5
    prob = torch.rand(B, C, 2, 3); prob = prob / prob.sum(1, keepdim=True)
    present = torch.tensor([[True, True, False, True, False],
                            [True, False, False, True, True],
                            [True, True, True, True, True]])
    out = _collapse_probs(prob, present)
    for b in range(B):                                     # matches per-sample loop core
        ref = collapse_to_present(prob[b], present[b].tolist())
        assert torch.allclose(out[b], ref, atol=1e-6)
    assert torch.allclose(out.sum(1), prob.sum(1), atol=1e-5)   # probability conserved


def test_all_ignore_gives_near_zero_loss():
    import torch
    from ai_bone.train.marginal_loss import MarginalDiceCELoss
    B, C = 1, 3
    logits = torch.randn(B, C, 4, 4)
    present = torch.ones(B, C, dtype=torch.bool)
    loss = MarginalDiceCELoss(ignore_label=C)              # ignore value just past classes
    tgt = torch.full((B, 1, 4, 4), C, dtype=torch.long)   # every voxel is ignore
    assert abs(float(loss(logits, tgt, present))) < 1e-4


def test_ignore_voxels_excluded_lowers_loss():
    import torch
    from ai_bone.train.marginal_loss import MarginalDiceCELoss
    B, C = 1, 3
    logits = torch.zeros(B, C, 1, 1); logits[0, 0, 0, 0] = 10.0   # confidently predicts bg
    present = torch.ones(B, C, dtype=torch.bool)
    loss = MarginalDiceCELoss(ignore_label=C)
    tgt_valid = torch.ones((B, 1, 1, 1), dtype=torch.long)       # true class 1 → mispredicted
    tgt_ignore = torch.full((B, 1, 1, 1), C, dtype=torch.long)   # same voxel marked ignore
    assert float(loss(logits, tgt_ignore, present)) < float(loss(logits, tgt_valid, present))


def test_marginal_unannotated_prediction_equals_background():
    # Predicting an UNANNOTATED class must be equivalent to predicting background
    # (its mass folds into bg). Annotating that same class instead penalizes it.
    import torch
    from ai_bone.train.marginal_loss import MarginalDiceCELoss
    B, C = 1, 3
    tgt = torch.zeros((B, 1, 1, 1), dtype=torch.long)                # true background
    logits_c2 = torch.zeros(B, C, 1, 1); logits_c2[0, 2, 0, 0] = 10.0   # predict class 2
    logits_bg = torch.zeros(B, C, 1, 1); logits_bg[0, 0, 0, 0] = 10.0   # predict bg (correct)
    loss = MarginalDiceCELoss()
    l_c2_unann = float(loss(logits_c2, tgt, torch.tensor([[True, True, False]])))
    l_bg = float(loss(logits_bg, tgt, torch.tensor([[True, True, False]])))
    l_c2_ann = float(loss(logits_c2, tgt, torch.tensor([[True, True, True]])))
    assert abs(l_c2_unann - l_bg) < 1e-3          # unannotated class ≈ predicting bg
    assert l_c2_ann > l_c2_unann + 1.0            # annotating it → heavily penalized


def test_make_marginal_ds_loss_uses_getter():
    import torch
    from ai_bone.train.marginal_loss import make_marginal_ds_loss, MarginalDiceCELoss
    B, C = 1, 3
    logits = torch.zeros(B, C, 1, 1); logits[0, 2, 0, 0] = 10.0
    tgt = torch.zeros((B, 1, 1, 1), dtype=torch.long)
    pm = torch.tensor([[True, True, False]])
    mod = make_marginal_ds_loss(get_present=lambda: pm)
    ref = MarginalDiceCELoss()(logits, tgt, pm)
    assert torch.allclose(mod(logits, tgt), ref)
