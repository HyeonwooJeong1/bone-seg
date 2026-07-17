"""Marginal loss for partial-label multi-dataset training (Shi et al., 2021).

A dataset that annotates only some classes (e.g. CTPelvic1K = pelvis) must not
teach "unannotated bone = background". For a case with annotated set P, the
non-annotated foreground channels are collapsed into the background BEFORE the
loss, so predicting an unannotated class in a background voxel is not penalized
(its probability mass counts toward the merged background, which matches the
background target).

Two orthogonal partial-label mechanisms are combined here:
  - **marginal collapse** (this file): whole *classes* a dataset did not annotate.
  - **ignore label** (`ignore_label`, taxonomy value 54): individual *voxels*
    labelled as ignore (L6 / T13 / lesions) — masked out of the loss.

Components:
  - `collapse_to_present` — pure reference core (numpy or torch), unit-tested.
  - `present_mask_from_ids` — case present-id list → length-C bool mask.
  - `MarginalDiceCELoss` — vectorized, ignore-aware Dice+CE (torch); the compute,
    callable with explicit `present_masks` so it is unit-testable off-server.
  - `MarginalDeepSupervisionLoss` — nn.Module that pulls the current batch's
    present masks from a getter and delegates to `MarginalDiceCELoss`; this is
    what the trainer drops into nnU-Net's deep-supervision wrapper so the loss
    call signature stays the standard `loss(output, target)`.

TotalSeg (whole-body, all classes annotated) reduces to the standard loss; only
partial datasets benefit.
"""
import numpy as np


def collapse_to_present(prob, present, bg_index=0):
    """prob: (C, *spatial) class probabilities; present: length-C bool (True =
    annotated). Non-present foreground channels are zeroed and their mass is added
    to the background channel. Probability mass is conserved. numpy or torch."""
    out = prob.clone() if hasattr(prob, "clone") else prob.copy()
    absorb = None
    for c in range(prob.shape[0]):
        if c == bg_index or present[c]:
            continue
        absorb = prob[c] if absorb is None else absorb + prob[c]
        out[c] = out[c] * 0
    if absorb is not None:
        out[bg_index] = out[bg_index] + absorb
    return out


def present_mask_from_ids(present_ids, num_classes, bg_index=0):
    """present_ids (iterable of foreground class ids) → length-num_classes bool mask
    with background always True. Empty → all True (treat as fully annotated)."""
    m = np.zeros(num_classes, dtype=bool)
    m[bg_index] = True
    ids = list(present_ids)
    if not ids:
        m[:] = True
        return m
    for i in ids:
        if 0 <= int(i) < num_classes:
            m[int(i)] = True
    return m


def _collapse_probs(prob, present_masks, bg_index=0):
    """Vectorized marginal collapse for a batch (torch).
    prob: (B, C, *sp) softmax probs; present_masks: (B, C) bool/float.
    Non-present fg channels are zeroed, their mass folded into background.
    Equivalent to `collapse_to_present` applied per sample. Mass conserved."""
    B, C = prob.shape[0], prob.shape[1]
    view = (B, C) + (1,) * (prob.ndim - 2)
    keep = present_masks.to(prob.dtype).view(view).clone()   # (B,C,1,...)
    keep[:, bg_index] = 1.0                                   # background always kept
    absorbed = (prob * (1.0 - keep)).sum(dim=1, keepdim=True)  # (B,1,*sp)
    collapsed = prob * keep
    bg = collapsed[:, bg_index:bg_index + 1] + absorbed
    collapsed = collapsed.clone()
    collapsed[:, bg_index:bg_index + 1] = bg
    return collapsed


class MarginalDiceCELoss:
    """Vectorized, ignore-aware marginal Dice + CE (torch).

    __call__(logits, target, present_masks):
      logits (B,C,*sp), target (B,1,*sp) long, present_masks (B,C) bool tensor.
    Voxels where target == ignore_label are excluded from both CE and Dice."""

    def __init__(self, ignore_label=None, bg_index=0, smooth=1e-5, eps=1e-7):
        self.ignore_label = ignore_label
        self.bg_index = bg_index
        self.smooth = smooth
        self.eps = eps

    def __call__(self, logits, target, present_masks):
        import torch
        import torch.nn.functional as F
        C = logits.shape[1]
        prob = F.softmax(logits, dim=1)
        collapsed = _collapse_probs(prob, present_masks, self.bg_index)

        tgt = target.long()
        if tgt.ndim == prob.ndim:            # (B,1,*sp) → (B,*sp)
            tgt = tgt[:, 0]
        if self.ignore_label is not None:
            valid = (tgt != self.ignore_label)               # (B,*sp) bool
            tgt = torch.where(valid, tgt, torch.zeros_like(tgt))  # ignore→bg for one_hot
        else:
            valid = torch.ones_like(tgt, dtype=torch.bool)
        vf = valid.unsqueeze(1).to(prob.dtype)               # (B,1,*sp)

        # masked cross-entropy over the collapsed probabilities
        logp = collapsed.clamp_min(self.eps).log()
        ce_map = F.nll_loss(logp, tgt, reduction="none").unsqueeze(1)  # (B,1,*sp)
        ce = (ce_map * vf).sum() / vf.sum().clamp_min(1.0)

        # masked soft Dice
        onehot = F.one_hot(tgt, C).movedim(-1, 1).to(prob.dtype) * vf   # (B,C,*sp)
        collapsed_m = collapsed * vf
        dims = tuple(range(2, collapsed.ndim))
        inter = (collapsed_m * onehot).sum(dims)
        denom = collapsed_m.sum(dims) + onehot.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        return ce + dice


def make_marginal_ds_loss(get_present, ignore_label=None, bg_index=0):
    """Build an nn.Module that pulls the current batch's present masks from
    `get_present()` (the trainer sets it before each forward) and delegates to
    `MarginalDiceCELoss`, keeping nnU-Net's standard `forward(output, target)`
    signature so it drops straight into DeepSupervisionWrapper. torch is imported
    lazily here so this file still imports in a torch-free env for the numpy core."""
    import torch
    import torch.nn as nn

    core = MarginalDiceCELoss(ignore_label=ignore_label, bg_index=bg_index)

    class _MarginalDSLoss(nn.Module):
        def forward(self, output, target):
            pm = get_present() if get_present is not None else None
            if pm is None:                       # safety: treat as fully annotated
                B, C = output.shape[0], output.shape[1]
                pm = torch.ones((B, C), dtype=torch.bool, device=output.device)
            return core(output, target, pm)

    return _MarginalDSLoss()
