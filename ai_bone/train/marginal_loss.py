"""Marginal loss for partial-label multi-dataset training (Shi et al., 2021).

A dataset that annotates only some classes (e.g. CTPelvic1K = pelvis) must not
teach "unannotated bone = background". For a case with annotated set P, the
non-annotated foreground channels are collapsed into the background BEFORE the
loss, so predicting an unannotated class in a background voxel is not penalized
(its probability mass counts toward the merged background, which matches the
background target).

- `collapse_to_present` — pure core (numpy or torch), unit-tested.
- `MarginalDiceCELoss` — torch loss applying the collapse per sample (server).
- `nnUNetTrainerMarginal` — wires per-case `present_labels` (from the merged
  dataset's sidecars / case_datasets.json) into the loss (server scaffold).

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


class MarginalDiceCELoss:
    """Marginal Dice + CE (torch, server). __call__(logits, target, present_masks):
    logits (B,C,*sp), target (B,1,*sp) long, present_masks (B,C) bool tensor."""

    def __init__(self, bg_index=0, smooth=1e-5):
        self.bg_index = bg_index
        self.smooth = smooth

    def __call__(self, logits, target, present_masks):
        import torch
        import torch.nn.functional as F
        prob = F.softmax(logits, dim=1)
        B, C = prob.shape[0], prob.shape[1]
        collapsed = prob.clone()
        for b in range(B):
            pm = present_masks[b]
            absorb = None
            for c in range(C):
                if c == self.bg_index or bool(pm[c]):
                    continue
                absorb = prob[b, c] if absorb is None else absorb + prob[b, c]
                collapsed[b, c] = collapsed[b, c] * 0
            if absorb is not None:
                collapsed[b, self.bg_index] = prob[b, self.bg_index] + absorb
        eps = 1e-7
        tgt = target.squeeze(1).long()
        ce = F.nll_loss(collapsed.clamp_min(eps).log(), tgt)
        # soft Dice over the collapsed probabilities
        onehot = F.one_hot(tgt, C).movedim(-1, 1).float()
        dims = tuple(range(2, collapsed.ndim))
        inter = (collapsed * onehot).sum(dims)
        denom = collapsed.sum(dims) + onehot.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        return ce + dice


def main():
    """Server scaffold: a trainer that applies the marginal loss with per-case
    present masks loaded from case_datasets.json + present sidecars. Kept here as a
    reference; on the server, subclass the ES_PL trainer and override the loss +
    train_step to pass present_masks for the batch's case ids."""
    raise SystemExit("import nnUNetTrainerMarginal on the server; see runbook §7")


if __name__ == "__main__":
    main()
