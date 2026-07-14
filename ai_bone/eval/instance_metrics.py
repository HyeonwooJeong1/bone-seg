"""Instance / identification metrics for individual-bone segmentation.

Grounded in the bone benchmarks (see docs/metrics_justification.md):
  - Panoptic Quality PQ = RQ x SQ (panoptica / SPINEPS / VerSe), TP @ IoU>=0.5.
  - Identification Rate + centroid localization error (VerSe).
  - Adjacent-label confusion (off-by-one enumeration, L/R swaps).

Each unified taxonomy id already denotes ONE instance (e.g. a specific vertebra
or rib), so instance matching is per-label: a GT instance is a true positive when
the same-id prediction overlaps it with IoU >= threshold. Off-by-one / swap errors
surface as FN+FP (the prediction carries a neighbouring id) and in the confusion map.
"""
import numpy as np
from scipy.ndimage import center_of_mass


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def instance_scores(gt, pred, label_ids, iou_thr=0.5):
    """Panoptic-style scores over a fixed set of instance labels.
    Returns {tp, fp, fn, rq, sq, pq, id_rate, gt_present}.
      RQ = TP/(TP+0.5FP+0.5FN) (detection F1), SQ = mean IoU over TP, PQ = RQ*SQ,
      id_rate = TP / (#GT-present labels)  (VerSe identification rate)."""
    tp = fp = fn = gt_present = 0
    ious = []
    for lid in label_ids:
        g = gt == lid; p = pred == lid
        gp, pp = g.any(), p.any()
        if not gp and not pp:
            continue                      # label absent in both → not scored
        if gp:
            gt_present += 1
        iou = _iou(g, p) if (gp and pp) else 0.0
        if gp and pp and iou >= iou_thr:
            tp += 1; ious.append(iou)
        else:
            if gp: fn += 1
            if pp: fp += 1
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom > 0 else float("nan")
    sq = float(np.mean(ious)) if ious else 0.0
    pq = rq * sq if not (isinstance(rq, float) and np.isnan(rq)) else float("nan")
    id_rate = tp / gt_present if gt_present else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "rq": rq, "sq": sq, "pq": pq,
            "id_rate": id_rate, "gt_present": gt_present}


def localization_error(gt, pred, label_ids, spacing):
    """Mean centroid Euclidean distance (mm) over instances present in both (VerSe)."""
    sp = np.asarray(spacing, float)
    errs = []
    for lid in label_ids:
        g = gt == lid; p = pred == lid
        if not g.any() or not p.any():
            continue
        cg = np.asarray(center_of_mass(g)); cp = np.asarray(center_of_mass(p))
        errs.append(float(np.linalg.norm((cg - cp) * sp)))
    return float(np.mean(errs)) if errs else float("nan")


def confusion_pairs(gt, pred, label_ids, min_frac=0.01):
    """For each GT label, the fraction of its voxels predicted as each label.
    Captures off-by-one enumeration (e.g. T12→L1) and adjacent confusions.
    Returns {gt_id: {pred_id: frac, ...}} keeping entries >= min_frac."""
    out = {}
    for lid in label_ids:
        g = gt == lid
        n = int(g.sum())
        if n == 0:
            continue
        vals, counts = np.unique(pred[g], return_counts=True)
        frac = {int(v): float(c / n) for v, c in zip(vals, counts) if c / n >= min_frac}
        out[int(lid)] = frac
    return out


def rib_recall(gt, pred, rib_ids):
    """Per-rib recall = |gt_i ∩ pred_i| / |gt_i|, for GT-present ribs (RibSeg v2)."""
    out = {}
    for lid in rib_ids:
        g = gt == lid
        n = int(g.sum())
        if n:
            out[int(lid)] = float((pred[g] == lid).sum() / n)
    return out


def label_accuracy(recalls, thr=0.7):
    """RibSeg v2 Label-Accuracy: fraction of ribs with recall > thr."""
    vals = list(recalls.values())
    return float(np.mean([v > thr for v in vals])) if vals else float("nan")


def lr_swap_rate(gt, pred, lr_pairs):
    """Left/Right swap rate. lr_pairs = [(left_id, right_id), ...]. For each
    GT-present side, a swap = more of its voxels predicted as the OPPOSITE side
    than the correct side. Returns swaps / (#sides evaluated)."""
    swaps = total = 0
    for left, right in lr_pairs:
        for a, b in ((left, right), (right, left)):
            g = gt == a
            if not g.any():
                continue
            total += 1
            same = float((pred[g] == a).mean())
            opp = float((pred[g] == b).mean())
            if opp > same:
                swaps += 1
    return swaps / total if total else float("nan")
