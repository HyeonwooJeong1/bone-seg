"""Aggregate evaluation over a folder of prediction/GT pairs.

- evaluate_dir: per-class semantic metrics (Dice, NSD@tau, HD95).
- region_summary / difficulty_summary: macro-average over anatomical / difficulty
  groups (docs/experiment_design.md §4).
- evaluate_instances_dir: instance/identification metrics for the individual-bone
  groups (spine, ribs) + centroid localization + L/R swap, per case then averaged.
"""
import glob
import os

import numpy as np
import SimpleITK as sitk

from ai_bone import taxonomy_v1 as tx
from ai_bone.eval.metrics import dice, hd95, nsd
from ai_bone.eval import bone_groups as bg
from ai_bone.eval.instance_metrics import (
    instance_scores, localization_error, lr_swap_rate,
)


def _pairs(gt_dir, pred_dir):
    for gp in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        cid = os.path.basename(gp)
        pp = os.path.join(pred_dir, cid)
        if os.path.exists(pp):
            yield cid, gp, pp


def evaluate_dir(gt_dir, pred_dir, spacing=0.6, tau=3.0):
    """Per-class Dice / NSD@tau / HD95, averaged over cases (present labels only)."""
    sp = (spacing,) * 3
    res = {n: {"dice": [], "nsd": [], "hd95": []} for n in tx.FG_NAMES}
    for _, gp, pp in _pairs(gt_dir, pred_dir):
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        for i, name in enumerate(tx.FG_NAMES, start=1):
            d = dice(gt, pr, i)
            if np.isnan(d):
                continue
            res[name]["dice"].append(d)
            res[name]["nsd"].append(nsd(gt, pr, i, sp, tau=tau))
            res[name]["hd95"].append(hd95(gt, pr, i, sp))
    out = {}
    for n, v in res.items():
        out[n] = {
            "dice": float(np.nanmean(v["dice"])) if v["dice"] else float("nan"),
            "nsd": float(np.nanmean(v["nsd"])) if v["nsd"] else float("nan"),
            "hd95": float(np.nanmean(v["hd95"])) if v["hd95"] else float("nan"),
            "n": len(v["dice"]),
        }
    return out


def _macro(per_class, names, metric):
    vals = [per_class[n][metric] for n in names
            if n in per_class and not np.isnan(per_class[n][metric])]
    return float(np.mean(vals)) if vals else float("nan")


def region_summary(per_class, metric="dice"):
    """Macro-average a metric over anatomical region groups."""
    return {region: _macro(per_class, [tx.id_to_name(i) for i in ids], metric)
            for region, ids in bg.REGION_GROUPS.items()}


def difficulty_summary(per_class, metric="dice"):
    """Macro-average a metric over difficulty strata (thin ribs, vertebrae, ...)."""
    return {stratum: _macro(per_class, [tx.id_to_name(i) for i in ids], metric)
            for stratum, ids in bg.DIFFICULTY_STRATA.items()}


def evaluate_instances_dir(gt_dir, pred_dir, spacing=0.6):
    """Instance/identification metrics averaged over cases:
    per-group PQ/RQ/SQ/id_rate for spine and ribs, centroid localization error,
    and left/right swap rate."""
    sp = (spacing,) * 3
    acc = {"spine": [], "ribs": [], "loc_err": [], "lr_swap": []}
    for _, gp, pp in _pairs(gt_dir, pred_dir):
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        acc["spine"].append(instance_scores(gt, pr, bg.SPINE))
        acc["ribs"].append(instance_scores(gt, pr, bg.RIBS))
        acc["loc_err"].append(localization_error(gt, pr, bg.VERTEBRAE + bg.RIBS, sp))
        acc["lr_swap"].append(lr_swap_rate(gt, pr, bg.LR_PAIRS))

    def _avg(rows):
        keys = ("pq", "rq", "sq", "id_rate")
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys} if rows else {}

    return {
        "spine": _avg(acc["spine"]),
        "ribs": _avg(acc["ribs"]),
        "loc_err_mm": float(np.nanmean(acc["loc_err"])) if acc["loc_err"] else float("nan"),
        "lr_swap_rate": float(np.nanmean(acc["lr_swap"])) if acc["lr_swap"] else float("nan"),
    }
