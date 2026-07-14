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
from ai_bone.eval.metrics import dice, hd95, nsd, cldice
from ai_bone.eval import bone_groups as bg
from ai_bone.eval.instance_metrics import (
    instance_scores, localization_error, lr_swap_rate, rib_recall, label_accuracy,
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
    acc = {"spine": [], "ribs": [], "loc_err": [], "lr_swap": [], "trans_enum": [],
           "rib_la_all": [], "rib_la_first": [], "rib_la_inter": [], "rib_la_twelfth": [],
           "rib_cldice": []}
    for _, gp, pp in _pairs(gt_dir, pred_dir):
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        acc["spine"].append(instance_scores(gt, pr, bg.SPINE))
        acc["ribs"].append(instance_scores(gt, pr, bg.RIBS))
        acc["loc_err"].append(localization_error(gt, pr, bg.VERTEBRAE + bg.RIBS, sp))
        acc["lr_swap"].append(lr_swap_rate(gt, pr, bg.LR_PAIRS))
        # off-by-one enumeration at transition zones (reuse the swap detector)
        acc["trans_enum"].append(lr_swap_rate(gt, pr, bg.TRANSITION_ZONES))
        # RibSeg v2 Label-Accuracy (recall>0.7), stratified by rib position
        rr = rib_recall(gt, pr, bg.RIBS)
        acc["rib_la_all"].append(label_accuracy(rr))
        acc["rib_la_first"].append(label_accuracy({k: rr[k] for k in bg.RIB_FIRST if k in rr}))
        acc["rib_la_inter"].append(label_accuracy({k: rr[k] for k in bg.RIB_INTERMEDIATE if k in rr}))
        acc["rib_la_twelfth"].append(label_accuracy({k: rr[k] for k in bg.RIB_TWELFTH if k in rr}))
        # centerline Dice for thin ribs (mean over GT-present ribs)
        cl = [cldice(gt, pr, i) for i in bg.RIBS if (gt == i).any()]
        acc["rib_cldice"].append(float(np.nanmean(cl)) if cl else float("nan"))

    def _avg(rows):
        keys = ("pq", "rq", "sq", "id_rate")
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys} if rows else {}

    def _m(key):
        return float(np.nanmean(acc[key])) if acc[key] else float("nan")

    return {
        "spine": _avg(acc["spine"]),
        "ribs": _avg(acc["ribs"]),
        "loc_err_mm": _m("loc_err"),
        "lr_swap_rate": _m("lr_swap"),
        "transition_enum_error": _m("trans_enum"),
        "rib_label_accuracy": {"all": _m("rib_la_all"), "first": _m("rib_la_first"),
                               "intermediate": _m("rib_la_inter"), "twelfth": _m("rib_la_twelfth")},
        "rib_cldice": _m("rib_cldice"),
    }


def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Evaluate segmentation (semantic + instance).")
    ap.add_argument("--gt", required=True, help="ground-truth dir of *.nii.gz")
    ap.add_argument("--pred", required=True, help="prediction dir (same filenames)")
    ap.add_argument("--spacing", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=3.0, help="NSD tolerance (mm)")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()
    per_class = evaluate_dir(args.gt, args.pred, spacing=args.spacing, tau=args.tau)
    report = {
        "per_class": per_class,
        "region_dice": region_summary(per_class, "dice"),
        "region_nsd": region_summary(per_class, "nsd"),
        "difficulty_dice": difficulty_summary(per_class, "dice"),
        "difficulty_nsd": difficulty_summary(per_class, "nsd"),
        "instance": evaluate_instances_dir(args.gt, args.pred, spacing=args.spacing),
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print("== region Dice =="); print(json.dumps(report["region_dice"], indent=2))
    print("== difficulty Dice =="); print(json.dumps(report["difficulty_dice"], indent=2))
    print("== instance =="); print(json.dumps(report["instance"], indent=2))


if __name__ == "__main__":
    main()
