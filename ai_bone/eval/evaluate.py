import glob, os
import numpy as np, SimpleITK as sitk
from ai_bone import taxonomy_v1 as tx
from ai_bone.eval.metrics import dice, hd95

def evaluate_dir(gt_dir, pred_dir, spacing=0.6):
    res = {n: {"dice": [], "hd95": []} for n in tx.FG_NAMES}
    for gp in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        cid = os.path.basename(gp)
        pp = os.path.join(pred_dir, cid)
        if not os.path.exists(pp): continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        for i, name in enumerate(tx.FG_NAMES, start=1):
            d = dice(gt, pr, i)
            if not np.isnan(d):
                res[name]["dice"].append(d)
                res[name]["hd95"].append(hd95(gt, pr, i, (spacing,)*3))
    out = {}
    for n, v in res.items():
        out[n] = {"dice": float(np.nanmean(v["dice"])) if v["dice"] else float("nan"),
                  "hd95": float(np.nanmean(v["hd95"])) if v["hd95"] else float("nan"),
                  "n": len(v["dice"])}
    return out
