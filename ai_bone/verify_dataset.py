import numpy as np
import SimpleITK as sitk

def verify_case(ct: sitk.Image, seg: sitk.Image, hu_thr: int = 200) -> dict:
    hu = sitk.GetArrayFromImage(ct)
    lab = sitk.GetArrayFromImage(seg)
    fg = lab > 0
    n = int(fg.sum())
    overlap = float(((fg) & (hu >= hu_thr)).sum()) / n if n else 0.0
    return {
        "empty": n == 0,
        "size_match": tuple(ct.GetSize()) == tuple(seg.GetSize()),
        "overlap_ratio": overlap,
        "labels": set(np.unique(lab).tolist()),
    }

def is_pass(report: dict) -> bool:
    return (not report["empty"]) and report["size_match"] and report["overlap_ratio"] >= 0.5
