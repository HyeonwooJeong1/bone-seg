import numpy as np
from scipy.ndimage import distance_transform_edt

def dice(gt, pred, label):
    g = gt == label; p = pred == label
    denom = g.sum() + p.sum()
    if denom == 0:
        return float("nan")            # 라벨이 GT·pred 모두 없음
    return 2.0 * (g & p).sum() / denom

def _surface_dist(a, b, spacing):
    # a 표면에서 b 표면까지 거리
    if not a.any() or not b.any():
        return np.array([np.inf])
    dt = distance_transform_edt(~b, sampling=spacing)
    border = a & ~_erode(a)
    return dt[border]

def _erode(m):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m)

def hd95(gt, pred, label, spacing):
    g = gt == label; p = pred == label
    if not g.any() and not p.any():
        return float("nan")
    if not g.any() or not p.any():
        return float("inf")
    d = np.concatenate([_surface_dist(g, p, spacing), _surface_dist(p, g, spacing)])
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 95)) if d.size else float("inf")
