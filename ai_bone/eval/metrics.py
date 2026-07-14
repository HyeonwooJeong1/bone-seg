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
        # one mask present, other absent → inf (intentional miss penalty;
        # propagates to the label's mean HD95 via nanmean)
        return float("inf")
    d = np.concatenate([_surface_dist(g, p, spacing), _surface_dist(p, g, spacing)])
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 95)) if d.size else float("inf")


def _surface(mask):
    return mask & ~_erode(mask)


def _surface_to_surface(gt, pred, label, spacing):
    """Symmetric surface-to-surface distances (gt→pred, pred→gt) in mm.
    Returns (d_g2p, d_p2g) arrays, or None if either mask is empty."""
    g = gt == label; p = pred == label
    if not g.any() or not p.any():
        return None
    sg, sp = _surface(g), _surface(p)
    d_g2p = distance_transform_edt(~sp, sampling=spacing)[sg]   # gt surf → pred surf
    d_p2g = distance_transform_edt(~sg, sampling=spacing)[sp]   # pred surf → gt surf
    return d_g2p, d_p2g


def nsd(gt, pred, label, spacing, tau=3.0):
    """Normalized Surface Dice @ tolerance tau (mm): fraction of surface points
    within tau of the other surface (TotalSegmentator/Skellytour standard, tau=3mm).
    both absent → nan; one absent → 0.0; identical → 1.0."""
    if not (gt == label).any() and not (pred == label).any():
        return float("nan")
    sd = _surface_to_surface(gt, pred, label, spacing)
    if sd is None:
        return 0.0
    d_g2p, d_p2g = sd
    num = int((d_g2p <= tau).sum() + (d_p2g <= tau).sum())
    den = d_g2p.size + d_p2g.size
    return float(num / den) if den else float("nan")


def assd(gt, pred, label, spacing):
    """Average Symmetric Surface Distance (mm). both absent → nan; one absent → inf."""
    if not (gt == label).any() and not (pred == label).any():
        return float("nan")
    sd = _surface_to_surface(gt, pred, label, spacing)
    if sd is None:
        return float("inf")
    d_g2p, d_p2g = sd
    return float(np.concatenate([d_g2p, d_p2g]).mean())


def cldice(gt, pred, label):
    """Centerline Dice (topology-aware) — suited to thin/elongated bones (ribs).
    Harmonic mean of topology precision (pred skeleton inside gt) and topology
    sensitivity (gt skeleton inside pred). both absent → nan; one absent → 0.0."""
    from skimage.morphology import skeletonize
    g = gt == label; p = pred == label
    if not g.any() and not p.any():
        return float("nan")
    if not g.any() or not p.any():
        return 0.0
    sk_g = skeletonize(g); sk_p = skeletonize(p)
    ng, npx = int(sk_g.sum()), int(sk_p.sum())
    tprec = float((sk_p & g).sum() / npx) if npx else 0.0
    tsens = float((sk_g & p).sum() / ng) if ng else 0.0
    return float(2 * tprec * tsens / (tprec + tsens)) if (tprec + tsens) > 0 else 0.0
