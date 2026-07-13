import numpy as np
import SimpleITK as sitk

def align_geometry(ct: sitk.Image, seg: sitk.Image) -> sitk.Image:
    """CT-seg 정합. size 동일=배열그대로 CT 메타 복사, 다름=물리 Nearest resample."""
    if tuple(seg.GetSize()) == tuple(ct.GetSize()):
        out = sitk.Cast(seg, seg.GetPixelID())
        out.CopyInformation(ct)          # 배열 유지, 메타만 CT로
        return out
    return sitk.Resample(seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, seg.GetPixelID())

def resample_to_isotropic(img: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image:
    in_sp = np.array(img.GetSpacing(), float)
    in_sz = np.array(img.GetSize(), int)
    out_sp = np.array([spacing_mm]*3, float)
    out_sz = np.round(in_sz * in_sp / out_sp).astype(int).tolist()
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    return sitk.Resample(img, out_sz, sitk.Transform(), interp, img.GetOrigin(),
                         out_sp.tolist(), img.GetDirection(), 0, img.GetPixelID())
