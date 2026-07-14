"""Tolerant NIfTI reader.

TotalSegmentator (and some other) volumes store direction cosines that are only
approximately orthonormal (float error ~1e-4). ITK/SimpleITK reject these with
"ITK only supports orthonormal direction cosines". `read_sitk` reads normally with
SimpleITK and, on that failure, falls back to nibabel and rebuilds a SimpleITK image
with the direction snapped to the nearest orthonormal matrix (polar decomposition) —
a negligible geometric change, and consistent across a case's CT + masks.
"""
import numpy as np


def _nib_to_sitk(nii):
    """nibabel image → SimpleITK image with an orthonormalized (RAS→LPS) direction."""
    import SimpleITK as sitk
    arr = np.asanyarray(nii.dataobj)
    aff = np.asarray(nii.affine, dtype=float)
    R = aff[:3, :3]
    sp = np.linalg.norm(R, axis=0)
    sp[sp == 0] = 1.0
    D = R / sp
    U, _, Vt = np.linalg.svd(D)                 # nearest orthonormal (polar)
    D_ortho = U @ Vt
    lps = np.diag([-1.0, -1.0, 1.0])            # nibabel RAS → ITK LPS
    d_lps = lps @ D_ortho
    origin = lps @ aff[:3, 3]
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
    img.SetSpacing([float(x) for x in sp])
    img.SetDirection([float(x) for x in d_lps.flatten()])
    img.SetOrigin([float(x) for x in origin])
    return img


def read_sitk(path):
    """SimpleITK read with a nibabel-orthonormalize fallback for TotalSeg-style files."""
    import SimpleITK as sitk
    try:
        return sitk.ReadImage(path)
    except Exception:
        import nibabel as nib
        return _nib_to_sitk(nib.load(path))
