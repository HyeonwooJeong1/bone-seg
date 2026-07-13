import numpy as np, SimpleITK as sitk, pytest
from ai_bone.geometry import align_geometry, resample_to_isotropic

def _img(arr, spacing=(1,1,1), origin=(0,0,0), direction=None):
    im = sitk.GetImageFromArray(arr.astype(np.uint8) if arr.dtype==bool else arr)
    im.SetSpacing(spacing); im.SetOrigin(origin)
    if direction: im.SetDirection(direction)
    return im

def test_align_same_size_copies_info():
    ct = _img(np.zeros((4,4,4),np.int16), spacing=(2,2,2), origin=(5,6,7))
    seg = _img(np.ones((4,4,4),np.uint8))                 # 다른 origin/spacing
    out = align_geometry(ct, seg)
    assert out.GetSize()==ct.GetSize()
    assert np.allclose(out.GetOrigin(), ct.GetOrigin())
    assert np.array_equal(
        sitk.GetArrayFromImage(out), np.ones((4,4,4),np.uint8))  # 배열 보존

def test_align_diff_size_resamples_to_ct():
    ct = _img(np.zeros((8,8,8),np.int16), spacing=(1,1,1))
    seg = _img(np.ones((4,4,4),np.uint8), spacing=(2,2,2))   # 물리적으로 동일 FOV
    out = align_geometry(ct, seg)
    assert out.GetSize()==ct.GetSize()

def test_isotropic_spacing():
    im = _img(np.zeros((10,10,20),np.int16), spacing=(0.8,0.8,0.6))
    out = resample_to_isotropic(im, 0.6, is_label=False)
    assert np.allclose(out.GetSpacing(), (0.6,0.6,0.6))
