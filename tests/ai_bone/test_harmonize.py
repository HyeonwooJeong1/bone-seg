import numpy as np, SimpleITK as sitk
from ai_bone.harmonize import harmonize_case
from ai_bone.label_map import LabelMap
from ai_bone import taxonomy_v1 as tx

def _img(arr, spacing=(1,1,1)):
    im = sitk.GetImageFromArray(arr); im.SetSpacing(spacing); return im

def test_harmonize_remaps_and_isotropic():
    ct = _img(np.zeros((8,8,8),np.int16), spacing=(0.8,0.8,0.6))
    seg = _img((np.arange(8*8*8).reshape(8,8,8) % 3).astype(np.uint8), spacing=(0.8,0.8,0.6))
    lm = LabelMap("t","nifti_seg","public", {1:"C1",2:"C2"}, {}, ["C1","C2"])
    out_ct, out_seg = harmonize_case(ct, seg, lm, spacing_mm=0.6)
    assert np.allclose(out_ct.GetSpacing(), (0.6,0.6,0.6))
    vals = set(np.unique(sitk.GetArrayFromImage(out_seg)).tolist())
    assert vals <= {0, tx.name_to_id("C1"), tx.name_to_id("C2")}
