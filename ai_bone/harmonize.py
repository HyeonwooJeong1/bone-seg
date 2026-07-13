import numpy as np
import SimpleITK as sitk
from ai_bone.geometry import align_geometry, resample_to_isotropic
from ai_bone.label_map import LabelMap

def harmonize_case(ct_img, seg_img, lm: LabelMap, spacing_mm: float = 0.6):
    # 1) 원본 라벨값 → 통합 id (배열 연산, 메타 유지)
    arr = sitk.GetArrayFromImage(seg_img)
    remapped = lm.remap_array(arr).astype(np.uint8)   # 0..53, 255=ignore
    seg_u = sitk.GetImageFromArray(remapped)
    seg_u.CopyInformation(seg_img)
    # 2) CT-seg 정합
    seg_a = align_geometry(ct_img, seg_u)
    # 3) 등방 resample (CT=BSpline, label=Nearest)
    out_ct = resample_to_isotropic(ct_img, spacing_mm, is_label=False)
    out_seg = resample_to_isotropic(seg_a, spacing_mm, is_label=True)
    return out_ct, out_seg
