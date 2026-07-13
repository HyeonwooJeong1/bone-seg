import numpy as np, SimpleITK as sitk
from ai_bone.verify_dataset import verify_case, is_pass

def _pair(seg_arr, hu_arr):
    return (sitk.GetImageFromArray(hu_arr.astype(np.int16)),
            sitk.GetImageFromArray(seg_arr.astype(np.uint8)))

def test_good_case_passes():
    seg=np.zeros((6,6,6)); seg[2:4,2:4,2:4]=5
    hu=np.zeros((6,6,6)); hu[2:4,2:4,2:4]=300
    ct,sg=_pair(seg,hu); r=verify_case(ct,sg); assert is_pass(r) and r["overlap_ratio"]>0.9

def test_empty_label_fails():
    ct,sg=_pair(np.zeros((6,6,6)), np.zeros((6,6,6)))
    assert not is_pass(verify_case(ct,sg))

def test_misaligned_low_overlap_fails():
    seg=np.zeros((6,6,6)); seg[0:2,0:2,0:2]=5
    hu=np.zeros((6,6,6)); hu[4:6,4:6,4:6]=300   # 라벨과 뼈가 딴 곳
    ct,sg=_pair(seg,hu); assert not is_pass(verify_case(ct,sg))
