import json
import numpy as np
from ai_bone import taxonomy_v1 as tx
from ai_bone.datasets.combine import combine_arrays, TS_NAME_TO_UNIFIED
from ai_bone.datasets.make_pairs import match_by_token, write_pairs

def test_ts_name_map_covers_axial_bones():
    # every TotalSeg bone name maps to a valid unified label
    for src, uni in TS_NAME_TO_UNIFIED.items():
        assert uni in tx.NAME_TO_ID
    assert TS_NAME_TO_UNIFIED["vertebrae_C1"] == "C1"
    assert TS_NAME_TO_UNIFIED["rib_left_1"] == "Rib_L_1"
    assert TS_NAME_TO_UNIFIED["hip_right"] == "Hip_R"
    assert TS_NAME_TO_UNIFIED["sacrum"] == "Sacrum"

def test_combine_arrays_assigns_unified_ids():
    a = np.zeros((6, 6, 6), bool); a[1, 1, 1] = True     # C1
    b = np.zeros((6, 6, 6), bool); b[4, 4, 4] = True     # Rib_L_3
    out = combine_arrays({"vertebrae_C1": a, "rib_left_3": b})
    assert out[1, 1, 1] == tx.NAME_TO_ID["C1"]
    assert out[4, 4, 4] == tx.NAME_TO_ID["Rib_L_3"]
    assert out.max() == max(tx.NAME_TO_ID["C1"], tx.NAME_TO_ID["Rib_L_3"])

def test_combine_ignores_unmapped_names():
    a = np.ones((3, 3, 3), bool)
    out = combine_arrays({"aorta": a})     # not a bone → nothing mapped
    assert out is None
    # mixed: unmapped ignored, mapped kept
    c1 = np.zeros((3, 3, 3), bool); c1[0, 0, 0] = True
    out2 = combine_arrays({"aorta": a, "vertebrae_C1": c1})
    assert out2[0, 0, 0] == tx.NAME_TO_ID["C1"] and int((out2 > 0).sum()) == 1

def test_match_by_token_pairs_ct_and_seg():
    ct = ["/x/dataset6_CLINIC_0060_data.nii.gz", "/x/dataset6_CLINIC_0061_data.nii.gz"]
    seg = ["/y/dataset6_CLINIC_0060_mask_4label.nii.gz"]   # only 0060 has a mask
    pairs = match_by_token(ct, seg, ct_strip="_data", seg_strip="_mask_4label")
    assert len(pairs) == 1
    assert pairs[0][2] == "dataset6_CLINIC_0060"
    assert pairs[0][0].endswith("0060_data.nii.gz") and pairs[0][1].endswith("0060_mask_4label.nii.gz")

def test_write_pairs_roundtrip(tmp_path):
    pairs = [("a_ct.nii.gz", "a_seg.nii.gz", "a")]
    p = write_pairs(pairs, str(tmp_path / "pairs.json"))
    data = json.loads(open(p).read())
    assert data == [["a_ct.nii.gz", "a_seg.nii.gz", "a"]]
