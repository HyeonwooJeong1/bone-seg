import numpy as np, SimpleITK as sitk
from ai_bone.dedup import image_fingerprint, find_duplicates

def _img(seed):
    rng = np.random.default_rng(seed)
    return sitk.GetImageFromArray(rng.integers(-500,500,(16,16,16)).astype(np.int16))

def test_same_image_same_fp():
    a=_img(1); assert image_fingerprint(a)==image_fingerprint(a)

def test_find_duplicates_groups_identical():
    a=_img(1); b=_img(1); c=_img(999)
    groups = find_duplicates([("a",a),("b",b),("c",c)])
    dup = [g for g in groups if len(g)>1]
    assert dup and set(dup[0])=={"a","b"}
