import json
import os
from ai_bone.datasets.merge_raw import merge_raw


def _mk_source(root, cases, present):
    img = os.path.join(root, "imagesTr"); lab = os.path.join(root, "labelsTr")
    os.makedirs(img); os.makedirs(lab)
    for c in cases:
        open(os.path.join(img, f"{c}_0000.nii.gz"), "w").close()
        open(os.path.join(lab, f"{c}.nii.gz"), "w").close()
        with open(os.path.join(lab, f"{c}.present.json"), "w") as f:
            json.dump({"present_labels": present}, f)
    return root


def test_merge_raw_prefixes_and_aggregates(tmp_path):
    s1 = _mk_source(str(tmp_path / "A"), ["s0", "s1"], ["C1", "Sacrum"])
    s2 = _mk_source(str(tmp_path / "B"), ["c0"], ["Sacrum", "Hip_L"])
    out = str(tmp_path / "merged")
    res = merge_raw({"totalseg": s1, "ctpelvic1k": s2}, out, link=False, logf=lambda *a: None)

    assert res["total"] == 3
    assert res["per_dataset"] == {"totalseg": 2, "ctpelvic1k": 1}
    assert res["present_union"] == {"C1", "Sacrum", "Hip_L"}
    # prefixed, collision-free files
    assert os.path.exists(os.path.join(out, "imagesTr", "totalseg__s0_0000.nii.gz"))
    assert os.path.exists(os.path.join(out, "labelsTr", "ctpelvic1k__c0.nii.gz"))
    assert os.path.exists(os.path.join(out, "labelsTr", "totalseg__s1.present.json"))
    # dataset.json + case→dataset map
    dj = json.loads(open(os.path.join(out, "dataset.json")).read())
    assert dj["numTraining"] == 3 and dj["labels"]["ignore"] == 255
    c2d = json.loads(open(os.path.join(out, "case_datasets.json")).read())
    assert c2d["totalseg__s0"] == "totalseg" and c2d["ctpelvic1k__c0"] == "ctpelvic1k"
