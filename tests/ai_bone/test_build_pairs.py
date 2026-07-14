import json, os
import numpy as np
import SimpleITK as sitk
from ai_bone.build_raw import build_from_pairs
from ai_bone.label_map import LabelMap

# NOTE: SimpleITK's disk IO fails on non-ASCII paths (the local temp dir carries a
# non-ASCII username); on the server (/data1/bone, ASCII) it is fine. So this test
# injects in-memory reader/writer to exercise the orchestration logic without
# SimpleITK file IO. JSON sidecars use Python open(), which handles unicode paths.

def _img(aligned):
    ct = np.zeros((8, 8, 8), np.int16)
    seg = np.ones((8, 8, 8), np.uint8)          # whole volume = label 1
    ct[:] = 300 if aligned else 0               # bone HU present only if aligned
    return sitk.GetImageFromArray(ct), sitk.GetImageFromArray(seg)

def test_build_from_pairs_writes_good_skips_bad(tmp_path):
    good_ct, good_seg = _img(True)
    bad_ct, bad_seg = _img(False)
    store = {"good_ct": good_ct, "good_seg": good_seg, "bad_ct": bad_ct, "bad_seg": bad_seg}
    written = []
    pairs = [("good_ct", "good_seg", "good"), ("bad_ct", "bad_seg", "bad")]
    lm = LabelMap("t", "nifti_seg", "public", {1: "C1"}, {}, ["C1"])
    out = tmp_path / "Dataset999_Test"

    res = build_from_pairs(
        pairs, lm, str(out),
        reader=lambda p: store[p],
        writer=lambda img, p: written.append(os.path.basename(p)),
        logf=lambda *a: None,
    )

    assert res["written"] == 1
    assert [cid for cid, _ in res["skipped"]] == ["bad"]
    assert "good_0000.nii.gz" in written        # image written
    assert "good.nii.gz" in written             # label written
    assert "bad_0000.nii.gz" not in written     # skipped case not written
    assert (out / "labelsTr" / "good.present.json").exists()

    dj = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert dj["numTraining"] == 1
    assert dj["labels"]["ignore"] == 255

def test_ribseg_pairs_matches_by_case_token(tmp_path):
    from ai_bone.datasets.make_pairs import ribseg_pairs
    ct_root = tmp_path / "ribfrac_img"; ct_root.mkdir()
    seg_dir = tmp_path / "seg"; seg_dir.mkdir()
    # matched pair + a CT with no seg + a seg with no CT → only the pair survives
    (ct_root / "RibFrac1-image.nii.gz").write_bytes(b"")
    (ct_root / "RibFrac2-image.nii.gz").write_bytes(b"")          # no seg
    (seg_dir / "RibFrac1-rib-seg.nii.gz").write_bytes(b"")
    (seg_dir / "RibFrac9-rib-seg.nii.gz").write_bytes(b"")        # no ct
    pairs = ribseg_pairs(str(ct_root), str(seg_dir))
    assert len(pairs) == 1
    ct, seg, cid = pairs[0]
    assert cid == "RibFrac1"
    assert ct.endswith("RibFrac1-image.nii.gz")
    assert seg.endswith("RibFrac1-rib-seg.nii.gz")

def test_workers_gt1_with_injected_io_stays_sequential(tmp_path):
    # Injected reader/writer can't be pickled to a Pool, so workers>1 must fall
    # back to the sequential path (no crash, same result).
    good_ct, good_seg = _img(True)
    store = {"c": good_ct, "s": good_seg}
    written = []
    lm = LabelMap("t", "nifti_seg", "public", {1: "C1"}, {}, ["C1"])
    res = build_from_pairs(
        [("c", "s", "good")], lm, str(tmp_path / "D"),
        reader=lambda p: store[p],
        writer=lambda img, p: written.append(os.path.basename(p)),
        logf=lambda *a: None, workers=4,
    )
    assert res["written"] == 1
    assert "good_0000.nii.gz" in written
