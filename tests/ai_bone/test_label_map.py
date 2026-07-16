import json, numpy as np, pytest
from ai_bone.label_map import load_label_map, LabelMap
from ai_bone import taxonomy_v1 as tx

def _write(tmp_path, d):
    p = tmp_path / "label_map.json"; p.write_text(json.dumps(d)); return p

def test_remap_basic(tmp_path):
    p = _write(tmp_path, {
        "dataset":"verse","source_format":"nifti_seg","provenance_license":"public",
        "map":{"1":"C1","2":"C2"}, "grouped":{}, "present_labels":["C1","C2"]})
    lm = load_label_map(p)
    arr = np.array([0,1,2,1], dtype=np.uint8)
    out = lm.remap_array(arr)
    assert out.tolist() == [0, tx.name_to_id("C1"), tx.name_to_id("C2"), tx.name_to_id("C1")]

def test_remap_accepts_float_dtype_masks(tmp_path):
    # VerSe dir-iso resamples arrive as float; remap must cast indices to int.
    p = _write(tmp_path, {
        "dataset":"verse","source_format":"nifti_seg","provenance_license":"public",
        "map":{"1":"C1","2":"C2"}, "grouped":{}, "present_labels":["C1","C2"]})
    lm = load_label_map(p)
    out = lm.remap_array(np.array([0.0, 1.0, 2.0, 1.0], dtype=np.float32))
    assert out.tolist() == [0, tx.name_to_id("C1"), tx.name_to_id("C2"), tx.name_to_id("C1")]

def test_grouped_becomes_ignore(tmp_path):
    p = _write(tmp_path, {
        "dataset":"ctpelvic1k","source_format":"nifti_seg","provenance_license":"public",
        "map":{"2":"Sacrum"},
        "grouped":{"lumbar":{"source_value":1,"covers":["L1","L2","L3","L4","L5"]}},
        "present_labels":["Sacrum"]})
    lm = load_label_map(p)
    out = lm.remap_array(np.array([0,1,2], dtype=np.uint8))
    assert out.tolist() == [0, tx.IGNORE_LABEL, tx.name_to_id("Sacrum")]

def test_validate_rejects_unknown_label(tmp_path):
    p = _write(tmp_path, {"dataset":"x","source_format":"nifti_seg",
        "provenance_license":"public","map":{"1":"NotABone"},
        "grouped":{}, "present_labels":["NotABone"]})
    with pytest.raises(ValueError):
        load_label_map(p)

from ai_bone.datasets import registry

def test_all_registered_label_maps_load():
    assert {"totalseg","verse","ctspine1k","ribseg","ctpelvic1k",
            "spinemets","mug500","cads"} <= set(registry.DATASETS)
    for spec in registry.DATASETS.values():
        lm = load_label_map(spec.label_map_path)   # validate() 내부 호출
        assert lm.present_labels

def test_ribseg_covers_24_ribs():
    lm = load_label_map(registry.DATASETS["ribseg"].label_map_path)
    names = set(lm.value_to_name.values())
    for s in ("L","R"):
        for i in range(1,13):
            assert f"Rib_{s}_{i}" in names
