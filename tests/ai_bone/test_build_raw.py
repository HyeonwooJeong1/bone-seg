import json
from ai_bone.build_raw import write_dataset_json, write_present_sidecar
from ai_bone import taxonomy_v1 as tx

def test_dataset_json_has_all_labels_and_ignore(tmp_path):
    d = write_dataset_json(tmp_path, num_training=3, present_union=set(tx.FG_NAMES))
    assert d["labels"]["background"] == 0
    assert d["labels"]["Hip_R"] == 53
    assert d["labels"]["ignore"] == 255
    assert d["numTraining"] == 3
    assert (tmp_path/"dataset.json").exists()

def test_present_sidecar(tmp_path):
    p = write_present_sidecar(tmp_path, "case_0001", ["C1","C2"])
    assert json.loads(open(p).read())["present_labels"] == ["C1","C2"]
