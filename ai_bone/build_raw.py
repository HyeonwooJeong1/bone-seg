import json, os
from ai_bone import taxonomy_v1 as tx

def write_dataset_json(out_dir, num_training: int, present_union) -> dict:
    labels = {name: i for i, name in tx.UNIFIED_V1.items()}   # background..Hip_R
    labels["ignore"] = tx.IGNORE_LABEL
    d = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(d, f, indent=2)
    return d

def write_present_sidecar(out_dir, case_id: str, present_labels) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{case_id}.present.json")
    with open(p, "w") as f:
        json.dump({"present_labels": list(present_labels)}, f)
    return p
