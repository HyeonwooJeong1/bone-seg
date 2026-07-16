import json
from dataclasses import dataclass, field
import numpy as np
from ai_bone import taxonomy_v1 as tx

@dataclass
class LabelMap:
    dataset: str
    source_format: str
    provenance_license: str
    value_to_name: dict          # int -> unified name
    grouped: dict = field(default_factory=dict)
    present_labels: list = field(default_factory=list)

    def validate(self):
        for name in list(self.value_to_name.values()) + list(self.present_labels):
            if name not in tx.NAME_TO_ID:
                raise ValueError(f"unknown unified label: {name!r} in {self.dataset}")
        for g in self.grouped.values():
            for name in g.get("covers", []):
                if name not in tx.NAME_TO_ID:
                    raise ValueError(f"unknown grouped label: {name!r}")
        if not set(self.present_labels) <= set(self.value_to_name.values()) \
           | {c for g in self.grouped.values() for c in g.get("covers", [])}:
            raise ValueError("present_labels not subset of mapped/grouped labels")

    def remap_array(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        lut = np.zeros(int(max([0, *self.value_to_name,
                                *[g["source_value"] for g in self.grouped.values()]])) + 1,
                       dtype=np.int32)
        for v, name in self.value_to_name.items():
            lut[int(v)] = tx.name_to_id(name)
        for g in self.grouped.values():
            lut[int(g["source_value"])] = tx.IGNORE_LABEL
        # Some source masks (e.g. VerSe dir-iso resamples) are stored float-typed;
        # cast the gather indices to int so LUT indexing works (label values are
        # integer-valued regardless of storage dtype).
        safe = np.where(arr < len(lut), arr, 0).astype(np.intp)   # 미정의 원본값 → 배경
        return lut[safe].astype(np.int32)

def load_label_map(path) -> LabelMap:
    d = json.loads(open(path, encoding="utf-8").read())
    lm = LabelMap(
        dataset=d["dataset"], source_format=d["source_format"],
        provenance_license=d["provenance_license"],
        value_to_name={int(k): v for k, v in d["map"].items()},
        grouped=d.get("grouped", {}), present_labels=d.get("present_labels", []))
    lm.validate()
    return lm
