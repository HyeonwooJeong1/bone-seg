import os
from dataclasses import dataclass

_HERE = os.path.dirname(__file__)

@dataclass
class DatasetSpec:
    name: str
    stage: str            # "pretrain" | "ft"
    download_kind: str    # "zenodo" | "tcia" | "github" | "huggingface" | "figshare"
    tag: str              # 부위/파일 매칭 태그(없으면 "")

    @property
    def label_map_path(self): return os.path.join(_HERE, self.name, "label_map.json")

DATASETS = {s.name: s for s in [
    DatasetSpec("cads",      "pretrain", "huggingface", ""),
    DatasetSpec("totalseg",  "ft",       "zenodo",      ""),
    DatasetSpec("verse",     "ft",       "github",      ""),
    DatasetSpec("ctspine1k", "ft",       "github",      ""),
    DatasetSpec("ribseg",    "ft",       "zenodo",      ""),
    DatasetSpec("ctpelvic1k","ft",       "zenodo",      ""),
    DatasetSpec("spinemets", "ft",       "tcia",        ""),
    DatasetSpec("mug500",    "ft",       "figshare",    ""),
]}

def iter_ft():  return [s for s in DATASETS.values() if s.stage == "ft"]
def iter_all(): return list(DATASETS.values())
