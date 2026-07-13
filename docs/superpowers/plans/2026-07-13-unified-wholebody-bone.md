# 전신 뼈 통합 분할 모델 v1 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 부위별 공개 CT 뼈 데이터를 통합 taxonomy로 합쳐 축골격+골반대 nnU-Net을 학습하는 파이프라인·MERIT·평가 코드를 완성한다(실행은 사용자가 나중에 공유 H100에서).

**Architecture:** 순수 파이썬 데이터 파이프라인(taxonomy→label_map→harmonize→dedup→build_raw→verify) + nnU-Net v2 학습 스크립트(3-stage) + MERIT(충돌측정·PCA분할·가중병합) + 평가. 모든 로직 모듈은 로컬 `ct_env`에서 GPU 없이 pytest로 검증. GPU가 필요한 실행은 코드가 아닌 `ai_bone/runbook.md` 명령으로 분리.

**Tech Stack:** Python 3.10(ct_env, 로컬 테스트) / nnU-Net v2 2.8.1(서버) / SimpleITK · numpy · scipy · pytest. 서버 conda pt210_py312.

## Global Constraints

- 프레임워크: **nnU-Net v2** (서버 v2.8.1, conda pt210_py312). 로컬 테스트는 ct_env(GPU 불필요 모듈만).
- 목표 spacing: **등방 0.6mm**, plans 이름 **`nnUNetPlans_iso06`**.
- trainer: **`nnUNetTrainerNoMirroring_ES`** (좌우 라벨 → NoMirroring + EMA dice early stopping).
- 라벨 정합: CT-seg **size 같으면 `seg.CopyInformation(ct)`, 다르면 물리 resample(NearestNeighbor)**.
- 좌우 기준: **LPS**. IGNORE 라벨 값 = **255**. 배경=0, 전경 53클래스(총 54채널).
- 라이선스: **비상업**. CADS=사전학습 전용. provenance 태깅.
- 서버: 작업루트 `/data1/bone`, 전처리 캐시 **로컬 SSD `/home/ubuntu/nnunet_pre`**. 실행 시 `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH nnUNet_compile=f` 필수.
- ★ **이번 작업 = 코드만**. 각 학습 스크립트는 **GPU 1장** 사용(인자로 받음), 여러 fold는 순차/가용 GPU에 개별 실행, checkpoint 재개.
- 로컬 테스트 실행: `"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest`.
- 커밋 규약: 각 태스크 끝 커밋, 메시지 끝에 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## 파일 구조

```
ai_bone/
  taxonomy_v1.py          # Task1  통합 라벨 상수+역맵+검증
  label_map.py            # Task2  label_map.json 로더/검증 + remap_array
  datasets/
    _schema.md            # Task2  label_map.json 스키마 설명
    totalseg/label_map.json   verse/label_map.json   ctspine1k/label_map.json   # Task3
    ribseg/label_map.json     ctpelvic1k/label_map.json spinemets/label_map.json
    mug500/label_map.json     cads/label_map.json
    registry.py           # Task3  데이터셋 등록부(경로·다운로드 방식·tag)
  geometry.py             # Task4  align_geometry + resample_to_isotropic
  harmonize.py            # Task5  harmonize_case (remap+정합+iso resample)
  dedup.py                # Task6  image_fingerprint + find_duplicates
  build_raw.py            # Task7  nnUNet_raw 조립 + present_labels 사이드카 + dataset.json
  verify_dataset.py       # Task8  빈라벨/geometry/겹침률 검증
  download.py             # Task9  범용 재개 다운로더 + per-dataset 훅
  merit/
    split.py              # Task11 pca_conflict_split (pure numpy)
    merge.py              # Task12 weighted_average + ties_merge (state_dict)
    estimate_conflict.py  # Task13 (서버) gradient 추출 → g_d 저장
  eval/
    metrics.py            # Task10 dice + hd95 (pure)
    evaluate.py           # Task10 예측/GT 폴더 비교 리포트
  train/
    partial_label_trainer.py  # Task14 ignore-label + balanced sampling trainer
    merit_finetune_trainer.py # Task14 저LR 파티션 fine-tune trainer
    stage1_pretrain.sh        # Task15
    stage2_baseline.sh        # Task15
    merit_train_partition.sh  # Task15
  runbook.md              # Task16 (서버 실행 명령 전체)
tests/ai_bone/
  test_taxonomy.py test_label_map.py test_geometry.py test_harmonize.py
  test_dedup.py test_build_raw.py test_verify.py test_download.py
  test_split.py test_merge.py test_metrics.py test_trainers_import.py
conftest.py  (pytest 경로)
```

---

## Phase 1 — 데이터 파이프라인 (순수 파이썬, 로컬 테스트)

### Task 1: 통합 Taxonomy 모듈

**Files:**
- Create: `ai_bone/taxonomy_v1.py`
- Test: `tests/ai_bone/test_taxonomy.py`

**Interfaces:**
- Produces: `UNIFIED_V1: dict[int,str]`(0=background..53=Hip_R), `NAME_TO_ID: dict[str,int]`, `NUM_CLASSES=54`, `IGNORE_LABEL=255`, `FG_NAMES: list[str]`, `name_to_id(name)->int`, `id_to_name(i)->str`, `validate()->None`.

- [ ] **Step 1: 실패 테스트 작성** — `tests/ai_bone/test_taxonomy.py`

```python
from ai_bone import taxonomy_v1 as tx

def test_counts():
    assert tx.NUM_CLASSES == 54          # bg + 53 fg
    assert len(tx.FG_NAMES) == 53
    assert tx.IGNORE_LABEL == 255

def test_roundtrip_unique():
    # id<->name 왕복 + 중복 없음
    for i, name in tx.UNIFIED_V1.items():
        assert tx.name_to_id(name) == i
        assert tx.id_to_name(i) == name
    assert len(set(tx.UNIFIED_V1.values())) == len(tx.UNIFIED_V1)

def test_key_labels_present():
    for n in ["Skull","C1","C7","T1","T12","L1","L5","Sacrum",
              "Rib_L_1","Rib_R_12","Sternum","Hip_L","Hip_R"]:
        assert n in tx.NAME_TO_ID

def test_validate_ok():
    tx.validate()  # raises on any inconsistency
```

- [ ] **Step 2: 실패 확인**

Run: `"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_taxonomy.py -q`
Expected: FAIL (ModuleNotFoundError: ai_bone.taxonomy_v1)

- [ ] **Step 3: 구현** — `ai_bone/taxonomy_v1.py`

```python
"""통합 전신 뼈 taxonomy v1 (축골격 + 골반대). 배경=0, 전경 53클래스."""

IGNORE_LABEL = 255

def _build():
    d = {0: "background", 1: "Skull"}
    nid = 2
    for i in range(1, 8):   d[nid] = f"C{i}";  nid += 1      # 2..8
    for i in range(1, 13):  d[nid] = f"T{i}";  nid += 1      # 9..20
    for i in range(1, 6):   d[nid] = f"L{i}";  nid += 1      # 21..25
    d[nid] = "Sacrum"; nid += 1                              # 26
    for i in range(1, 13):  d[nid] = f"Rib_L_{i}"; nid += 1  # 27..38
    for i in range(1, 13):  d[nid] = f"Rib_R_{i}"; nid += 1  # 39..50
    d[nid] = "Sternum"; nid += 1                             # 51
    d[nid] = "Hip_L"; nid += 1                               # 52
    d[nid] = "Hip_R"; nid += 1                               # 53
    return d

UNIFIED_V1 = _build()
NAME_TO_ID = {v: k for k, v in UNIFIED_V1.items()}
NUM_CLASSES = len(UNIFIED_V1)          # 54
FG_NAMES = [UNIFIED_V1[i] for i in range(1, NUM_CLASSES)]

def name_to_id(name): return NAME_TO_ID[name]
def id_to_name(i): return UNIFIED_V1[i]

def validate():
    assert NUM_CLASSES == 54, NUM_CLASSES
    assert len(set(UNIFIED_V1.values())) == NUM_CLASSES, "duplicate label name"
    assert set(UNIFIED_V1) == set(range(NUM_CLASSES)), "ids must be contiguous 0..53"
    assert IGNORE_LABEL not in UNIFIED_V1
```

- [ ] **Step 4: 통과 확인** — Run 위 pytest. Expected: PASS (4 passed)
- [ ] **Step 5: 커밋**

```bash
git add ai_bone/taxonomy_v1.py tests/ai_bone/test_taxonomy.py
git commit -m "feat(bone-v1): unified axial taxonomy (54-class)"
```

---

### Task 2: label_map 로더/검증 + remap

**Files:**
- Create: `ai_bone/label_map.py`, `ai_bone/datasets/_schema.md`
- Test: `tests/ai_bone/test_label_map.py`

**Interfaces:**
- Consumes: `taxonomy_v1` (NAME_TO_ID, IGNORE_LABEL).
- Produces: `LabelMap` dataclass(fields: `dataset:str`, `source_format:str`, `provenance_license:str`, `value_to_name:dict[int,str]`, `grouped:dict`, `present_labels:list[str]`); `load_label_map(path)->LabelMap`; `LabelMap.remap_array(arr:np.ndarray)->np.ndarray` (원본 정수→통합 id, grouped source_value→IGNORE); `LabelMap.validate()`.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_label_map.py`

```python
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
```

- [ ] **Step 2: 실패 확인** — Run: `... -m pytest tests/ai_bone/test_label_map.py -q` → FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/label_map.py`

```python
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
        safe = np.where(arr < len(lut), arr, 0)   # 미정의 원본값 → 배경
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
```

- [ ] **Step 4: `ai_bone/datasets/_schema.md` 작성** (스펙 §4.2 스키마 복사 — map/grouped/present_labels/provenance_license 설명).
- [ ] **Step 5: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_label_map.py -q
git add ai_bone/label_map.py ai_bone/datasets/_schema.md tests/ai_bone/test_label_map.py
git commit -m "feat(bone-v1): label_map loader with grouped->ignore remap"
```

---

### Task 3: 데이터셋 등록부 + 8개 label_map.json

**Files:**
- Create: `ai_bone/datasets/registry.py`, `ai_bone/datasets/<name>/label_map.json` × 8
- Test: `tests/ai_bone/test_label_map.py` (확장: 실제 json 로드)

**Interfaces:**
- Produces: `registry.DATASETS: dict[name, DatasetSpec]` (fields: `name`, `stage`("pretrain"|"ft"), `label_map_path`, `download_kind`, `tag`), `registry.iter_ft()`, `registry.iter_all()`.

- [ ] **Step 1: 실패 테스트 추가** — `tests/ai_bone/test_label_map.py`에 append

```python
from ai_bone.datasets import registry
from ai_bone.label_map import load_label_map

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
```

- [ ] **Step 2: 실패 확인** — FAIL(import registry)

- [ ] **Step 3: `ai_bone/datasets/registry.py` 구현**

```python
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
    DatasetSpec("cads",      "pretrain","huggingface",""),
    DatasetSpec("totalseg",  "ft","zenodo",""),
    DatasetSpec("verse",     "ft","github",""),
    DatasetSpec("ctspine1k", "ft","github",""),
    DatasetSpec("ribseg",    "ft","zenodo",""),
    DatasetSpec("ctpelvic1k","ft","zenodo",""),
    DatasetSpec("spinemets", "ft","tcia",""),
    DatasetSpec("mug500",    "ft","figshare",""),
]}

def iter_ft():  return [s for s in DATASETS.values() if s.stage == "ft"]
def iter_all(): return list(DATASETS.values())
```

- [ ] **Step 4: 8개 label_map.json 작성.** 각 파일 예시(전체는 실제 데이터셋 라벨값 확인 후 계획 실행자가 채우되, 아래 값은 공개 문서 기준 초안):

`ai_bone/datasets/totalseg/label_map.json` (TotalSegmentator v2 뼈 클래스명 → 통합명; 앵커라 v1 전 클래스 커버):
```json
{"dataset":"totalseg","source_format":"nifti_seg","provenance_license":"ccby",
 "map":{"1":"Skull","2":"C1","3":"C2","4":"C3","5":"C4","6":"C5","7":"C6","8":"C7",
        "9":"T1","10":"T2","11":"T3","12":"T4","13":"T5","14":"T6","15":"T7","16":"T8",
        "17":"T9","18":"T10","19":"T11","20":"T12","21":"L1","22":"L2","23":"L3","24":"L4",
        "25":"L5","26":"Sacrum","27":"Sternum","28":"Hip_L","29":"Hip_R",
        "30":"Rib_L_1","31":"Rib_L_2","32":"Rib_L_3","33":"Rib_L_4","34":"Rib_L_5","35":"Rib_L_6",
        "36":"Rib_L_7","37":"Rib_L_8","38":"Rib_L_9","39":"Rib_L_10","40":"Rib_L_11","41":"Rib_L_12",
        "42":"Rib_R_1","43":"Rib_R_2","44":"Rib_R_3","45":"Rib_R_4","46":"Rib_R_5","47":"Rib_R_6",
        "48":"Rib_R_7","49":"Rib_R_8","50":"Rib_R_9","51":"Rib_R_10","52":"Rib_R_11","53":"Rib_R_12"},
 "grouped":{},
 "present_labels":["Skull","C1","C2","C3","C4","C5","C6","C7","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5","Sacrum","Sternum","Hip_L","Hip_R","Rib_L_1","Rib_L_2","Rib_L_3","Rib_L_4","Rib_L_5","Rib_L_6","Rib_L_7","Rib_L_8","Rib_L_9","Rib_L_10","Rib_L_11","Rib_L_12","Rib_R_1","Rib_R_2","Rib_R_3","Rib_R_4","Rib_R_5","Rib_R_6","Rib_R_7","Rib_R_8","Rib_R_9","Rib_R_10","Rib_R_11","Rib_R_12"],
 "notes":"원본 라벨값은 TotalSegmentator v2 총 라벨에서 뼈만 추출·재번호. 실제 값은 다운로드 후 확정."}
```
`verse/label_map.json`(척추만; VerSe 라벨규칙 1..25가 C1..L5+천추 → 통합명):
```json
{"dataset":"verse","source_format":"nifti_seg","provenance_license":"public",
 "map":{"1":"C1","2":"C2","3":"C3","4":"C4","5":"C5","6":"C6","7":"C7","8":"T1","9":"T2","10":"T3","11":"T4","12":"T5","13":"T6","14":"T7","15":"T8","16":"T9","17":"T10","18":"T11","19":"T12","20":"L1","21":"L2","22":"L3","23":"L4","24":"L5","26":"Sacrum"},
 "grouped":{},
 "present_labels":["C1","C2","C3","C4","C5","C6","C7","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","L1","L2","L3","L4","L5","Sacrum"],
 "notes":"VerSe 라벨 25=T13(변이), 27/28=천추 변이는 실제 데이터 확인 후 추가."}
```
`ctspine1k/label_map.json`: verse와 동일 척추 매핑(1..24=C1..L5), present_labels 척추.
`ribseg/label_map.json`(늑골 24; RibSeg 라벨값 1..12=우, 13..24=좌 규칙은 실제 확인 후 좌우 확정):
```json
{"dataset":"ribseg","source_format":"nifti_seg","provenance_license":"public",
 "map":{"1":"Rib_L_1","2":"Rib_L_2","3":"Rib_L_3","4":"Rib_L_4","5":"Rib_L_5","6":"Rib_L_6","7":"Rib_L_7","8":"Rib_L_8","9":"Rib_L_9","10":"Rib_L_10","11":"Rib_L_11","12":"Rib_L_12","13":"Rib_R_1","14":"Rib_R_2","15":"Rib_R_3","16":"Rib_R_4","17":"Rib_R_5","18":"Rib_R_6","19":"Rib_R_7","20":"Rib_R_8","21":"Rib_R_9","22":"Rib_R_10","23":"Rib_R_11","24":"Rib_R_12"},
 "grouped":{},
 "present_labels":["Rib_L_1","Rib_L_2","Rib_L_3","Rib_L_4","Rib_L_5","Rib_L_6","Rib_L_7","Rib_L_8","Rib_L_9","Rib_L_10","Rib_L_11","Rib_L_12","Rib_R_1","Rib_R_2","Rib_R_3","Rib_R_4","Rib_R_5","Rib_R_6","Rib_R_7","Rib_R_8","Rib_R_9","Rib_R_10","Rib_R_11","Rib_R_12"],
 "notes":"좌우 규칙은 다운로드 후 verify_dataset의 x-중심으로 검증·정정."}
```
`ctpelvic1k/label_map.json`(천골·좌우 관골 개별, 요추는 grouped→ignore):
```json
{"dataset":"ctpelvic1k","source_format":"nifti_seg","provenance_license":"public",
 "map":{"1":"Sacrum","3":"Hip_L","4":"Hip_R"},
 "grouped":{"lumbar":{"source_value":2,"covers":["L1","L2","L3","L4","L5"]}},
 "present_labels":["Sacrum","Hip_L","Hip_R"],
 "notes":"원본 4class: 1=천골,2=요추(덩어리),3/4=좌우 관골. 좌우는 verify로 검증."}
```
`spinemets/label_map.json`: 척추 개별(verse 규칙), source_format `dicom_seg`.
`mug500/label_map.json`: `{"1":"Skull"}`, present ["Skull"], source_format `nifti_seg`.
`cads/label_map.json`(pretrain; 축골격 라벨만 추출): totalseg와 동일 통합명으로 CADS 라벨값 매핑, provenance `ccbync`.

- [ ] **Step 5: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_label_map.py -q
git add ai_bone/datasets tests/ai_bone/test_label_map.py
git commit -m "feat(bone-v1): dataset registry + 8 label_map.json drafts"
```

---

### Task 4: geometry 정합 + 등방 resample

**Files:**
- Create: `ai_bone/geometry.py`
- Test: `tests/ai_bone/test_geometry.py`

**Interfaces:**
- Produces: `align_geometry(ct: sitk.Image, seg: sitk.Image) -> sitk.Image` (size 같으면 CopyInformation, 다르면 Nearest resample to ct); `resample_to_isotropic(img: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image`.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_geometry.py`

```python
import numpy as np, SimpleITK as sitk, pytest
from ai_bone.geometry import align_geometry, resample_to_isotropic

def _img(arr, spacing=(1,1,1), origin=(0,0,0), direction=None):
    im = sitk.GetImageFromArray(arr.astype(np.uint8) if arr.dtype==bool else arr)
    im.SetSpacing(spacing); im.SetOrigin(origin)
    if direction: im.SetDirection(direction)
    return im

def test_align_same_size_copies_info():
    ct = _img(np.zeros((4,4,4),np.int16), spacing=(2,2,2), origin=(5,6,7))
    seg = _img(np.ones((4,4,4),np.uint8))                 # 다른 origin/spacing
    out = align_geometry(ct, seg)
    assert out.GetSize()==ct.GetSize()
    assert np.allclose(out.GetOrigin(), ct.GetOrigin())
    assert sitk.GetArrayFromArray if False else np.array_equal(
        sitk.GetArrayFromImage(out), np.ones((4,4,4),np.uint8))  # 배열 보존

def test_align_diff_size_resamples_to_ct():
    ct = _img(np.zeros((8,8,8),np.int16), spacing=(1,1,1))
    seg = _img(np.ones((4,4,4),np.uint8), spacing=(2,2,2))   # 물리적으로 동일 FOV
    out = align_geometry(ct, seg)
    assert out.GetSize()==ct.GetSize()

def test_isotropic_spacing():
    im = _img(np.zeros((10,10,20),np.int16), spacing=(0.8,0.8,0.6))
    out = resample_to_isotropic(im, 0.6, is_label=False)
    assert np.allclose(out.GetSpacing(), (0.6,0.6,0.6))
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/geometry.py`

```python
import numpy as np
import SimpleITK as sitk

def align_geometry(ct: sitk.Image, seg: sitk.Image) -> sitk.Image:
    """CT-seg 정합. size 동일=배열그대로 CT 메타 복사, 다름=물리 Nearest resample."""
    if tuple(seg.GetSize()) == tuple(ct.GetSize()):
        out = sitk.Cast(seg, seg.GetPixelID())
        out.CopyInformation(ct)          # 배열 유지, 메타만 CT로
        return out
    return sitk.Resample(seg, ct, sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, seg.GetPixelID())

def resample_to_isotropic(img: sitk.Image, spacing_mm: float, is_label: bool) -> sitk.Image:
    in_sp = np.array(img.GetSpacing(), float)
    in_sz = np.array(img.GetSize(), int)
    out_sp = np.array([spacing_mm]*3, float)
    out_sz = np.round(in_sz * in_sp / out_sp).astype(int).tolist()
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    return sitk.Resample(img, out_sz, sitk.Transform(), interp, img.GetOrigin(),
                         out_sp.tolist(), img.GetDirection(), 0, img.GetPixelID())
```

- [ ] **Step 4: 통과 확인** — Run pytest test_geometry → PASS
- [ ] **Step 5: 커밋**

```bash
git add ai_bone/geometry.py tests/ai_bone/test_geometry.py
git commit -m "feat(bone-v1): geometry alignment (size-branch) + isotropic resample"
```

---

### Task 5: harmonize_case (remap + 정합 + iso)

**Files:**
- Create: `ai_bone/harmonize.py`
- Test: `tests/ai_bone/test_harmonize.py`

**Interfaces:**
- Consumes: `label_map.LabelMap`, `geometry.align_geometry/resample_to_isotropic`, `taxonomy_v1`.
- Produces: `harmonize_case(ct_img: sitk.Image, seg_img: sitk.Image, lm: LabelMap, spacing_mm=0.6) -> tuple[sitk.Image, sitk.Image]` (iso CT, iso 통합라벨 seg). 라벨은 remap→align→iso(Nearest) 순.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_harmonize.py`

```python
import numpy as np, SimpleITK as sitk
from ai_bone.harmonize import harmonize_case
from ai_bone.label_map import LabelMap
from ai_bone import taxonomy_v1 as tx

def _img(arr, spacing=(1,1,1)):
    im = sitk.GetImageFromArray(arr); im.SetSpacing(spacing); return im

def test_harmonize_remaps_and_isotropic():
    ct = _img(np.zeros((8,8,8),np.int16), spacing=(0.8,0.8,0.6))
    seg = _img((np.arange(8*8*8).reshape(8,8,8) % 3).astype(np.uint8), spacing=(0.8,0.8,0.6))
    lm = LabelMap("t","nifti_seg","public", {1:"C1",2:"C2"}, {}, ["C1","C2"])
    out_ct, out_seg = harmonize_case(ct, seg, lm, spacing_mm=0.6)
    assert np.allclose(out_ct.GetSpacing(), (0.6,0.6,0.6))
    vals = set(np.unique(sitk.GetArrayFromImage(out_seg)).tolist())
    assert vals <= {0, tx.name_to_id("C1"), tx.name_to_id("C2")}
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/harmonize.py`

```python
import numpy as np
import SimpleITK as sitk
from ai_bone.geometry import align_geometry, resample_to_isotropic
from ai_bone.label_map import LabelMap

def harmonize_case(ct_img, seg_img, lm: LabelMap, spacing_mm: float = 0.6):
    # 1) 원본 라벨값 → 통합 id (배열 연산, 메타 유지)
    arr = sitk.GetArrayFromImage(seg_img)
    remapped = lm.remap_array(arr).astype(np.uint8)   # 0..53, 255=ignore
    seg_u = sitk.GetImageFromArray(remapped)
    seg_u.CopyInformation(seg_img)
    # 2) CT-seg 정합
    seg_a = align_geometry(ct_img, seg_u)
    # 3) 등방 resample (CT=BSpline, label=Nearest)
    out_ct = resample_to_isotropic(ct_img, spacing_mm, is_label=False)
    out_seg = resample_to_isotropic(seg_a, spacing_mm, is_label=True)
    return out_ct, out_seg
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_harmonize.py -q
git add ai_bone/harmonize.py tests/ai_bone/test_harmonize.py
git commit -m "feat(bone-v1): harmonize_case (remap+align+isotropic)"
```

---

### Task 6: dedup (이미지 fingerprint)

**Files:**
- Create: `ai_bone/dedup.py`
- Test: `tests/ai_bone/test_dedup.py`

**Interfaces:**
- Produces: `image_fingerprint(img: sitk.Image, box=8) -> str` (다운샘플 그리드 강도 해시), `find_duplicates(items: list[tuple[key,sitk.Image]], thresh=0.98) -> list[list[key]]` (상관>thresh 군집).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_dedup.py`

```python
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
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/dedup.py`

```python
import hashlib
import numpy as np
import SimpleITK as sitk

def _grid(img, box=8):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    zi = np.linspace(0, arr.shape[0]-1, box).astype(int)
    yi = np.linspace(0, arr.shape[1]-1, box).astype(int)
    xi = np.linspace(0, arr.shape[2]-1, box).astype(int)
    g = arr[np.ix_(zi,yi,xi)]
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel()

def image_fingerprint(img, box=8) -> str:
    q = np.round(_grid(img, box), 2).tobytes()
    return hashlib.sha1(q).hexdigest()

def find_duplicates(items, thresh=0.98, box=8):
    vecs = {k: _grid(im, box) for k, im in items}
    keys = list(vecs); used=set(); groups=[]
    for i,k in enumerate(keys):
        if k in used: continue
        grp=[k]
        for k2 in keys[i+1:]:
            if k2 in used: continue
            c = float(np.corrcoef(vecs[k], vecs[k2])[0,1])
            if c >= thresh: grp.append(k2); used.add(k2)
        used.add(k); groups.append(grp)
    return groups
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_dedup.py -q
git add ai_bone/dedup.py tests/ai_bone/test_dedup.py
git commit -m "feat(bone-v1): image fingerprint dedup"
```

---

### Task 7: build_raw (nnUNet_raw 조립 + present_labels + dataset.json)

**Files:**
- Create: `ai_bone/build_raw.py`
- Test: `tests/ai_bone/test_build_raw.py`

**Interfaces:**
- Consumes: `taxonomy_v1`.
- Produces: `write_dataset_json(out_dir, num_training, present_union) -> dict` (nnU-Net v2 dataset.json, labels=통합, `"ignore": 255` 규약), `write_present_sidecar(out_dir, case_id, present_labels) -> path`.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_build_raw.py`

```python
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
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/build_raw.py`

```python
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
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_build_raw.py -q
git add ai_bone/build_raw.py tests/ai_bone/test_build_raw.py
git commit -m "feat(bone-v1): build_raw dataset.json(ignore=255)+present sidecar"
```

---

### Task 8: verify_dataset (검증 게이트)

**Files:**
- Create: `ai_bone/verify_dataset.py`
- Test: `tests/ai_bone/test_verify.py`

**Interfaces:**
- Produces: `verify_case(ct: sitk.Image, seg: sitk.Image, hu_thr=200) -> dict` (keys: `empty`(bool), `size_match`(bool), `overlap_ratio`(float: 라벨∩HU≥thr / 라벨), `labels`(set)); `is_pass(report) -> bool` (empty False & size_match & overlap>=0.5).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_verify.py`

```python
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
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/verify_dataset.py`

```python
import numpy as np
import SimpleITK as sitk

def verify_case(ct: sitk.Image, seg: sitk.Image, hu_thr: int = 200) -> dict:
    hu = sitk.GetArrayFromImage(ct)
    lab = sitk.GetArrayFromImage(seg)
    fg = lab > 0
    n = int(fg.sum())
    overlap = float(((fg) & (hu >= hu_thr)).sum()) / n if n else 0.0
    return {
        "empty": n == 0,
        "size_match": tuple(ct.GetSize()) == tuple(seg.GetSize()),
        "overlap_ratio": overlap,
        "labels": set(np.unique(lab).tolist()),
    }

def is_pass(report: dict) -> bool:
    return (not report["empty"]) and report["size_match"] and report["overlap_ratio"] >= 0.5
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_verify.py -q
git add ai_bone/verify_dataset.py tests/ai_bone/test_verify.py
git commit -m "feat(bone-v1): verify_dataset gate (empty/size/overlap)"
```

---

### Task 9: 범용 재개 다운로더

**Files:**
- Create: `ai_bone/download.py`
- Test: `tests/ai_bone/test_download.py`

**Interfaces:**
- Produces: `download_file(url, dest, resume=True, session=None) -> Path` (HTTP Range 재개, `session` 주입으로 테스트), `parse_zenodo_manifest(record_json: dict) -> list[dict]` (files→{name,url,size}).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_download.py`

```python
from ai_bone.download import parse_zenodo_manifest, download_file

def test_parse_zenodo():
    rec = {"files":[{"key":"a.zip","links":{"self":"https://x/a.zip"},"size":10}]}
    out = parse_zenodo_manifest(rec)
    assert out == [{"name":"a.zip","url":"https://x/a.zip","size":10}]

class _FakeResp:
    status_code=200; headers={"content-length":"3"}
    def iter_content(self, n): yield b"abc"
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def raise_for_status(self): pass
class _FakeSession:
    def get(self, url, stream, headers, timeout): return _FakeResp()

def test_download_writes(tmp_path):
    dest = tmp_path/"a.bin"
    download_file("https://x/a.bin", dest, resume=False, session=_FakeSession())
    assert dest.read_bytes()==b"abc"
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/download.py`

```python
import os
from pathlib import Path

def parse_zenodo_manifest(record_json: dict):
    out = []
    for f in record_json.get("files", []):
        out.append({"name": f["key"], "url": f["links"]["self"], "size": f.get("size")})
    return out

def download_file(url, dest, resume=True, session=None, chunk=1 << 20):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if session is None:
        import requests
        session = requests.Session()
    pos = dest.stat().st_size if (resume and dest.exists()) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    mode = "ab" if pos else "wb"
    with session.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            for c in r.iter_content(chunk):
                if c: f.write(c)
    return dest
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_download.py -q
git add ai_bone/download.py tests/ai_bone/test_download.py
git commit -m "feat(bone-v1): resumable downloader + zenodo manifest parse"
```

---

## Phase 2 — 평가 지표 (순수 파이썬)

### Task 10: metrics + evaluate

**Files:**
- Create: `ai_bone/eval/metrics.py`, `ai_bone/eval/evaluate.py`, `ai_bone/eval/__init__.py`
- Test: `tests/ai_bone/test_metrics.py`

**Interfaces:**
- Produces: `dice(gt: np.ndarray, pred: np.ndarray, label: int) -> float`; `hd95(gt, pred, label, spacing) -> float`(빈 마스크→nan); `evaluate_dir(gt_dir, pred_dir, spacing=0.6) -> dict[label_name, {dice, hd95, n}]`.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_metrics.py`

```python
import numpy as np
from ai_bone.eval.metrics import dice, hd95

def test_dice_perfect_and_zero():
    a=np.zeros((4,4,4),int); a[1:3,1:3,1:3]=5
    assert dice(a,a,5)==1.0
    b=np.zeros((4,4,4),int)
    assert dice(a,b,5)==0.0

def test_hd95_zero_when_identical():
    a=np.zeros((5,5,5),int); a[2,2,2]=3
    assert hd95(a,a,3,(1,1,1))==0.0

def test_dice_absent_label_is_nan():
    a=np.zeros((4,4,4),int); b=np.zeros((4,4,4),int)
    v=dice(a,b,7)
    assert np.isnan(v)
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/eval/metrics.py`

```python
import numpy as np
from scipy.ndimage import distance_transform_edt

def dice(gt, pred, label):
    g = gt == label; p = pred == label
    denom = g.sum() + p.sum()
    if denom == 0:
        return float("nan")            # 라벨이 GT·pred 모두 없음
    return 2.0 * (g & p).sum() / denom

def _surface_dist(a, b, spacing):
    # a 표면에서 b 표면까지 거리
    if not a.any() or not b.any():
        return np.array([np.inf])
    dt = distance_transform_edt(~b, sampling=spacing)
    border = a & ~_erode(a)
    return dt[border]

def _erode(m):
    from scipy.ndimage import binary_erosion
    return binary_erosion(m)

def hd95(gt, pred, label, spacing):
    g = gt == label; p = pred == label
    if not g.any() and not p.any():
        return float("nan")
    if not g.any() or not p.any():
        return float("inf")
    d = np.concatenate([_surface_dist(g, p, spacing), _surface_dist(p, g, spacing)])
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 95)) if d.size else float("inf")
```

`ai_bone/eval/evaluate.py`:
```python
import glob, os
import numpy as np, SimpleITK as sitk
from ai_bone import taxonomy_v1 as tx
from ai_bone.eval.metrics import dice, hd95

def evaluate_dir(gt_dir, pred_dir, spacing=0.6):
    res = {n: {"dice": [], "hd95": []} for n in tx.FG_NAMES}
    for gp in sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz"))):
        cid = os.path.basename(gp)
        pp = os.path.join(pred_dir, cid)
        if not os.path.exists(pp): continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gp))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(pp))
        for i, name in enumerate(tx.FG_NAMES, start=1):
            d = dice(gt, pr, i)
            if not np.isnan(d):
                res[name]["dice"].append(d)
                res[name]["hd95"].append(hd95(gt, pr, i, (spacing,)*3))
    out = {}
    for n, v in res.items():
        out[n] = {"dice": float(np.nanmean(v["dice"])) if v["dice"] else float("nan"),
                  "hd95": float(np.nanmean(v["hd95"])) if v["hd95"] else float("nan"),
                  "n": len(v["dice"])}
    return out
```

- [ ] **Step 4: `ai_bone/eval/__init__.py` 빈 파일 생성. 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_metrics.py -q
git add ai_bone/eval tests/ai_bone/test_metrics.py
git commit -m "feat(bone-v1): eval metrics (dice/hd95) + evaluate_dir"
```

---

## Phase 3 — MERIT (순수 파이썬 로직)

### Task 11: PCA conflict split

**Files:**
- Create: `ai_bone/merit/split.py`, `ai_bone/merit/__init__.py`
- Test: `tests/ai_bone/test_split.py`

**Interfaces:**
- Produces: `pca_conflict_split(grad_vectors: dict[str, np.ndarray], k: int = 2) -> dict[int, list[str]]` (데이터셋명→그룹id; 정규화→PCA 제1축 투영 부호로 2분할, k=3이면 제2축까지 사분면 축약).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_split.py`

```python
import numpy as np
from ai_bone.merit.split import pca_conflict_split

def test_splits_opposing_groups():
    # 두 무리: +방향 3개, -방향 3개 → 서로 다른 파티션
    g = {}
    for i in range(3): g[f"pos{i}"] = np.array([1.0,0.0]) + 0.01*i
    for i in range(3): g[f"neg{i}"] = np.array([-1.0,0.0]) + 0.01*i
    part = pca_conflict_split(g, k=2)
    groups = list(part.values())
    # pos끼리, neg끼리 같은 그룹
    gid = {name: p for p, names in part.items() for name in names}
    assert gid["pos0"]==gid["pos1"]==gid["pos2"]
    assert gid["neg0"]==gid["neg1"]==gid["neg2"]
    assert gid["pos0"]!=gid["neg0"]

def test_all_datasets_assigned():
    g={f"d{i}":np.random.default_rng(i).normal(size=5) for i in range(7)}
    part=pca_conflict_split(g,k=2)
    assigned=[n for names in part.values() for n in names]
    assert sorted(assigned)==sorted(g)
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/merit/split.py`

```python
import numpy as np

def pca_conflict_split(grad_vectors: dict, k: int = 2) -> dict:
    names = list(grad_vectors)
    X = np.stack([grad_vectors[n] for n in names]).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # 방향 정규화
    Xc = X - X.mean(0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt.T                        # (N, comps)
    if k <= 2:
        gid = (proj[:, 0] >= 0).astype(int)
    else:
        # k==3+: 제1축 부호 + 제2축 부호 사분면을 그룹으로 축약
        a = (proj[:, 0] >= 0).astype(int)
        b = (proj[:, 1] >= 0).astype(int) if proj.shape[1] > 1 else np.zeros(len(names), int)
        quad = a * 2 + b
        uniq = {q: i for i, q in enumerate(sorted(set(quad.tolist())))}
        gid = np.array([uniq[q] for q in quad])
    part = {}
    for name, g in zip(names, gid.tolist()):
        part.setdefault(int(g), []).append(name)
    return part
```

- [ ] **Step 4: `ai_bone/merit/__init__.py` 생성. 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_split.py -q
git add ai_bone/merit/split.py ai_bone/merit/__init__.py tests/ai_bone/test_split.py
git commit -m "feat(bone-v1): MERIT PCA conflict split"
```

---

### Task 12: weight merge (가중평균 + TIES)

**Files:**
- Create: `ai_bone/merit/merge.py`
- Test: `tests/ai_bone/test_merge.py`

**Interfaces:**
- Produces: `weighted_average(state_dicts: list[dict], weights: list[float]) -> dict`; `ties_merge(base: dict, state_dicts: list[dict], weights: list[float], density=0.2) -> dict` (task vector trim→sign elect→merge). 텐서는 numpy 또는 torch 모두 지원(덕타이핑: `*`, `+`, `abs`).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_merge.py`

```python
import numpy as np
from ai_bone.merit.merge import weighted_average, ties_merge

def test_weighted_average_convex():
    a={"w":np.array([0.0,0.0])}; b={"w":np.array([2.0,4.0])}
    out=weighted_average([a,b],[0.25,0.75])
    assert np.allclose(out["w"], [1.5,3.0])

def test_weights_normalized():
    a={"w":np.array([1.0])}; b={"w":np.array([3.0])}
    out=weighted_average([a,b],[1,1])
    assert np.allclose(out["w"], [2.0])

def test_ties_preserves_shape_and_sign():
    base={"w":np.array([0.0,0.0,0.0])}
    m1={"w":np.array([1.0,0.0,-1.0])}
    m2={"w":np.array([1.0,0.0, 0.0])}
    out=ties_merge(base,[m1,m2],[1,1],density=1.0)
    assert out["w"].shape==(3,)
    assert out["w"][0] > 0            # 부호 합의(+) 유지
```

- [ ] **Step 2: 실패 확인** — FAIL(import)

- [ ] **Step 3: 구현** — `ai_bone/merit/merge.py`

```python
import numpy as np

def _norm(weights):
    s = float(sum(weights))
    return [w / s for w in weights]

def weighted_average(state_dicts, weights):
    w = _norm(weights)
    out = {}
    for k in state_dicts[0]:
        acc = None
        for sd, wi in zip(state_dicts, w):
            term = sd[k] * wi
            acc = term if acc is None else acc + term
        out[k] = acc
    return out

def ties_merge(base, state_dicts, weights, density=0.2):
    w = _norm(weights)
    out = {}
    for k in base:
        taus = [sd[k] - base[k] for sd in state_dicts]        # task vectors
        trimmed = []
        for t in taus:
            a = np.abs(np.asarray(t))
            if a.size:
                thr = np.quantile(a, 1.0 - density)
                trimmed.append(np.where(a >= thr, np.asarray(t), 0.0))
            else:
                trimmed.append(np.asarray(t))
        stack = np.stack([tw * wi for tw, wi in zip(trimmed, w)])
        sign = np.sign(stack.sum(0))                          # elect sign
        agree = np.where(np.sign(stack) == sign, stack, 0.0)
        cnt = np.sum(np.sign(stack) == sign, axis=0)
        merged = np.where(cnt > 0, agree.sum(0) / np.maximum(cnt, 1), 0.0)
        out[k] = np.asarray(base[k]) + merged
    return out
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_merge.py -q
git add ai_bone/merit/merge.py tests/ai_bone/test_merge.py
git commit -m "feat(bone-v1): MERIT weight merge (weighted-avg + TIES)"
```

---

### Task 13: estimate_conflict (서버 gradient 추출; 문법·import만 로컬 검증)

**Files:**
- Create: `ai_bone/merit/estimate_conflict.py`
- Test: `tests/ai_bone/test_trainers_import.py` (ast.parse만)

**Interfaces:**
- Produces: CLI `python -m ai_bone.merit.estimate_conflict --init <ckpt> --raw <DatasetXXX> --datasets a,b,c --out g.npz`. 각 데이터셋 소량 배치 gradient(seg head + 디코더 말단)를 랜덤투영으로 저차원화하여 `g.npz`(dataset명→벡터)로 저장. **GPU 필요 → 서버 실행**. 로컬은 파싱만.

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_trainers_import.py`

```python
import ast, pathlib
FILES = [
  "ai_bone/merit/estimate_conflict.py",
  "ai_bone/train/partial_label_trainer.py",
  "ai_bone/train/merit_finetune_trainer.py",
]
def test_parse_server_scripts():
    for f in FILES:
        ast.parse(pathlib.Path(f).read_text(encoding="utf-8"))
```

- [ ] **Step 2: 실패 확인** — FAIL(파일 없음)

- [ ] **Step 3: 구현** — `ai_bone/merit/estimate_conflict.py` (서버용; 저차원 gradient 추출)

```python
"""데이터셋별 gradient 방향 추정 → PCA split 입력용 g.npz. (서버 GPU 실행)"""
import argparse, numpy as np

def reduce_grad(flat_grad: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """고정 랜덤투영으로 저차원화. proj: (D_low, D_full)."""
    return proj @ flat_grad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--datasets", required=True)   # comma-sep
    ap.add_argument("--out", default="g.npz")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa
    # 아래는 서버에서 nnU-Net predictor/trainer 내부를 재사용해 grad를 뽑는다.
    # 핵심: 각 dataset 배치로 forward+backward → seg head+디코더말단 grad flatten
    #       → reduce_grad(proj) 저차원 → dataset별 평균 벡터.
    # (실제 nnU-Net 내부 연결은 서버 환경에서 확정: Task16 runbook에 실행법 명시)
    raise SystemExit("run on server; see ai_bone/runbook.md §MERIT")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인(파싱)** — `... -m pytest tests/ai_bone/test_trainers_import.py -q` (Task14 파일 생성 후 최종 PASS)
- [ ] **Step 5: 커밋**

```bash
git add ai_bone/merit/estimate_conflict.py tests/ai_bone/test_trainers_import.py
git commit -m "feat(bone-v1): MERIT gradient estimation scaffold (server)"
```

---

## Phase 4 — 학습 스크립트 (서버; 로컬은 파싱/import 검증)

### Task 14: 커스텀 trainer (partial-label + MERIT fine-tune)

**Files:**
- Create: `ai_bone/train/partial_label_trainer.py`, `ai_bone/train/merit_finetune_trainer.py`
- Test: `tests/ai_bone/test_trainers_import.py` (Task13에서 이미 포함)

**Interfaces:**
- Produces: `nnUNetTrainerNoMirroring_ES_PL` (기존 ES trainer 상속 + nnU-Net 네이티브 `ignore_label` 활용; 문서화 목적의 얇은 서브클래스, oversample 비율 상향), `nnUNetTrainerMERITFinetune` (낮은 LR·짧은 epoch, `initial_lr=1e-3`, `num_epochs=300`).

- [ ] **Step 1** — (테스트는 Task13의 `test_parse_server_scripts`가 커버: 두 파일 ast.parse)
- [ ] **Step 2: 실패 확인** — 파일 없으면 FAIL

- [ ] **Step 3: 구현** — `ai_bone/train/partial_label_trainer.py`

```python
"""Partial-label trainer. nnU-Net v2는 dataset.json의 ignore label(255)을 네이티브
지원하므로 loss 마스킹은 프레임워크가 처리한다. 여기선 좌우 NoMirroring + ES를
상속하고, 통합셋의 데이터셋 불균형 완화를 위해 foreground oversample을 올린다."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoMirroring_ES import (
    nnUNetTrainerNoMirroring_ES,
)

class nnUNetTrainerNoMirroring_ES_PL(nnUNetTrainerNoMirroring_ES):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.5   # 희소 뼈 대비 상향
```

`ai_bone/train/merit_finetune_trainer.py`:
```python
"""MERIT 파티션 fine-tune: shared init에서 짧고 낮은 LR (병합 basin 유지)."""
from ai_bone.train.partial_label_trainer import nnUNetTrainerNoMirroring_ES_PL

class nnUNetTrainerMERITFinetune(nnUNetTrainerNoMirroring_ES_PL):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300
```

- [ ] **Step 4: 파싱 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_trainers_import.py -q
git add ai_bone/train/partial_label_trainer.py ai_bone/train/merit_finetune_trainer.py
git commit -m "feat(bone-v1): partial-label + MERIT finetune trainers"
```

> **주의(런북에 명시):** 두 trainer 파일은 서버 `.../nnunetv2/training/nnUNetTrainer/`에 복사해야 nnU-Net이 `-tr` 이름으로 찾는다. 로컬은 nnunetv2 미설치라 import는 안 되지만 ast.parse로 문법만 검증한다. 기존 ES trainer의 **명시적 `__init__` 시그니처** 규칙(plans,configuration,fold,dataset_json,device) 준수 필수(과거 KeyError 교훈).

---

### Task 15: 학습 shell 스크립트 (GPU 1장 단위)

**Files:**
- Create: `ai_bone/train/stage1_pretrain.sh`, `ai_bone/train/stage2_baseline.sh`, `ai_bone/merit/merit_train_partition.sh`
- Test: `tests/ai_bone/test_scripts_shellcheck.py` (구문·필수 export 존재 검사)

**Interfaces:**
- Produces: 3개 실행 스크립트. 모두 **인자로 GPU 1장·fold** 받고, 공통 env export 포함, checkpoint 재개(`--c`).

- [ ] **Step 1: 실패 테스트** — `tests/ai_bone/test_scripts_shellcheck.py`

```python
import pathlib
SCRIPTS = ["ai_bone/train/stage1_pretrain.sh","ai_bone/train/stage2_baseline.sh",
           "ai_bone/merit/merit_train_partition.sh"]
def test_scripts_have_required_env():
    for s in SCRIPTS:
        t = pathlib.Path(s).read_text()
        assert "LD_LIBRARY_PATH" in t
        assert "nnUNet_compile=f" in t
        assert "CUDA_VISIBLE_DEVICES" in t
```

- [ ] **Step 2: 실패 확인** — FAIL(파일 없음)

- [ ] **Step 3: 구현** — `ai_bone/train/stage1_pretrain.sh`

```bash
#!/usr/bin/env bash
# Stage1: CADS 사전학습 (GPU 1장). 사용법: bash stage1_pretrain.sh <GPU_ID>
set -euo pipefail
GPU="${1:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 500 3d_fullres all \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL --c
```

`ai_bone/train/stage2_baseline.sh`:
```bash
#!/usr/bin/env bash
# Stage2: joint-pooling baseline. 사용법: bash stage2_baseline.sh <FOLD> <GPU_ID>
set -euo pipefail
FOLD="${1:?fold}"; GPU="${2:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 510 3d_fullres "$FOLD" \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES_PL \
  -pretrained_weights /data1/bone/nnunet/results/Dataset500_AxialPretrain/*_all/checkpoint_final.pth --c
```

`ai_bone/merit/merit_train_partition.sh`:
```bash
#!/usr/bin/env bash
# MERIT 파티션 fine-tune. 사용법: bash merit_train_partition.sh <PARTITION_DATASET_ID> <FOLD> <GPU_ID>
set -euo pipefail
DID="${1:?dataset id}"; FOLD="${2:?fold}"; GPU="${3:-0}"
source ~/miniforge3/etc/profile.d/conda.sh && conda activate pt210_py312
export nnUNet_raw=/data1/bone/nnunet/raw
export nnUNet_preprocessed=/home/ubuntu/nnunet_pre
export nnUNet_results=/data1/bone/nnunet/results
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export nnUNet_compile=f
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train "$DID" 3d_fullres "$FOLD" \
  -p nnUNetPlans_iso06 -tr nnUNetTrainerMERITFinetune \
  -pretrained_weights /data1/bone/nnunet/results/Dataset500_AxialPretrain/*_all/checkpoint_final.pth --c
```

- [ ] **Step 4: 통과 확인 + 커밋**

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone/test_scripts_shellcheck.py -q
git add ai_bone/train/*.sh ai_bone/merit/merit_train_partition.sh tests/ai_bone/test_scripts_shellcheck.py
git commit -m "feat(bone-v1): stage1/2 + MERIT partition train scripts (1-GPU)"
```

---

## Phase 5 — 런북 + 전체 검증

### Task 16: runbook.md (사용자 실행 매뉴얼)

**Files:**
- Create: `ai_bone/runbook.md`
- Test: 없음(문서). 대신 Task 17에서 전체 pytest+파싱 스모크.

**Interfaces:** 없음(문서).

- [ ] **Step 1: 작성** — `ai_bone/runbook.md`. 아래 섹션 필수(각 명령에 예상시간·재개법·검증 게이트 명시):
  1. **환경 준비**(conda, env export 3줄, trainer 파일 서버 복사 위치)
  2. **다운로드**(데이터셋별 `python -m ai_bone.datasets.<name>.download` 또는 download.py 사용; Zenodo 8병렬 제한 교훈)
  3. **통합 빌드**(`python -m ai_bone.build_raw --stage ft --out .../Dataset510_AxialFT`; CADS는 `--stage pretrain --out Dataset500`) → **`verify_dataset` 게이트 통과 필수**
  4. **전처리**(`nnUNetv2_plan_experiment ... -overwrite_target_spacing 0.6 0.6 0.6 -overwrite_plans_name nnUNetPlans_iso06` → `nnUNetv2_preprocess -d 500 510 -plans_name nnUNetPlans_iso06 -c 3d_fullres`; SSD 캐시 복사)
  5. **Stage1**(`bash stage1_pretrain.sh <gpu>`; 재개 `--c` 이미 포함)
  6. **Stage2 baseline**(가용 GPU마다 `bash stage2_baseline.sh <fold> <gpu>`; fold 0~4 순차/병렬 자유)
  7. **MERIT**(`estimate_conflict.py` → `split.py`로 partitions.json → 파티션별 `build_raw`로 Dataset520/521 생성 → `merit_train_partition.sh` → `merge.py`)
  8. **평가/추론**(`evaluate.py` A vs B, Mako 추론 + qc_overlay)
  9. **GPU 나눠쓰기 규약**(스크립트=GPU 1장, 동시 실행은 가용 GPU 수만큼, 각자 `--c` 재개, `nvidia-smi`로 여유 확인)
- [ ] **Step 2: 커밋**

```bash
git add ai_bone/runbook.md
git commit -m "docs(bone-v1): server runbook for shared-GPU execution"
```

---

### Task 17: 전체 스모크 + conftest

**Files:**
- Create: `conftest.py`(repo 루트; `sys.path`에 repo 추가), `tests/ai_bone/__init__.py`
- Test: 전체 `pytest`

**Interfaces:** 없음.

- [ ] **Step 1: `conftest.py` 작성**

```python
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
```

- [ ] **Step 2: 전체 테스트 실행**

Run: `"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -m pytest tests/ai_bone -q`
Expected: PASS (전 Phase 테스트 green)

- [ ] **Step 3: 파싱 스모크(전 .py)** — Run:

```bash
"C:\ProgramData\anaconda3\envs\ct_env\python.exe" -c "import ast,glob; [ast.parse(open(f,encoding='utf-8').read()) for f in glob.glob('ai_bone/**/*.py',recursive=True)]; print('parse OK')"
```
Expected: `parse OK`

- [ ] **Step 4: 커밋**

```bash
git add conftest.py tests/ai_bone/__init__.py
git commit -m "test(bone-v1): conftest + full smoke green"
```

---

## Self-Review 체크(작성자 확인 완료)
- **스펙 커버리지:** taxonomy(§3)=T1, label_map/partial-label(§4)=T2·T3, 파이프라인(§5)=T4~T9, 학습 3-stage(§6)=T14·T15(+pretrain/baseline/MERIT), MERIT(§6.3)=T11·T12·T13·T15, 평가(§7)=T10, 파일구조(§8)=전체, 런북(§9)=T16, 로컬테스트(§10)=각 태스크, 위험(§11)=검증게이트/1GPU/재개.
- **오픈이슈(§12)** 는 label_map 초안 `notes`와 런북에서 "다운로드 후 확정"으로 명시(값 확인 필요 부분만).
- **타입 일관성:** `remap_array`, `align_geometry`, `harmonize_case`, `pca_conflict_split`, `weighted_average/ties_merge`, `dice/hd95/evaluate_dir`, trainer 클래스명, 스크립트 인자 규약 전 태스크 일치.
- **주의:** T3의 label_map 원본 라벨값(TotalSeg/RibSeg/VerSe 등)은 **실제 데이터 다운로드 후 검증·정정**이 필요한 유일한 비확정부 → `verify_dataset` 게이트가 좌우/정합 오류를 잡도록 설계.
```
