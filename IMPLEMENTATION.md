# 3D CT Bone Visualizer — 구현 상세 정리

이 문서는 `main.py` + `dicom_utils.py`로 구성된 본 도구의 **모든 기능**이 어떤 라이브러리와 알고리즘으로 어떻게 동작하는지, 그리고 왜 그렇게 설계했는지를 한 곳에 정리한 기술 가이드입니다.

---

## 0. 전체 아키텍처

```
[ DICOM 파일들 ]
        │  load_scan(), load_series(), get_pixels_hu()  (dicom_utils.py)
        ▼
[ HU 3D 배열 + spacing + DICOM 메타 (IPP/IOP) ]
        │  np.savez_compressed (cache_v4.npz)
        ▼
[ pv.ImageData per series (series_volume_grids[i]) ]
        │  Stage C: voxel-level morphological opening (scipy.ndimage.binary_opening)
        │  contour() → isosurface (per series, native spacing 유지)
        │  Smoothing: Laplacian / Taubin (per series)
        │  Stage A: mesh-level fragment cleanup (per series)
        ▼
[ Mesh per series ]
        │  Multi-Series Fusion:
        │    base_grid = inv(T_base) @ T_i 적용해 같은 환자 좌표계로 정렬
        │    → 골반/무릎/발목이 자기 위치에 자기 spacing으로 배치
        ▼
[ Fused meshes (PolyData × N series) ]
        │  Stage B: click-to-remove (단일 모드 한정)
        │  3D Cropping (clip_box, 단일 모드 한정)
        ▼
[ 렌더링 + 인터랙션 ]
        │  Landmark Picking (surface point picking + hover preview)
        │  Measurements (거리/각도)
        │  Export: STL / CSV / JSON / Session JSON
```

**모듈 분리**
- `dicom_utils.py` — 순수 데이터 처리 (DICOM 파싱, HU 변환, 정렬, spacing 계산). Qt/PyVista 의존성 없음.
- `main.py` — 얇은 진입점 (`QApplication` + `MainWindow` + `exec`). 실행은 계속 `main.py`로 합니다.
- `app/main_window.py` — `MainWindow` 상태 초기화 및 `setup_controls` UI 구성.
- `app/mixins/*.py` — 기능별 mixin (`mesh_pipeline`, `fusion`, `cropping`, `particle_removal`, `bone_separation`, `landmarks`, `session_io`, `patient_load`, `export_scout`).
- `app/ui/collapsible.py` — 접이식 패널 위젯 `CollapsibleSection`.
- `app/constants.py` — `BASE_DATA_DIR`, `SESSION_FORMAT`, `SESSION_VERSION`, `MAKO_KEYWORDS`.

---

## 1. DICOM 로딩 & 캐싱

### 1.1 시리즈 분리 (`dicom_utils.load_scan`)
DICOM 폴더 안에는 메인 3D 볼륨뿐 아니라 스카우트(localizer), 도즈 리포트 등이 섞여있음. 분리 기준:
- `ImageType` 태그에 `LOCALIZER` 또는 `SCOUT` → `scout_slices` 리스트로 분리
- 나머지는 `(SeriesInstanceUID, SliceThickness)` 튜플을 key로 그룹화. **같은 UID라도 ST가 다르면 별개 볼륨**으로 처리

### 1.2 강건한 슬라이스 정렬 (`get_slice_position`)
파일명 순서를 신뢰하지 않음. 슬라이스가 axial인지 sagittal인지 모르므로:
- `IOP[:3]` × `IOP[3:]` = slice normal vector
- `dot(normal, IPP)` = slice의 normal축 방향 1D 좌표
- 이 1D 좌표로 정렬 → axial/coronal/sagittal 무관하게 정확

### 1.3 spacing 계산 (`load_series`)
- **Z (slice thickness)**: 헤더 값 신뢰하지 않고, 인접 슬라이스 IPP 차이의 **median** 사용. 표준편차가 10%↑면 non-uniform 경고
- **X, Y**: `PixelSpacing[1], PixelSpacing[0]` (DICOM 순서: row, column)
- 반환 튜플 순서: `(z_spacing, y_spacing, x_spacing)`

### 1.4 HU 변환 (`get_pixels_hu`)
`pixel × RescaleSlope + RescaleIntercept`. CT 스캐너 출고값을 표준 Hounsfield Unit으로 정규화 (뼈 ≈ +300~+3000 HU).

### 1.5 DICOM 환자 좌표계 메타 추출 (`main.on_load_clicked`)
정렬된 첫 슬라이스에서 추출해 meta dict에 저장:
- `ipp_first` — Image Position Patient (LPS mm, 첫 슬라이스 원점)
- `row_dir`, `col_dir` — IOP의 두 방향 벡터
- `normal_dir` — `cross(row_dir, col_dir)` (slice 적층 방향)
- `patient_position` — HFS/FFS 등

→ 이걸로 모든 voxel의 환자 좌표계 변환이 가능해짐 (§3 참조).

### 1.6 캐시 (cache v4)
- 파일명: `{patient_id}_cache_v4.npz`
- `np.savez_compressed`로 image_hu, spacing, meta, scout array들을 저장
- 두 번째 로드부터는 원본 DICOM 파싱을 완전히 건너뛰어 **수십 배 빠름**
- 버전을 v3→v4로 올린 이유: IPP/IOP 메타 추가. v3 캐시는 자동 무시되고 raw에서 재생성됨

---

## 2. 3D 메쉬 생성 파이프라인 (`update_base_mesh`)

### 2.1 ImageData 구성 (`on_series_switched`)
```python
self.volume_grid = pv.ImageData(
    dimensions=(nx, ny, nz),
    spacing=(sx, sy, sz)
)
self.volume_grid.point_data["values"] = image_hu.flatten(order="C")
```
PyVista의 `ImageData`는 균일 격자라 contour 등 isosurface 추출이 매우 빠름.

### 2.2 Threshold masking (boundary 보존 전략)
**핵심 설계**: contour의 sub-voxel 보간 정확도를 유지하기 위해 **min 미만 voxel은 절대 건드리지 않음**.
```python
mask_out = hu > self.current_max_threshold     # max 초과만 OFF
# (+ Stage C에서 작은 컴포넌트도 mask_out에 추가)
values[mask_out] = min_threshold - 1
```
- min 미만 voxel을 임의로 값 변경하면 marching cubes가 계단(staircase) artifact를 만듦
- 이건 medical imaging의 "partial volume effect 보존" 원칙

### 2.3 Isosurface 추출
`self.volume_grid.contour([min_threshold], scalars="masked")` — vtkContourFilter 기반. 결과는 `pv.PolyData` (vtkMarchingCubes).

### 2.4 Smoothing
- **None** — raw marching cubes 결과
- **Laplacian** — `mesh.smooth(n_iter=100)`. 단순/빠름, 부피 축소 부작용 있음
- **Windowed Sinc / Taubin** — `mesh.smooth_taubin(n_iter=50, pass_band=0.05)`. 의료영상 표준, 부피·feature 보존

---

## 3. 좌표계 시스템

세 가지 좌표계를 모두 지원. 정확한 변환을 위해 IPP/IOP가 필요.

| 좌표계 | 정의 | 사용처 |
|---|---|---|
| Grid | PyVista `ImageData` 월드 좌표, 원점 (0,0,0), mm | 모든 picking 좌표의 원본 |
| LPS | DICOM 환자 Left/Posterior/Superior, mm | DICOM 표준, RT planning |
| RAS | Right/Anterior/Superior, mm | 3D Slicer, 일반 medical imaging |

### 3.1 Grid → LPS (`_grid_to_lps`)
```
LPS = IPP_first + X·row_dir + Y·col_dir + Z·normal_dir
```
물리적으로: 첫 슬라이스 원점에서 각 grid 축 방향으로 mm만큼 이동. row_dir/col_dir/normal_dir이 모두 unit vector이고 orthogonal하므로 그대로 mm 거리로 작동.

### 3.2 LPS → RAS (`_lps_to_ras`)
```
RAS = (-L, -P, S)
```
X(L→R), Y(P→A) 부호만 반전.

### 3.3 좌표계 불변성
**거리와 각도는 모든 좌표계에서 동일** (rigid transform invariant). 그래서 §6의 측정 계산은 항상 **grid 좌표**(가장 안정적, 모든 landmark에서 존재 보장)로만 수행.

---

## 4. 파티클 제거 (3-Stage 시스템)

### Stage C: Voxel-level Morphological Opening (pre-contour, 자동)

**위치**: `update_base_mesh` 내부, contour 직전
**알고리즘**: `scipy.ndimage.binary_opening` — erosion N회 → dilation N회

#### 왜 morphological opening인가
- 이전 버전(`remove_small_objects`)은 connected component 단위로만 동작. 뼈 본체에 가는 가시처럼 붙은 노이즈는 본체와 연결되어 있어 한 개의 큰 component로 묶이는 바람에 제거되지 않음.
- Opening은 **굵기**(voxel 폭)를 기준으로 잘라내므로, 본체와 연결된 얇은 가지/판/그물 구조도 깨끗하게 제거 가능.

#### 동작
```python
struct = generate_binary_structure(3, self.opening_connectivity)
opened_mask = binary_opening(
    bone_mask, structure=struct, iterations=self.opening_iterations
)
removed = bone_mask & ~opened_mask   # → mask_out에 추가
```
- **Erosion**: 이웃 중 하나라도 non-bone이면 그 voxel 제거 → 표면이 N voxel만큼 깎임
- **Dilation**: 살아남은 코어에서 N voxel만큼 다시 부풀어 오름
- **순효과**: 폭이 2N voxel 이하인 구조는 erosion 단계에서 끊어져 사라지고, 그보다 두꺼운 본체는 dilation으로 거의 복원

#### UI 파라미터
| 컨트롤 | 의미 |
|---|---|
| `[x] Morphological denoise (opening)` | Stage C ON/OFF 마스터 토글 (`particle_removal_enabled`) |
| `Iterations` (0~5) | erosion/dilation 반복 횟수. **0이면 opening을 완전히 skip** (no-op 경로) |
| `Connectivity` (6 / 18 / 26) | structuring element 모양. `generate_binary_structure(3, k)`로 매핑 |

| connectivity | 이웃 수 | 효과 |
|---|---|---|
| 1 → 6  | 면 공유   | 가장 보수적, 곡선이 매끄러움 |
| 2 → 18 | + 엣지   | 중간 |
| 3 → 26 | + 꼭짓점 | 가장 공격적, 대각선 노이즈에 강함 |

기본값 `iterations=1, connectivity=6`은 "표면 1 voxel 정도만 살짝 다듬는" 매우 가벼운 모드.

#### Boundary preservation (이전 버전과 동일하게 유지)
`hu < min_threshold` 인 voxel은 **절대 손대지 않음**. 그래야 marching cubes의 sub-voxel 선형 보간이 정상 작동해서 계단 현상이 없음. opening으로 잘려나간 voxel과 `hu > max_threshold` voxel만 `min-1`로 마스킹.

#### 비용 (대략)
1억 voxel(=512×512×~400) 기준:
- `iterations=1, conn=6`: ~수백 ms (slider 드래그 시 약간 끊김 가능)
- `iterations=2, conn=26`: 2~4배. 더 강한 denoise가 필요할 때만 사용
- `iterations=0`: skip 분기, 비용 0

뼈 형태가 살짝 줄어들 수 있다는 사용자 합의 하에 진행 — 협업자가 지적한 "iteration이 과해서 형태가 망가지는" 위험을 방지하기 위해 기본 1 iter, 6-conn 으로 보수적으로 시작.

### Stage A: Mesh-level (post-contour + smoothing)
**위치**: `update_base_mesh` 내부, smoothing 직후
**알고리즘**: PyVista `connectivity()` + `remove_cells`
```python
labeled = self.base_mesh.connectivity(extraction_mode='all')
# label 별 cell 개수 계산
unique, counts = np.unique(labeled.cell_data['RegionId'], return_counts=True)
keep_regions = unique[counts >= self.min_fragment_faces]
self.base_mesh = labeled.remove_cells(np.where(np.isin(region_ids_cell, remove_ids))[0])
```
또는 `Keep largest only` 토글 시 `mesh.extract_largest()` 한 줄.

**PolyData 보장**: `remove_cells`가 `UnstructuredGrid`를 반환할 수 있어서 `extract_surface()`로 변환. STL export 호환성 유지.

### Stage B: Click-to-Remove (수동)
**핸들러**: `on_picking_toggled` → `on_point_picked`
**Picking**: `plotter.enable_point_picking(use_picker='cell', left_clicking=True)`
**알고리즘**:
1. 클릭 좌표 받아 `base_mesh.connectivity()`로 라벨링
2. `find_closest_point(picked_point)`로 클릭 위치의 region ID 찾기
3. 그 region에 속한 cell만 `remove_cells`
**Undo 스택**: 클릭 직전 `base_mesh.copy(deep=True)` 스냅샷을 `_manual_undo_stack`에 push (최대 20개). threshold/series 변경 시 자동 클리어.

---

## 4.5 Bone Separation (Phase 1)

**목적**: 한 화면에 합쳐진 뼈 mesh를 *개별 해부학적 뼈* 단위로 분리해 각각 ON/OFF 가능하게 함. 척추 마디·비골/경골·손가락 같은 좁게 붙어있는 뼈를 다른 객체로 인식시키는 게 핵심.

**위치**: `_compute_separation_labels` + `_build_separated_bone_meshes` + 동적 UI 체크박스 (UI 섹션 "Bone Separation", Cropping 다음).

### 4.5.1 파이프라인 (`_compute_separation_labels`)

| 단계 | 연산 | 효과 |
|---|---|---|
| 1 | `(min ≤ HU ≤ max)` + Stage C opening (현재 설정 그대로) | bone mask 일관성 보장 |
| 2 | Crop box ROI로 마스크 (활성 시) | 계산량 ↓ + 사용자 ROI 한정 |
| 3 | `binary_closing(K)` | 표면 균열·좁은 틈 메움 |
| 4 | `binary_fill_holes` | 내부 골수강 등 닫힌 빈공간 채움 |
| 5 | `binary_erosion(N)` | 뼈 사이 좁은 연결부 끊기 |
| 6 | `ndi_label` | 끊긴 core에 정수 ID 부여 |
| 7 | `min_bone_voxels` 미만 컴포넌트 제거 + 1..k 재번호 | 노이즈 컷오프 |
| 8 | `watershed(-distance, markers=cores, mask=closed_mask)` | 표면을 잃지 않고 core를 원래 mask까지 확장, 라벨끼리는 distance ridge에서 멈춤 |

→ 결과 `label_volume`: shape == `image_hu`, dtype int32, 0=배경, 1..N=각 뼈.

### 4.5.2 메쉬 빌드 (`_build_separated_bone_meshes`)

label 한 개당:
1. `sub = (label_volume == lid).astype(uint8)`
2. `pv.ImageData`의 `point_data["sub"]` 갈아끼움
3. `grid.contour([0.5], scalars="sub")` — binary marching cubes
4. 현재 `_apply_smoothing` 적용 (Stage A는 의도적으로 미적용 — 분리된 뼈를 추가로 쪼개지 않기 위해)
5. PolyData + voxel count 반환

색상은 `_bone_color_palette` → matplotlib `tab20` 사이클로 자동 할당 (20개까지 distinct hue).

### 4.5.3 UI 동작

`Bone Separation` collapsible section:
- **Hole fill iter** (0~5) — `binary_closing` 반복 수
- **Separation erosion** (0~15) — 분리용 erosion 반복 수  
- **Min bone voxels** (0~1M) — 노이즈 컷오프
- **Separate Bones** 버튼 — 파이프라인 실행
- **Clear** 버튼 — 단일 mesh 복원
- 상태 라벨 — "Separated into 7 bones (took 1.4s, params: …)"
- 동적 체크박스 — `Bone 1 (12,340 vox)` × N개, indicator 색이 해당 뼈와 동일

체크 토글 시 `actor.SetVisibility()` 즉시 반영, `plotter.render()` 한 번 호출.

### 4.5.4 Fusion 모드 (`_compute_fusion_separated_bone_entries`)

Fusion ON일 때 **포함된 모든 series** (`fusion_include_flags`)에 대해 위 파이프라인을 각각 native spacing/grid에서 실행한 뒤, fusion 렌더링과 동일하게 `mesh.transform(inv(T_base) @ T_i)`로 base grid frame에 정렬.

- **Crop box**: base series grid mm 기준. base series는 그대로 `_bounds_to_voxel_slice`. 다른 series는 crop box 8모서리를 `base→LPS→series` 변환 후 AABB를 잡아 mask 제한. mesh는 base frame에서 `clip_box`로 한 번 더 클립 (안전망).
- **이름**: `[series_idx.sub_idx] description — B{label_id}` (UI 체크박스)
- **색상**: 전체 bone 수에 대해 `tab20` 한 번에 할당 (series 간 색 중복 가능)

### 4.5.5 Mutual Exclusion

| 동작 | 효과 |
|---|---|
| 단일/fusion mesh 재빌드 (threshold/smoothing/series 변경) | `_update_single_mesh` / `_update_fused_meshes` 시작 시 `_auto_clear_separation_on_remesh` → 자동 invalidate |
| Patient 교체 | `on_load_clicked`에서 명시적 clear |
| Click-to-Remove | fusion에서 여전히 disable (단일 mesh 편집 전제) |
| Landmark hover | 분리 모드일 때 visible separated actor만 picker 대상 |
| Clear Separation | fusion이면 `fusion_actors` visibility 복원, 단일이면 `current_mesh_actor` 복원 |

### 4.5.6 Phase 2 (정교화)

| 기능 | UI | 구현 |
|---|---|---|
| **이름 편집** | 리스트 더블클릭 또는 `Rename` | `QInputDialog` → `bone['name']` 갱신 |
| **병합** | `Merge Selected` (Ctrl+다중 선택) | `mesh.merge()` 연쇄 → 기존 actor 제거 → 새 actor 1개 |
| **개별 STL** | `Export Bones STL…` | visible bone마다 `{sanitized_name}.stl` + `bones_export_report.txt` |
| **Stage A per bone** | `Stage A cleanup per bone` 체크 | `_build_separated_bone_meshes`에서 contour/smooth 후 `_apply_stage_a` (Particle Removal의 Stage A 설정 공유) |

리스트는 `QListWidget` (체크=visibility, ExtendedSelection=병합). 각 bone은 `uid`(UUID)로 추적.

### 4.5.7 Morphology 주의 (왜 잘못 분리되었는지)

| 연산 | scipy 동작 | 과하면 생기는 문제 |
|---|---|---|
| **Closing** (Hole fill iter) | dilate → erode | **서로 다른 뼈가 붙음** — 가까운 골반·척추·비골 사이 틈을 메움 |
| **Erosion** (Separation) | 기본값은 **26-conn** (수정 전) | **같은 뼈가 쪼개짐** — 한 iteration마다 26방향으로 깎임 |
| **Opening** (Bridge break) | erode → dilate | 얇은 연결부만 끊음, 인접 뼈를 붙이지 않음 (closing과 반대) |
| **fill_holes** | 내부 완전 둘러싸인 공간만 | component 내부 골수강 등 — closing으로 이미 붙은 뒤면 무의미 |

**수정 (2026-05):**
- Erosion/closing 모두 **`generate_binary_structure(3, 1)` = 6-conn** 명시
- 기본값: `bridge_open=1`, `closing=0`, `erode=1`
- 콘솔에 `CC_after_fill` / `CC_after_erosion(seeds)` 로그 → 붙음/쪼개짐 진단용

**튜닝 가이드:**
- 뼈가 **여전히 붙음** → `Bridge break` 2, `Closing`은 0 유지
- 뼈가 **여전히 쪼개짐** → `Separation erosion` 0, `Min bone voxels` 올리기
- 관절이 **안 끊김** → `Separation erosion` 2 (6-conn 기준)

### 4.5.8 기본값 (2026-05 수정)
`bridge=0, close=0, erode=0, min=500`

- **erode=0 (CC 모드)**: mask에서 connected component마다 뼈 1개 — mask에 voxel이 있으면 거의 항상 뼈를 찾음.
- **Stage C opening은 분리 mask에 적용하지 않음** (표시용 mesh contour에만 적용).
- **min_bone_voxels**는 watershed **이후** 최종 label 크기로 필터 (작은 seed를 미리 제거하지 않음).
- erosion이 mask를 전부 지우면 자동으로 CC 모드로 fallback.

### 4.5.9 의존성

| 함수 | 출처 |
|---|---|
| `binary_closing`, `binary_fill_holes`, `binary_erosion`, `distance_transform_edt`, `label` | `scipy.ndimage` |
| `watershed` | `skimage.segmentation` |

scikit-image는 이미 의존성에 있음(0.23.2).

---

## 5. 3D Cropping

**Widget**: PyVista `add_box_widget` (vtkBoxWidget 래퍼)
**Clip**: `mesh.clip_box(self.cropping_bounds, invert=False)`

### 5.1 Persistent state
토글 OFF시 `cropping_bounds`와 `last_box_bounds`를 **그대로 유지**. `update_rendered_mesh`에서:
```python
if self.cropping_bounds is not None and self.crop_checkbox.isChecked():
    mesh = mesh.clip_box(...)
```
→ 토글로 잠시 해제했다 다시 켜면 같은 영역 복원됨.

### 5.2 Reset 버튼
`cropping_bounds`와 `last_box_bounds`를 `None`으로, 박스 위젯을 전체 bounds로 재생성.

### 5.3 Hide Crop Handles
박스 위치는 유지한 채 위젯의 시각적 handle만 숨김 (`clear_box_widgets()` + 토글 OFF시 같은 bounds로 재추가).

---

## 6. Landmark Picking

### 6.1 활성화 & Mutual Exclusion (`on_landmark_picking_toggled`)
- PyVista `enable_surface_point_picking(left_clicking=True)` 우선
- 버전 호환 fallback: `enable_point_picking(use_picker='cell', left_clicking=True)`
- **Click-to-Remove와 상호 배타**: 한 쪽 켤 때 다른 쪽 강제 OFF (같은 picking 채널 공유)

### 6.2 Hover Preview (마우스 따라다니는 노란 sphere)
**메커니즘**: VTK MouseMoveEvent observer + vtkCellPicker
```python
self._hover_observer_id = self.plotter.iren.add_observer(
    "MouseMoveEvent", self._on_hover_mouse_move
)
```
**매 마우스 이동**:
1. `iren.GetEventPosition()`으로 화면 좌표
2. `vtkCellPicker.Pick(x, y, 0, renderer)` ray casting
3. picker가 잡은 actor가 **bone mesh actor일 때만** sphere 표시
4. `actor.SetPosition()`으로 sphere를 즉시 이동 (geometry 재생성 X → 매우 빠름)

**최적화**: 매 frame render 호출은 비싸므로 **60Hz throttle** (`time.time() - last_render >= 1/60`)

**Lazy/Persistent**: sphere actor를 한 번만 만들고 visibility만 토글. picking OFF시 actor는 유지(빠른 재활성화), 환자 교체 시에만 destroy(스케일 재계산 위해).

### 6.3 클릭 확정 (`on_landmark_picked`)
1. picked_point (grid 좌표) 받아 `_grid_to_lps()`, `_lps_to_ras()` 변환
2. dict entry 생성: `{name: 'L{counter}', grid, lps, ras}`
3. **빨간 sphere** 추가 (`pickable=False` → 다음 클릭 시 sphere를 가리키지 않고 그 뒤 bone surface를 잡음)
4. 테이블 재구성

### 6.4 Sphere 크기 자동 스케일 (`_estimate_landmark_radius`)
```python
diag = sqrt((xmax-xmin)² + (ymax-ymin)² + (zmax-zmin)²)
radius = diag × 0.005   # 메쉬 대각선의 0.5%
```
환자마다 메쉬 크기가 달라도 비율은 일정.

### 6.5 자동 라벨 (`L1, L2, ...`)
`landmark_counter` 카운터 사용. 라벨은 **삭제 시 재사용되지 않음** (중복 방지). 사용자가 테이블에서 더블클릭으로 inline 편집 가능 (`_on_landmark_table_item_changed`).

---

## 7. UI 테이블 & 좌표계 토글

### 7.1 테이블 구조 (`QTableWidget`)
컬럼: `#` | `Name` | `X` | `Y` | `Z` | `✕` (개별 삭제 버튼)
- `setSelectionMode(ExtendedSelection)` — Ctrl+Click 멀티 선택 가능 (§8 측정용)
- Name 컬럼만 인라인 편집 가능, 나머지는 read-only
- 빈 이름 입력하면 자동 복원

### 7.2 좌표계 토글 (`landmark_coord_combo`)
콤보박스 변경 시 `_refresh_landmark_table()` → 테이블 헤더(`X/Y/Z → L/P/S → R/A/S`)와 값 모두 자동 변환. DICOM 메타 없는 경우 셀에 `N/A` 표시.

### 7.3 Series 전환 시 LPS/RAS 자동 재계산 (`on_series_switched`)
같은 환자 내 다른 series로 옮겨도 새 IPP/IOP 기준으로 모든 landmark의 lps/ras를 재계산. grid 좌표와 sphere 위치는 그대로 (같은 voxel grid이므로).

---

## 8. 측정 (거리 / 각도)

### 8.1 트리거
테이블 `itemSelectionChanged` 시그널 → `_on_landmark_selection_changed`. 선택 행 수에 따라 자동:
- 1개 이하 → 안내 메시지만
- **2개** → 거리
- **3개** → 각도 (vertex = 행 인덱스 기준 가운데)
- 4개 이상 → 안내 메시지

### 8.2 계산
좌표계 불변이라 **항상 grid 좌표** 사용.
```python
distance = np.linalg.norm(p1 - p2)
angle = degrees(arccos(clip(dot(v1, v2) / (|v1|·|v2|), -1, 1)))
```

### 8.3 3D 시각화
- **측정선**: `plotter.add_lines(np.array([p1, p2]), color='lime', width=3)`
- **거리 라벨**: 선분 중간에 `plotter.add_point_labels` (`always_visible=True`, `shape=None`)
- **각도 호** (3점일 때): 두 unit vector 사이의 **slerp**(spherical linear interpolation)으로 24개 호점 생성, magenta 라인 + 각도 라벨

### 8.4 Cleanup
`_clear_measurement_visualization()`이 모든 측정 actor를 plotter에서 제거. 호출 시점:
- 다음 선택 변경 (다시 그려짐)
- landmark 삭제 (인덱스 무효화)
- Clear All
- 새 환자 로드

---

## 9. Export

### 9.1 STL Export (`on_export_stl_clicked`)
- 현재 `base_mesh` 그대로 (모든 Stage C/A/B 결과 포함)
- Cropping이 active면 `clip_box` 적용
- 저장 폴더 안에 `mesh.stl` + `export_report.txt` (환자 정보, threshold, smoothing, cropping, particle removal 설정 모두 기록)

### 9.2 Landmark CSV (`_export_landmarks_csv`)
```
index, name, grid_X, grid_Y, grid_Z, [LPS_L, LPS_P, LPS_S, RAS_R, RAS_A, RAS_S]
```
DICOM 메타 없으면 LPS/RAS 컬럼 생략.

### 9.3 Landmark JSON (`_export_landmarks_json`)
환자 메타 + 좌표계 설명 + spacing + landmark 배열 (`grid`/`lps`/`ras`).

---

## 10. Session Save / Load

전체 워크플로우 상태를 단일 JSON으로 저장/복원.

### 10.1 저장 (`_collect_session_state`)
```json
{
  "format": "stanford_medicine_session",
  "version": 3,
  "patient": { patient_id, patient_name, series_uid, slice_thickness, ... },
  "render": { min_threshold, max_threshold, smooth_index },
  "particle_removal": {
    "voxel_enabled": true,
    "opening_iterations": 1,         // v2+: morphological opening
    "opening_connectivity": 1,       // 1=6, 2=18, 3=26
    "mesh_cleanup_enabled": false,
    "keep_largest_only": false,
    "min_fragment_faces": 100
  },
  "cropping": { enabled, hide_handles, bounds: [x0,x1,y0,y1,z0,z1] },
  "landmarks": { counter, coord_system, points: [...] },
  "view": { show_axes },
  "fusion": {                          // v3+
    "enabled": true,
    "base_series_index": 0
  },
  "camera": [[pos], [focal], [view_up]]
}
```

**버전 호환성**:
- v1 → v3: `min_particle_volume_mm3` 무시, opening 파라미터 기본값(1, 6-conn), fusion 기본값(ON, base=0)
- v2 → v3: fusion 블록 누락 시 기본값(ON, base=0)
- 더 새로운 파일은 경고 후 인식 가능한 필드만 로드

### 10.2 복원 (`_apply_session_state`) — 9단계
1. **환자 매칭**: `patient_id` 다르면 `patient_combo` 변경 + `on_load_clicked()` 자동 호출
2. **시리즈 매칭**: `series_uid` + `slice_thickness`로 정확한 series 선택
3. **렌더 파라미터**: `blockSignals(True)` 후 슬라이더/스핀박스 setValue
4. **파티클 제거** 설정
5. **Cropping**: `pv.Box(bounds=...)`로 PolyData 재생성, box widget 재설치
6. **메쉬 1회 빌드**: `_loading_session=False`로 풀고 `update_base_mesh()` 1회
7. **View 토글**
8. **Landmark 복원**: dict 재구성, 각 점의 sphere actor 재생성
9. **Camera 복원**: 마지막에 적용

### 10.3 핵심 최적화: `_loading_session` flag
복원 중 핸들러가 cascading으로 `update_base_mesh()`를 여러 번 트리거하는 걸 막음:
```python
def update_base_mesh(self):
    if self.volume_grid is None: return
    if self._loading_session: return  # ← 가드
    ...
```
컨트롤들을 다 세팅한 뒤 flag를 풀고 `update_base_mesh()`를 **딱 한 번** 호출 → 큰 환자도 빠르게 복원.

---

## 11. UI 구조

### 11.1 `QScrollArea` wrap
좌측 패널 전체가 스크롤 영역. 작은 화면이나 컨트롤이 많아져도 모든 위젯 접근 가능. 가로 스크롤바는 항상 숨김, 최소 너비 290px.

### 11.2 `CollapsibleSection` 헬퍼 클래스
```python
section = CollapsibleSection("Particle Removal", expanded=False)
section.addWidget(...)
parent_layout.addWidget(section)
```
헤더 버튼 (`▶`/`▼`)을 클릭하면 `content_widget.setVisible()` 토글.

### 11.3 섹션 구조 (워크플로우 순)
1. **Patient** — 환자 선택, Load, Export STL, Save/Load Session
2. **Series & Display** — series combo, 3D Axes, Smoothing
3. **Thresholds (HU)** — Min/Max 슬라이더 (가장 자주 사용)
4. **3D Cropping** (collapsible, 기본 접힘)
5. **Particle Removal** (collapsible, 기본 접힘)
6. **Landmark Points** (collapsible, 기본 펴짐 — 주 작업)
7. (stretch)
8. **하단**: Open Scout Viewer | Open Patient Info

### 11.4 외부 윈도우
- **Scout Viewer** — Matplotlib Figure로 scout/localizer 이미지 표시, Prev/Next 네비게이션
- **Patient Info** — DICOM 메타데이터를 HTML로 정리한 별도 창

---

## 12. 안전장치 / Edge Case 처리

### 12.1 환자 로드 시 자동 정리 (`on_load_clicked`)
- click-to-remove, landmark picking OFF
- landmark + actor + 측정 시각화 클리어
- hover preview destroy (다음 toggle 때 새 mesh 크기로 재생성)
- 좌표계가 환자마다 다르므로 stale landmark 방지

### 12.2 Series 전환 시 (`on_series_switched`)
- landmark는 유지 (같은 voxel grid)
- LPS/RAS만 새 meta로 재계산
- crop widget은 새 bounds로 재설치

### 12.3 PolyData ↔ UnstructuredGrid 변환
`remove_cells`가 `UnstructuredGrid`를 반환하면 `extract_surface()`로 다시 PolyData로 변환 — STL 저장과 후속 PolyData 메서드 호환성 보장.

### 12.4 PyVista picking 호환성
`enable_*_picking`은 기본적으로 P 키 바인딩. 좌클릭으로 동작시키려면 **`left_clicking=True` 반드시 필요**. 두 picking 모드가 같은 이벤트 채널을 쓰므로 mutual exclusion.

### 12.5 Signal blocking 패턴
세션 복원 시 모든 슬라이더/체크박스/콤보를 `blockSignals(True)` → `setValue/setChecked` → `blockSignals(False)` 패턴으로 처리. 핸들러 cascading 차단.

### 12.6 DICOM geometry 없을 때
IPP/IOP 추출 실패해도 앱은 정상 동작. 단지:
- 테이블 LPS/RAS 셀에 `N/A` + 안내 툴팁
- CSV export는 grid 컬럼만
- JSON export는 `dicom_geometry_available: false` 표시

---

## 12.7 Multi-Series Fusion (한 환자의 여러 ST를 동시 렌더링)

같은 환자(=같은 폴더)에서 추출된 모든 CT series를 한 화면에 동시에 표시하는 모드. 골반(ST=2.5mm), 무릎(ST=0.625mm), 발목(ST=2.5mm) 같은 분할 스캔이 각자의 native spacing을 유지한 채 환자 좌표계 상 정확한 위치에 배치됨.

### 12.7.1 좌표계 정렬

각 series는 자체 `IPP_first`(첫 슬라이스 origin in LPS), `row_dir`, `col_dir`, `normal_dir`(IOP의 외적)을 가짐. grid → LPS 변환 행렬:

```
        ⎡ row_dir[0]  col_dir[0]  normal_dir[0]  IPP[0] ⎤
T_i  =  ⎢ row_dir[1]  col_dir[1]  normal_dir[1]  IPP[1] ⎥
        ⎢ row_dir[2]  col_dir[2]  normal_dir[2]  IPP[2] ⎥
        ⎣     0           0            0            1    ⎦
```

`base_series_index`(기본 0)의 시리즈가 **공용 좌표계의 기준**. 다른 시리즈 mesh는

```
T_composite = inv(T_base) @ T_i
mesh_i.transform(T_composite, inplace=True)
```

으로 변환해서 plotter에 add. base 시리즈는 identity라 transform 불필요.

### 12.7.2 파이프라인 흐름 (`_update_fused_meshes`)

```
for i, series in enumerate(all_series_data):
    grid = series_volume_grids[i]   ← lazy cache (pv.ImageData wrap)
    grid.point_data["masked"] = _compute_masked_values(image_hu)   ← Stage C 공용
    mesh = grid.contour([min_threshold], scalars="masked")
    mesh = _apply_smoothing(mesh)   ← smooth_combo
    mesh = _apply_stage_a(mesh)     ← Stage A per-series
    if i != base_idx:
        mesh.transform(T_base_inv @ T_i, inplace=True)
    plotter.add_mesh(mesh, color="ivory")
```

모든 series에 **공통**으로 적용되는 것:
- HU thresholds
- Smoothing 방식
- Stage C 파라미터(`opening_iterations`, `connectivity`)
- Stage A 파라미터(`mesh_cleanup_enabled`, `keep_largest_only`, `min_fragment_faces`)

series별 **독립**적으로 적용되는 것:
- Marching cubes contour (각자 native spacing)
- Smoothing/Stage A는 각 mesh 단위로 실행 (다른 series에 영향 X)

### 12.7.3 캐시: `series_volume_grids[i]`

각 series의 `pv.ImageData` 객체는 환자 로드 시점에 첫 사용 직전 lazy-build, 그 다음부터는 메모리에 보관. threshold 변경 시 `point_data["masked"]`만 갈아끼우고 `contour()`를 다시 호출. 환자 교체 시 `on_load_clicked`에서 비움.

### 12.7.4 Mutual exclusion

Fusion 모드에서 비활성화되는 기능:
- **Stage B click-to-remove** — picked_point가 어느 series에 속하는지 추적이 추가로 필요 (`pick_btn` disable)

Fusion 모드에서 **동작하는** 기능:
- **3D Cropping** — base grid frame; 모든 fused mesh에 동일 `clip_box`
- **Bone Separation** — 포함된 series마다 native grid에서 분리 후 base frame으로 transform (§4.5.4)
- **Landmark picking / hover** — base frame 기준

### 12.7.5 Landmark picking 호환

`self.current_meta_info`는 항상 **base series의 meta**를 가리킴(`on_series_switched`가 설정). 모든 transformed mesh가 base series의 grid 공간에 정렬되어 있으므로:
- picked_point = base grid 좌표 (단일 모드와 동일한 의미)
- `_grid_to_lps(picked_point)`는 base meta로 변환 → 정확한 LPS
- sphere visualization은 base grid 좌표 그대로 사용

**Base series가 바뀌면** (`series_combo` 변경) landmark grid 좌표는 새 frame과 호환되지 않으므로 `_invalidate_landmarks_on_base_change`가 자동으로 비우고 사용자에게 알림.

### 12.7.6 STL Export

Fusion ON일 때: 모든 visible mesh를 `mesh.merge()`로 단일 PolyData로 합쳐 하나의 STL 파일로 저장. 리포트에 fusion에 포함된 series 목록 기록.

### 12.7.7 IOP 불일치 처리

서로 다른 series의 `ImageOrientationPatient`가 다르면(예: 하나는 axial, 다른 하나는 sagittal) 위 transform이 비-cubic 정렬을 만들어 PyVista에서 mesh가 회전된 채 합쳐짐. Mako 프로토콜은 보통 모두 axial이라 안전. DICOM geometry가 아예 누락된 series는 fusion에서 제외(콘솔에 경고).

---

## 13. 의존성 요약 (`requirements.txt`)

| 패키지 | 용도 |
|---|---|
| PyQt5 | GUI |
| pydicom | DICOM 파싱 |
| numpy | 배열 연산 |
| scipy | `ndimage.binary_opening`, `generate_binary_structure` (Stage C) |
| scikit-image | 향후 morphology 확장 여지 (현재 직접 사용은 없음, transitive로 유지) |
| pyvista | 3D 메쉬, 렌더링, picking |
| pyvistaqt | Qt embed (QtInteractor) |
| vtk | 저수준 picking (vtkCellPicker, MouseMoveEvent observer) |
| matplotlib | Scout Viewer |

---

## 14. 주요 함수 인덱스 (`app/mixins/` + `app/main_window.py`)

| 기능 | 함수 |
|---|---|
| 메쉬 생성 (단일/fusion 분기) | `update_base_mesh`, `_update_single_mesh`, `_update_fused_meshes` |
| 메쉬 헬퍼 (공용) | `_build_image_data`, `_compute_masked_values`, `_apply_smoothing`, `_apply_stage_a` |
| Fusion 좌표 정렬 | `_series_grid_to_lps_matrix`, `_invalidate_landmarks_on_base_change` |
| Fusion 토글 | `on_fusion_toggled` |
| 환자 로드 | `on_load_clicked`, `on_series_switched` |
| 파티클 제거 Stage C/A | `_compute_masked_values`, `_apply_stage_a` |
| 파티클 제거 Stage B | `on_picking_toggled`, `on_point_picked`, `on_undo_clicked` |
| Bone Separation 알고리즘 | `_compute_separation_labels`, `_build_separated_bone_meshes`, `_crop_bounds_in_series_grid`, `_apply_crop_to_bone_mask`, `_bone_color_palette`, `_bounds_to_voxel_slice` |
| Bone Separation fusion | `_compute_fusion_separated_bone_entries`, `_compute_single_separated_bone_entries`, `_show_separated_bones`, `_hide_combined_mesh_for_separation`, `_restore_combined_mesh_after_separation` |
| Bone Separation 핸들러 | `on_separate_bones_clicked`, `on_clear_separation_clicked`, `on_merge_bones_clicked`, `on_export_separated_bones_stl`, `on_rename_bone_clicked`, `_refresh_separation_list`, `_auto_clear_separation_on_remesh` |
| Cropping | `on_crop_toggled`, `on_crop_reset_clicked`, `on_box_cropped` |
| Landmark 클릭 | `on_landmark_picking_toggled`, `on_landmark_picked` |
| Hover preview | `_enable_hover_preview`, `_on_hover_mouse_move` |
| 좌표 변환 | `_grid_to_lps`, `_lps_to_ras` |
| 측정 | `_on_landmark_selection_changed`, `_draw_segment`, `_draw_angle_marker` |
| Landmark export | `on_landmark_export_clicked`, `_export_landmarks_csv`, `_export_landmarks_json` |
| Session 저장 | `on_save_session_clicked`, `_collect_session_state` |
| Session 복원 | `on_load_session_clicked`, `_apply_session_state` |
| STL export | `on_export_stl_clicked` |

---

## 15. 향후 확장 후보

- **Phase 3 — 클릭한 sphere 드래그 이동**: `vtkSphereWidget` 또는 custom drag observer
- **DICOM RT-STRUCT export**: `rt-utils` 라이브러리 활용 (임상 통합용)
- **다중 박스 누적 삭제** (inverse crop): 베드/금속 artifact 제거
- **자동 CT 테이블 제거**: `PatientPosition` 휴리스틱 기반
- **Landmark 좌표 저장 시 측정 결과도 함께 export**
- **Stage C 미리보기 캐싱**: `(min, max, iter, conn) → mask` 결과를 LRU 캐시로 보관해 슬라이더 왕복 시 즉시 반응
- **Fusion 모드에서 Click-to-Remove / Cropping 활성화**: picked actor → series index 매핑 추적, base frame 기준 box clip을 모든 series에 분배
- **시리즈별 visibility / 색상 토글**: fusion 화면에서 특정 series만 강조하거나 숨기기
- **IOP 불일치 자동 보정**: 다른 orientation의 series를 base frame으로 reslice

---

## 16. 변경 로그

### v3 (현재) — 2026-05-18
- **Multi-Series Fusion**: 한 환자의 모든 CT series를 동시 렌더링하는 모드 추가. 기본 ON.
- `pv.ImageData per series` 캐시 (`series_volume_grids`), `T_i = grid→LPS` 행렬을 통한 base grid frame 정렬, `mesh.transform(inv(T_base) @ T_i)`로 환자 좌표계 정합.
- `update_base_mesh()` 디스패치 + 공용 헬퍼 추출 (`_compute_masked_values`, `_apply_smoothing`, `_apply_stage_a`, `_build_image_data`, `_series_grid_to_lps_matrix`).
- `Fuse all series` 체크박스, fusion ON 시 Click-to-Remove / 3D Cropping mutual-exclusion.
- STL export: fusion 시 모든 series를 단일 STL로 merge.
- Session schema v2 → v3 (`fusion` 블록 추가, backward-compat).

### v2 — 2026-05-18
- **Stage C** 알고리즘: `skimage.morphology.remove_small_objects` (size-based connected component) → `scipy.ndimage.binary_opening` (morphological opening)
- **이유**: 본체와 연결된 얇은 가지/판/그물 노이즈가 connected component 단위에서는 분리되지 않아 제거 불가했음. opening은 굵기 기준으로 동작해 이런 구조를 잘라낼 수 있음.
- **UI**: `Min volume (mm³)` 단일 spinbox → `Iterations (0~5)` + `Connectivity (6/18/26)` 두 컨트롤
- **Session schema**: v1 → v2. `min_particle_volume_mm3` 삭제, `opening_iterations` + `opening_connectivity` 추가. v1 세션 로드 시 기본값으로 자동 마이그레이션.
- **의존성**: `scipy==1.13.1` 명시적 추가 (이전엔 scikit-image transitive).

### v1
- 3-Stage 파티클 제거 (Stage C: `remove_small_objects`, Stage A: mesh connectivity, Stage B: click-to-remove)
- Landmark picking, LPS/RAS 변환, hover preview
- 거리/각도 측정, CSV/JSON export
- Session save/load
- `QScrollArea` + `CollapsibleSection` UI

