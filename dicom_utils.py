"""
dicom_utils.py — DICOM 로딩 및 HU 변환 유틸리티

데이터 흐름 순서:
  ① load_scan(path)            DICOM 파일 읽기 → Scout 분리 → 시리즈 그룹핑 → z-gap 분할
  ② get_series_info(dict)      UI 표시용 시리즈 메타데이터 요약
  ③ load_series(dict, key)     단일 시리즈 z-정렬 + spacing(z,y,x) 계산
  ④ get_pixels_hu(slices)      Raw pixel → HU 변환 + 3D numpy 스택

Spacing 컨벤션:
  이 모듈에서 반환하는 spacing은 항상 (z, y, x) 순서.
  PyVista ImageData에 전달할 때 (x, y, z) 순으로 뒤집어야 함 — _build_image_data() 참고.
"""

import os
import numpy as np
import pydicom


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _slice_z_position(ds):
    """슬라이스의 stacking-axis 위치를 계산.

    ImageOrientationPatient에서 image plane의 법선 벡터를 구한 뒤,
    ImagePositionPatient를 그 법선에 투영하여 slice 위치를 반환.
    Axial / Coronal / Sagittal 관계없이 동작.

    DICOM 태그가 없으면 InstanceNumber로 fallback.
    """
    if "ImageOrientationPatient" not in ds or "ImagePositionPatient" not in ds:
        return float(ds.get("InstanceNumber", 0))

    iop = ds.ImageOrientationPatient
    ipp = ds.ImagePositionPatient

    row_vec = np.array(iop[:3])
    col_vec = np.array(iop[3:])
    normal = np.cross(row_vec, col_vec)

    return float(np.dot(normal, np.array(ipp)))


def _split_by_z_gap(slices, slice_thickness):
    """z-정렬된 슬라이스 리스트를 큰 z-gap 기준으로 sub-stack으로 분할.

    실제 사례: Mako 프로토콜에서 pelvis + ankle처럼 먼 부위 2개가
    하나의 SeriesInstanceUID에 합쳐진 경우, 이 함수가 둘을 분리하여
    marching cubes가 두 영역 사이에 가짜 geometry를 생성하는 것을 방지.

    반환: [(sub_slices, sub_z_positions), ...] 리스트
    """
    if len(slices) <= 1:
        return [(slices, [_slice_z_position(s) for s in slices])]

    z_positions = [_slice_z_position(s) for s in slices]
    z_diffs = np.abs(np.diff(z_positions))

    # 중앙값 spacing 계산 (동일 위치 쌍 무시)
    nonzero = z_diffs[z_diffs > 0.01]
    if len(nonzero) == 0:
        return [(slices, z_positions)]
    median_step = float(np.median(nonzero))

    # Gap 판별: 중앙값의 3배 또는 slice thickness의 2.5배 중 큰 값 초과 시
    gap_threshold = max(
        3.0 * median_step,
        2.5 * float(slice_thickness) if slice_thickness > 0 else 5.0,
    )
    gap_indices = [i for i, d in enumerate(z_diffs) if d > gap_threshold]

    if not gap_indices:
        return [(slices, z_positions)]

    sub_stacks = []
    starts = [0] + [g + 1 for g in gap_indices]
    ends = [g + 1 for g in gap_indices] + [len(slices)]
    for s, e in zip(starts, ends):
        sub_stacks.append((slices[s:e], z_positions[s:e]))
    return sub_stacks


# ──────────────────────────────────────────────────────────────────────
# ① load_scan — DICOM 파일 읽기 + 시리즈 그룹핑
# ──────────────────────────────────────────────────────────────────────

def load_scan(path):
    """디렉토리 내 모든 DICOM 파일을 읽고 시리즈별로 그룹핑.

    처리 순서:
      1) Scout/Localizer 분리 (ImageType 태그 확인)
      2) Volume 슬라이스를 (SeriesInstanceUID, SliceThickness) 기준 그룹핑
      3) 각 그룹 내 z-정렬 후, 큰 z-gap으로 sub-stack 분할

    Returns
    -------
    series_dict : dict
        key = (uid, slice_thickness, sub_idx), value = z-정렬된 슬라이스 리스트
    scout_slices : list
        Localizer/Scout 슬라이스들
    """
    raw_slices = [
        pydicom.dcmread(os.path.join(path, f))
        for f in os.listdir(path) if f.endswith('.dcm')
    ]

    # Phase 1: Scout와 Volume 분리 + (uid, slice_thickness) 그룹핑
    by_uid_st = {}
    scout_slices = []
    for s in raw_slices:
        image_type = getattr(s, "ImageType", [])
        if "LOCALIZER" in image_type or "SCOUT" in image_type:
            scout_slices.append(s)
            continue

        uid = s.SeriesInstanceUID
        st = getattr(s, 'SliceThickness', None)
        st_val = float(st) if st is not None else 0.0
        by_uid_st.setdefault((uid, st_val), []).append(s)

    # Phase 2: 각 그룹 z-정렬 → z-gap sub-stack 분할
    series_dict = {}
    for (uid, st_val), group_slices in by_uid_st.items():
        group_slices.sort(key=_slice_z_position)
        sub_stacks = _split_by_z_gap(group_slices, st_val)
        if len(sub_stacks) > 1:
            print(
                f"[load_scan] UID …{uid[-12:]} ST={st_val}mm → "
                f"{len(sub_stacks)} sub-volumes "
                f"(sizes: {[len(ss) for ss, _ in sub_stacks]})"
            )
        for sub_idx, (sub_slices, _) in enumerate(sub_stacks):
            series_dict[(uid, st_val, sub_idx)] = sub_slices

    return series_dict, scout_slices


# ──────────────────────────────────────────────────────────────────────
# ② get_series_info — UI용 시리즈 메타데이터 요약
# ──────────────────────────────────────────────────────────────────────

def get_series_info(series_dict):
    """각 시리즈의 UI 표시용 요약 정보를 반환.

    ≤3 슬라이스 시리즈(dose report 등)는 필터링.

    Returns
    -------
    list[dict]
        key, uid, sub_idx, count, description, slice_thickness, z_min, z_max 등
    """
    info_list = []
    for key, slices in series_dict.items():
        if len(slices) <= 3:
            continue
        uid, st_val, sub_idx = key
        sample = slices[0]
        z0 = float(_slice_z_position(slices[0]))
        z1 = float(_slice_z_position(slices[-1]))
        info_list.append({
            'key': key,
            'uid': uid,
            'sub_idx': sub_idx,
            'count': len(slices),
            'description': getattr(sample, 'SeriesDescription', 'N/A'),
            'patient_name': str(getattr(sample, 'PatientName', 'N/A')),
            'study_date': getattr(sample, 'StudyDate', 'N/A'),
            'modality': getattr(sample, 'Modality', 'N/A'),
            'slice_thickness': st_val,
            'z_min': min(z0, z1),
            'z_max': max(z0, z1),
        })
    info_list.sort(key=lambda x: (x['uid'], x['z_min']))
    return info_list


# ──────────────────────────────────────────────────────────────────────
# ③ load_series — 단일 시리즈 z-정렬 + spacing 계산
# ──────────────────────────────────────────────────────────────────────

def load_series(series_dict, series_key):
    """선택된 시리즈를 z-정렬하고 물리적 spacing을 계산.

    z-정렬은 load_scan에서 이미 수행되지만, cache 미사용 경로에서도
    안전하게 동작하도록 여기서 다시 한 번 수행.

    Returns
    -------
    slices : list[pydicom.Dataset]
        z-정렬된 슬라이스
    spacing : tuple[float, float, float]
        (z_spacing, y_spacing, x_spacing) in mm
    """
    slices = series_dict[series_key]
    slices.sort(key=_slice_z_position)

    # z-spacing: 인접 슬라이스 간 거리의 중앙값
    try:
        z_positions = [_slice_z_position(s) for s in slices]
        z_diffs = np.abs(np.diff(z_positions))
        valid_diffs = z_diffs[z_diffs > 0.01]
        if len(valid_diffs) > 0:
            z_spacing = float(np.median(valid_diffs))
            std_dev = float(np.std(valid_diffs))
            if std_dev > 0.1 * z_spacing:
                print(
                    f"  [WARNING] Non-uniform z-spacing: "
                    f"median={z_spacing:.3f}mm, std={std_dev:.3f}mm"
                )
        else:
            z_spacing = float(slices[0].SliceThickness)
    except Exception:
        z_spacing = float(slices[0].SliceThickness)

    # xy-spacing: PixelSpacing 태그 (row_spacing, col_spacing)
    try:
        ps = slices[0].PixelSpacing
        y_spacing, x_spacing = float(ps[0]), float(ps[1])
    except Exception:
        y_spacing, x_spacing = 1.0, 1.0

    spacing = (z_spacing, y_spacing, x_spacing)
    return slices, spacing


# ──────────────────────────────────────────────────────────────────────
# ④ get_pixels_hu — Raw pixel → HU 변환 + 3D 스택
# ──────────────────────────────────────────────────────────────────────

def get_pixels_hu(slices):
    """DICOM 슬라이스들의 Raw pixel 값을 Hounsfield Unit(HU)으로 변환.

    변환 공식: HU = slope × raw_pixel + intercept
    (DICOM 태그: RescaleSlope, RescaleIntercept)

    대부분의 CT 시리즈는 모든 슬라이스에서 slope=1, intercept=-1024로 동일하므로
    uniform한 경우 벡터 연산 한번으로 처리 (성능 최적화).

    Returns
    -------
    np.ndarray
        shape (nz, ny, nx), dtype int16, 값은 HU
    """
    # 3D 스택: (nz, ny, nx)
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    # 모든 슬라이스의 slope/intercept 수집
    slopes = np.array([float(getattr(s, 'RescaleSlope', 1)) for s in slices])
    intercepts = np.array([float(getattr(s, 'RescaleIntercept', 0)) for s in slices])

    # Fast path: 모든 슬라이스가 동일한 slope/intercept (대부분의 CT)
    if np.all(slopes == slopes[0]) and np.all(intercepts == intercepts[0]):
        slope, intercept = slopes[0], intercepts[0]
        if slope != 1:
            image = (slope * image.astype(np.float64)).astype(np.int16)
        image += np.int16(intercept)
    else:
        # Slow path: 슬라이스마다 다른 경우 (드문 케이스)
        for i in range(len(slices)):
            if slopes[i] != 1:
                image[i] = (slopes[i] * image[i].astype(np.float64)).astype(np.int16)
            image[i] += np.int16(intercepts[i])

    return image
