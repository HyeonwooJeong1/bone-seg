"""Voxel-level math kernels shared by Restore and Merge & Fill.

Methods grouped here:
  - _build_other_bones_exclusion: HU CC + Voronoi exclusion of other bones
  - _compute_voronoi_territory: global Voronoi (target vs other) over local box
  - _voxelize_mesh_interior: rasterize+fill mesh to a binary solid mask
  - _hu_region_grow: HU-threshold region growing from a seed mask
  - _opposing_axes_fill: legacy 3-axis hole filling (kept available)

These are pure-math helpers — no Qt/UI calls.
"""

import numpy as np
from scipy.ndimage import (binary_dilation, binary_fill_holes,
                           binary_opening, distance_transform_edt,
                           generate_binary_structure,
                           label as ndi_label)


class BoneVoxelOpsMixin:
    def _build_other_bones_exclusion(self, target_uids, local_shape,
                                       spacing, offsets,
                                       working_series_idx=None,
                                       local_hu=None, threshold=None):
        """target_uids에 포함되지 않는 다른 모든 뼈의 voxel exclusion mask 생성.

        ★ HU connected component 기반 (local_hu, threshold가 주어진 경우):
          각 뼈의 mesh seed가 속한 HU connected component를 정확히 식별.
          - "Only other" CC (target과 공유 안 함): CC 전체를 exclusion
          - "Shared" CC (target도 닿음, touching bones): Voronoi (distance transform)로
            target seed보다 other seed에 더 가까운 voxel만 exclusion
          1-voxel safety buffer 적용 (단, target seed/CC는 침범 금지).

          이 방식은 mesh shell + fill_holes의 다음 문제를 해결:
            - 큰 뼈의 sparse mesh vertex 때문에 shell에 gap → fill_holes 실패 →
              thin shell만 남고 region grow가 누출
            - 30× safety check가 큰 뼈에서 자주 트리거되어 thin shell로 revert

        Legacy fallback (local_hu/threshold가 None인 경우):
          이전 mesh shell + fill_holes 방식. 사용 비권장.

        Parameters
        ----------
        target_uids : iterable
            제외할 뼈의 uid 집합 (target bones — 자신은 exclusion에서 빠짐)
        local_shape : (lz, ly, lx)
        spacing : (sz, sy, sx)
        offsets : (iz0, iy0, ix0)
            local 배열의 global voxel 시작 위치
        working_series_idx : int or None
            작업 좌표계 시리즈. None/base → mesh.points 직접 사용.
            다른 값 → 점들을 해당 시리즈 grid로 변환.
        local_hu : ndarray (lz, ly, lx) int16 or None
            local 영역의 HU 값. None이면 legacy 모드.
        threshold : float or None
            HU bone threshold. None이면 legacy 모드.
        """
        sz, sy, sx = spacing
        iz0, iy0, ix0 = offsets
        lz, ly, lx = local_shape
        target_uids = set(target_uids)

        # 좌표 변환 준비 (working grid가 base가 아닌 경우)
        need_transform = (
            working_series_idx is not None
            and working_series_idx != self.base_series_index
            and getattr(self, 'fusion_enabled', False)
        )
        T_to_working = None
        if need_transform:
            work_meta = self.all_series_data[working_series_idx].get('meta') or {}
            base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
            T_base = self._series_grid_to_lps_matrix(base_meta)
            T_work = self._series_grid_to_lps_matrix(work_meta)
            if T_base is not None and T_work is not None:
                # base grid → working series grid
                T_composite = np.linalg.inv(T_base) @ T_work
                T_to_working = np.linalg.inv(T_composite)
            else:
                need_transform = False

        struct26 = generate_binary_structure(3, 3)

        def _pts_to_local_idx(pts):
            """mesh points → local voxel index (clipped to local box)."""
            if need_transform and T_to_working is not None:
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                pts = (T_to_working @ pts_h.T).T[:, :3]
            vix = np.round(pts[:, 0] / sx).astype(int) - ix0
            viy = np.round(pts[:, 1] / sy).astype(int) - iy0
            viz = np.round(pts[:, 2] / sz).astype(int) - iz0
            valid = ((vix >= 0) & (vix < lx) &
                     (viy >= 0) & (viy < ly) &
                     (viz >= 0) & (viz < lz))
            return vix[valid], viy[valid], viz[valid]

        # ── HU CC + Voronoi 모드 (정확) ──
        if local_hu is not None and threshold is not None:
            local_mask = local_hu >= threshold
            labeled_hu, n_cc = ndi_label(local_mask, structure=struct26)

            target_seed = np.zeros((lz, ly, lx), dtype=bool)
            other_seed = np.zeros((lz, ly, lx), dtype=bool)
            target_labels = set()
            other_labels = set()

            for bone in self.separated_bones:
                mesh = bone.get('mesh')
                if mesh is None or mesh.n_points == 0:
                    continue
                is_target = bone.get('uid') in target_uids
                # invisible others는 건너뛰지만, target은 visibility 무관하게 처리
                if not is_target and not bone.get('visible', True):
                    continue

                vix, viy, viz = _pts_to_local_idx(mesh.points)
                if len(vix) == 0:
                    continue

                if is_target:
                    target_seed[viz, viy, vix] = True
                else:
                    other_seed[viz, viy, vix] = True

                if n_cc > 0:
                    labels = set(labeled_hu[viz, viy, vix].tolist())
                    labels.discard(0)
                    if is_target:
                        target_labels |= labels
                    else:
                        other_labels |= labels

            if not other_seed.any():
                return np.zeros((lz, ly, lx), dtype=bool)

            exclusion = np.zeros((lz, ly, lx), dtype=bool)

            # 1) Only-other CC: 전체 CC를 exclusion
            only_other = other_labels - target_labels
            if only_other:
                exclusion |= np.isin(labeled_hu, list(only_other))

            # 2) Shared CC (touching bones): Voronoi로 분할
            shared = other_labels & target_labels
            if shared:
                shared_mask = np.isin(labeled_hu, list(shared))
                # distance_transform_edt: True → 거리 계산 대상, 0이 가까울수록 작음
                # ~seed가 True인 곳에서 가장 가까운 False(=seed)까지의 거리
                d_target = distance_transform_edt(
                    ~target_seed, sampling=(sz, sy, sx)
                ) if target_seed.any() else np.full(local_shape, np.inf)
                d_other = distance_transform_edt(
                    ~other_seed, sampling=(sz, sy, sx)
                )
                # shared 안에서 other seed가 더 가까운 voxel만 exclusion
                voronoi_excl = shared_mask & (d_other < d_target)
                exclusion |= voronoi_excl

            # 3) 1-voxel safety buffer, 단 target은 침범 금지
            if exclusion.any():
                target_protected = target_seed.copy()
                if target_labels:
                    target_protected |= np.isin(labeled_hu, list(target_labels))
                exclusion_buf = binary_dilation(exclusion, structure=struct26,
                                                iterations=1)
                # target 영역으로 buffer가 침범하면 그 부분만 제거
                exclusion = exclusion_buf & ~target_protected

            # 4) 절대 보호: target seed 좌표는 어떤 경우에도 exclusion에 포함되지 않음
            #    (bridge 영역 등 HU < threshold인 seed 점이 sparse할 때 buffer에 잡히는 것 방지)
            if target_seed.any():
                exclusion = exclusion & ~target_seed

            return exclusion

        # ── Legacy: mesh shell + fill_holes (fallback) ──
        exclusion = np.zeros((lz, ly, lx), dtype=bool)

        for bone in self.separated_bones:
            if bone.get('uid') in target_uids:
                continue
            if not bone.get('visible', True):
                continue
            mesh = bone.get('mesh')
            if mesh is None or mesh.n_points == 0:
                continue

            vix, viy, viz = _pts_to_local_idx(mesh.points)
            if len(vix) > 0:
                bone_seed = np.zeros((lz, ly, lx), dtype=bool)
                bone_seed[viz, viy, vix] = True
                bone_shell = binary_dilation(bone_seed, structure=struct26,
                                             iterations=1)
                bone_filled = binary_fill_holes(bone_shell)
                if bone_filled.sum() > bone_shell.sum() * 30:
                    bone_filled = bone_shell
                exclusion |= bone_filled

        if exclusion.any():
            exclusion = binary_dilation(exclusion, structure=struct26,
                                        iterations=1)

        return exclusion

    def _compute_voronoi_territory(self, target_uids, local_shape,
                                     spacing, offsets,
                                     working_series_idx=None,
                                     extra_other_seed=None):
        """Global Voronoi territory: local box 내 각 voxel에 대해
        '가장 가까운 mesh가 target 뼈인지 other 뼈인지' 판정.

        Parameters
        ----------
        extra_other_seed : ndarray (bool) or None
            추가 other seed (예: 미분리된 뼈 voxel). self.separated_bones에 등록 안 된
            구조도 Voronoi에서 'other'로 잡아내기 위해 사용. None이면 mesh-only.

        반환값:
          - target_territory : ndarray (bool)
              True인 voxel = target 뼈 mesh가 다른 뼈 mesh보다 더 가까움.
              (other_seed가 비어있으면 모든 voxel이 True)
          - has_other        : bool — other_seed가 하나라도 있었으면 True.
              False면 territory 제약을 적용할 의미가 없음(전부 target territory).

        이 함수는 _build_other_bones_exclusion과 별개로, gap_fill/restore 결과를
        '다른 뼈에 더 가까운 영역'으로부터 강제로 분리하기 위한 strict 기하 제약.
        HU/CC와 무관하게 mesh 좌표만으로 판정 → soft-tissue 누출 차단에 효과적.
        """
        sz, sy, sx = spacing
        iz0, iy0, ix0 = offsets
        lz, ly, lx = local_shape
        target_uids = set(target_uids)

        # 좌표 변환 준비 (working grid가 base가 아닌 경우)
        need_transform = (
            working_series_idx is not None
            and working_series_idx != self.base_series_index
            and getattr(self, 'fusion_enabled', False)
        )
        T_to_working = None
        if need_transform:
            work_meta = self.all_series_data[working_series_idx].get('meta') or {}
            base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
            T_base = self._series_grid_to_lps_matrix(base_meta)
            T_work = self._series_grid_to_lps_matrix(work_meta)
            if T_base is not None and T_work is not None:
                T_composite = np.linalg.inv(T_base) @ T_work
                T_to_working = np.linalg.inv(T_composite)
            else:
                need_transform = False

        def _pts_to_local_idx(pts):
            if need_transform and T_to_working is not None:
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                pts = (T_to_working @ pts_h.T).T[:, :3]
            vix = np.round(pts[:, 0] / sx).astype(int) - ix0
            viy = np.round(pts[:, 1] / sy).astype(int) - iy0
            viz = np.round(pts[:, 2] / sz).astype(int) - iz0
            valid = ((vix >= 0) & (vix < lx) &
                     (viy >= 0) & (viy < ly) &
                     (viz >= 0) & (viz < lz))
            return vix[valid], viy[valid], viz[valid]

        target_seed = np.zeros((lz, ly, lx), dtype=bool)
        other_seed = np.zeros((lz, ly, lx), dtype=bool)

        for bone in self.separated_bones:
            mesh = bone.get('mesh')
            if mesh is None or mesh.n_points == 0:
                continue
            is_target = bone.get('uid') in target_uids
            # invisible others는 territory 계산에서도 제외 (사용자가 숨긴 뼈는 영향 X)
            if not is_target and not bone.get('visible', True):
                continue
            vix, viy, viz = _pts_to_local_idx(mesh.points)
            if len(vix) == 0:
                continue
            if is_target:
                target_seed[viz, viy, vix] = True
            else:
                other_seed[viz, viy, vix] = True

        # 추가 other seed (예: 미분리된 뼈 voxel) 병합
        if extra_other_seed is not None and extra_other_seed.shape == (lz, ly, lx):
            other_seed |= extra_other_seed

        has_other = bool(other_seed.any())
        if not has_other:
            return np.ones((lz, ly, lx), dtype=bool), False

        if not target_seed.any():
            # Target mesh 점이 모두 local box 밖이라면 territory 계산 의미 없음.
            # 안전쪽으로 'target territory 없음' = 모두 False 반환 → 호출자가 처리.
            return np.zeros((lz, ly, lx), dtype=bool), True

        d_target = distance_transform_edt(~target_seed, sampling=(sz, sy, sx))
        d_other = distance_transform_edt(~other_seed, sampling=(sz, sy, sx))
        # 동률은 target에게 양보 (boundary voxel 보존)
        target_territory = d_target <= d_other
        return target_territory, True

    def _voxelize_mesh_interior(self, mesh, local_shape, spacing, offsets,
                                  working_series_idx=None):
        """Mesh의 closed surface 내부를 binary mask로 voxelize.

        mesh.points (base grid mm) → local voxel index → 1-voxel dilation →
        binary_fill_holes로 내부 채움.

        이 함수의 핵심: 각 뼈의 "진짜 solid volume"을 mesh 자체에서만 정의함.
        HU CC 라벨에 의존하지 않으므로:
          - shared CC 문제 (두 뼈가 threshold에서 같은 CC에 연결) → 무관
          - 미분리 뼈 (separated_bones에 없음) → 자동으로 제외됨

        반환: binary mask (lz, ly, lx). mesh가 local box 밖이면 전부 False.

        Safety: fill_holes가 mesh의 큰 구멍을 통해 폭주하여 local box의
        95% 이상을 채우면 fallback으로 dilated surface만 반환.
        """
        sz, sy, sx = spacing
        iz0, iy0, ix0 = offsets
        lz, ly, lx = local_shape

        # 좌표 변환 (working grid가 base가 아닌 경우)
        pts = mesh.points
        need_transform = (
            working_series_idx is not None
            and working_series_idx != self.base_series_index
            and getattr(self, 'fusion_enabled', False)
        )
        if need_transform:
            work_meta = self.all_series_data[working_series_idx].get('meta') or {}
            base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
            T_base = self._series_grid_to_lps_matrix(base_meta)
            T_work = self._series_grid_to_lps_matrix(work_meta)
            if T_base is not None and T_work is not None:
                T_composite = np.linalg.inv(T_base) @ T_work
                T_to_working = np.linalg.inv(T_composite)
                pts_h = np.hstack([pts, np.ones((len(pts), 1))])
                pts = (T_to_working @ pts_h.T).T[:, :3]

        vix = np.round(pts[:, 0] / sx).astype(int) - ix0
        viy = np.round(pts[:, 1] / sy).astype(int) - iy0
        viz = np.round(pts[:, 2] / sz).astype(int) - iz0
        valid = ((vix >= 0) & (vix < lx) &
                 (viy >= 0) & (viy < ly) &
                 (viz >= 0) & (viz < lz))
        vix, viy, viz = vix[valid], viy[valid], viz[valid]

        if len(vix) == 0:
            return np.zeros((lz, ly, lx), dtype=bool)

        seed = np.zeros((lz, ly, lx), dtype=bool)
        seed[viz, viy, vix] = True

        struct26 = generate_binary_structure(3, 3)
        # 1-voxel dilation으로 marching cubes 표면의 작은 gap 닫기
        surf = binary_dilation(seed, structure=struct26, iterations=1)

        # 내부 채움
        interior = binary_fill_holes(surf)

        # 안전: fill_holes가 local box의 95% 이상을 채웠으면 mesh가 열려있다는 뜻 → fallback
        box_size = lz * ly * lx
        if interior.sum() > 0.95 * box_size:
            return surf

        return interior

    def _hu_region_grow(self, bone_mask, local_hu, exclusion,
                        hu_threshold, max_iters):
        """HU 기반 Region Growing — CT 원본 값으로 뼈 경계를 자연스럽게 확장.

        현재 뼈에서 시작하여 인접 1 voxel씩 확장하되,
        해당 voxel의 (smoothed) HU >= hu_threshold 인 경우에만 추가.
        완료 후 내부 hole 메움 + 표면 noise 제거로 매끈한 결과 보장.

        Parameters
        ----------
        bone_mask : ndarray (bool)   현재 뼈 voxel mask (시작점)
        local_hu  : ndarray (int16)  원본 HU 값 배열 (동일 shape)
        exclusion : ndarray (bool)   접근 금지 영역 (다른 뼈)
        hu_threshold : float         이 HU 이상인 voxel만 뼈로 추가
        max_iters : int              최대 성장 반복 횟수 (= 최대 성장 거리 voxel)

        Returns
        -------
        ndarray (bool) — 성장된 결과 (smoothed)
        """
        struct6 = generate_binary_structure(3, 1)   # 6-connected

        # ── Step 1: Region Growing (원본 HU 그대로 사용, Gaussian 없음) ──
        # Gaussian smooth를 제거: 표면 바깥으로 HU가 번지는 것을 원천 차단.
        # 원본 HU만 쓰면 공기(낮은 HU) 경계에서 정확히 정지함.
        hu_valid = local_hu >= hu_threshold

        result = bone_mask.copy()
        total_added = 0
        for i in range(max_iters):
            # 현재 뼈 표면에 인접한 빈 voxel들
            surface = binary_dilation(result, structure=struct6,
                                       iterations=1) & ~result
            # smoothed HU 조건 + 다른 뼈 회피
            fillable = surface & hu_valid & ~exclusion
            n = int(fillable.sum())
            if n == 0:
                print(f"[HU-Grow] Converged at iter {i+1}, "
                      f"total +{total_added} vox")
                break
            result |= fillable
            total_added += n
        else:
            print(f"[HU-Grow] Reached max {max_iters} iters, "
                  f"total +{total_added} vox")

        # ── Step 3: 내부 hole 메움 (파이는 곳 제거) ──
        # binary_fill_holes: 완전히 둘러싸인 빈 공간만 채움 → 두꺼워지지 않음
        filled = binary_fill_holes(result)
        if filled.sum() > result.sum() * 50:
            # 누출 방지 (fill_holes가 너무 많이 채웠으면 원래 결과 유지)
            filled = result
        filled = filled & ~exclusion
        fill_added = int(filled.sum() - result.sum())
        if fill_added > 0:
            print(f"[HU-Grow] fill_holes: +{fill_added} internal vox")
        result = filled

        # ── Step 4: 표면 noise 제거 (주름 제거) ──
        # binary_opening: 1-voxel 돌출 제거 (erosion→dilation)
        # 깎기만 하므로 절대 두꺼워지지 않음
        opened = binary_opening(result, structure=struct6, iterations=1)
        # opening이 너무 많이 깎으면 원래 결과 유지 (10% 이상 손실 방지)
        if opened.sum() >= result.sum() * 0.9:
            noise_removed = int(result.sum() - opened.sum())
            if noise_removed > 0:
                print(f"[HU-Grow] opening: -{noise_removed} noise vox")
            result = opened
        else:
            print(f"[HU-Grow] opening skipped (too aggressive)")

        return result

    def _opposing_axes_fill(self, bone_mask, boundary_mask, exclusion,
                             max_distance, min_axes=2):
        """Opposing Axes hole filling (legacy, kept for Merge & Fill). — 표면 두께를 보존하면서 구멍만 채움.

        각 빈 voxel V에 대해, X/Y/Z 3개 축마다:
          - +방향으로 max_distance 안에 뼈 voxel이 있는지
          - -방향으로도 max_distance 안에 뼈 voxel이 있는지
        둘 다 만족하면 그 축은 "양쪽이 뼈로 둘러싸임" 상태.

        ≥ min_axes 개 축에서 양쪽 뼈가 확인된 voxel만 채움.

        효과:
          - 평면/볼록면 바깥 voxel: 1축만 한쪽에 뼈 → 0축 → 안 채움 (두께 보존)
          - 컵형 오목 안쪽: 2축에서 양쪽 → 채움
          - 터널/완전 갇힌 hole: 3축 모두 양쪽 → 채움

        Parameters
        ----------
        bone_mask : ndarray (bool)
            현재 뼈 voxel mask
        boundary_mask : ndarray (bool)
            채우기 허용 범위 (seed dilation)
        exclusion : ndarray (bool)
            접근 금지 영역 (다른 뼈)
        max_distance : int
            각 축에서 양쪽 뼈를 찾을 최대 거리 (voxel)
        min_axes : int (1~3)
            양쪽에 뼈가 있어야 하는 최소 축 개수

        Returns
        -------
        ndarray (bool) — 채워진 결과
        """
        fillable = boundary_mask & ~exclusion
        shape = bone_mask.shape
        max_distance = max(1, int(max_distance))
        min_axes = max(1, min(3, int(min_axes)))

        axes_with_both = np.zeros(shape, dtype=np.uint8)

        for axis in range(3):
            pos_hit = np.zeros(shape, dtype=bool)
            neg_hit = np.zeros(shape, dtype=bool)

            for k in range(1, max_distance + 1):
                # +방향: voxel V 기준 V+k 위치에 뼈가 있나
                #   → bone_mask를 -k만큼 shift하면 됨 (그래야 V 위치에서 V+k 값 보임)
                shifted_pos = np.roll(bone_mask, -k, axis=axis)
                # 경계 wrap 제거 (반대편으로 감기는 부분)
                slc = [slice(None)] * 3
                slc[axis] = slice(-k, None)
                shifted_pos[tuple(slc)] = False
                pos_hit |= shifted_pos

                # -방향: V-k 위치에 뼈
                shifted_neg = np.roll(bone_mask, k, axis=axis)
                slc = [slice(None)] * 3
                slc[axis] = slice(0, k)
                shifted_neg[tuple(slc)] = False
                neg_hit |= shifted_neg

            axes_with_both += (pos_hit & neg_hit).astype(np.uint8)

        candidates = (axes_with_both >= min_axes) & ~bone_mask & fillable
        added = int(candidates.sum())
        print(f"[Opposing] dist={max_distance} min_axes={min_axes} "
              f"→ +{added} vox filled")
        return bone_mask | candidates
