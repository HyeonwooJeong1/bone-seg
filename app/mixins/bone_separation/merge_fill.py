"""Merge & Fill: combine selected bones across series and fill the gap.

Methods grouped here:
  - on_merge_fill_clicked: top-level UI handler (creates __merge__ undo entry)
  - _resample_series_to_base_local: bring a non-base series HU into base grid
  - _merge_and_fill_voxel: compute composite HU, bone CCs, bridge fill, re-mesh
"""

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QInputDialog,
    QMessageBox,
)
from scipy.ndimage import binary_dilation, generate_binary_structure


class BoneMergeFillMixin:
    # =================================================================
    # Merge & Fill: 선택한 뼈들을 합치고 사이 빈 공간을 자연스럽게 채움
    # =================================================================

    def on_merge_fill_clicked(self):
        """뼈 목록에서 선택된 2개 이상의 뼈를 합치고 gap을 채움."""
        if not self.bone_separation_enabled:
            return
        items = self.bone_list_widget.selectedItems()
        if len(items) < 2:
            QMessageBox.information(
                self, "Merge & Fill",
                "뼈 목록에서 2개 이상 선택 (Ctrl+Click)한 후 클릭하세요."
            )
            return

        uids = [it.data(Qt.UserRole) for it in items]
        bones = [self._bone_by_uid(uid) for uid in uids]
        bones = [b for b in bones if b is not None and b['mesh'] is not None]
        if len(bones) < 2:
            return

        # 카메라 시점 저장
        cam_pos = self.plotter.camera_position

        self.separation_status_label.setText("Merging & filling gap…")
        try:
            QApplication.processEvents()
        except Exception:
            pass

        new_mesh = self._merge_and_fill_voxel(bones)

        if new_mesh is None or new_mesh.n_points == 0:
            self.separation_status_label.setText(
                "Merge & Fill failed. Try increasing fill iterations."
            )
            return

        # Undo: 원본 뼈들의 전체 상태를 __merge__ 형식으로 저장
        original_snapshots = []
        for bone in bones:
            original_snapshots.append({
                'uid': bone['uid'],
                'id': bone.get('id', 0),
                'mesh': bone['mesh'].copy(deep=True),
                'visible': bone.get('visible', True),
                'color': bone.get('color', (1, 1, 1)),
                'voxel_count': bone.get('voxel_count', 0),
                'name': bone['name'],
                'series_index': bone.get('series_index'),
                'raw_mesh': bone['raw_mesh'].copy(deep=True) if bone.get('raw_mesh') is not None else None,
            })

        merged_uid = self._new_bone_uid()

        # 기존 뼈 actors 제거
        uid_set = set(b['uid'] for b in bones)
        for bone in list(self.separated_bones):
            if bone['uid'] in uid_set:
                actor = bone.get('actor')
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
        self.separated_bones = [
            b for b in self.separated_bones if b['uid'] not in uid_set
        ]

        # 새 뼈 추가
        merged_names = [b['name'] for b in bones]
        default_name = " + ".join(merged_names[:3])
        if len(merged_names) > 3:
            default_name += f" (+{len(merged_names) - 3})"

        name, ok = QInputDialog.getText(
            self, "Merged Bone Name",
            "합쳐진 뼈 이름:", text=default_name,
        )
        merged_name = (name.strip() if ok and name.strip() else default_name)

        color = self._bone_color_palette(1)[0]
        actor = self.plotter.add_mesh(
            new_mesh, color=color, specular=0.5, smooth_shading=True,
            reset_camera=False,
        )
        self.separated_bones.append({
            'uid': merged_uid,
            'id': 0,
            'mesh': new_mesh,
            'raw_mesh': new_mesh.copy(deep=True),
            'actor': actor,
            'visible': True,
            'color': color,
            'voxel_count': new_mesh.n_cells,
            'name': merged_name,
            # Cross-series merge 결과는 base grid 좌표 → base series로 귀속
            'series_index': self.base_series_index,
        })

        # Undo 스택에 merge 항목 추가
        self._restore_undo_stack.append(
            ('__merge__', merged_uid, original_snapshots)
        )
        if len(self._restore_undo_stack) > self._max_restore_undo:
            self._restore_undo_stack = self._restore_undo_stack[-self._max_restore_undo:]
        self.restore_undo_btn.setEnabled(True)

        n_series = len(set(b.get('series_index') for b in bones))
        self._refresh_separation_list()
        self.separation_status_label.setText(
            f"Merged {len(bones)} bones ({n_series} series) → \"{merged_name}\" "
            f"({self.merge_fill_iterations} closing iter, "
            f"{new_mesh.n_cells} cells)"
        )
        # 카메라 시점 복원
        self.plotter.camera_position = cam_pos
        self.plotter.render()

    def _resample_series_to_base_local(self, series_idx, local_bounds,
                                        base_spacing):
        """시리즈 series_idx의 HU를 base grid의 local 영역으로 리샘플링.

        Parameters
        ----------
        series_idx : int
        local_bounds : tuple (ix0, ix1, iy0, iy1, iz0, iz1) in base grid
        base_spacing : tuple (sz, sy, sx)

        Returns
        -------
        np.ndarray (lz, ly, lx) int16 or None
        """
        if series_idx == self.base_series_index:
            return None  # base는 직접 슬라이싱하면 됨

        other_sd = self.all_series_data[series_idx]
        other_hu = other_sd['image_hu']
        other_spacing = other_sd['spacing']
        other_meta = other_sd.get('meta') or {}

        base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(other_meta)

        if T_base is None or T_i is None:
            return None

        # T_composite: series_i grid → base grid
        # T_inv: base grid → series_i grid
        T_composite = np.linalg.inv(T_base) @ T_i
        T_inv = np.linalg.inv(T_composite)

        ix0, ix1, iy0, iy1, iz0, iz1 = local_bounds
        sz, sy, sx = base_spacing
        osz, osy, osx = other_spacing
        lx = ix1 - ix0
        ly = iy1 - iy0
        lz = iz1 - iz0
        if lx <= 0 or ly <= 0 or lz <= 0:
            return None
        onz, ony, onx = other_hu.shape

        # Base grid 좌표 생성 (vectorized)
        x_coords = (np.arange(ix0, ix1, dtype=float)) * sx
        y_coords = (np.arange(iy0, iy1, dtype=float)) * sy
        z_coords = (np.arange(iz0, iz1, dtype=float)) * sz

        zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        base_coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel(),
                                np.ones(lx * ly * lz)], axis=1)  # (N, 4)

        # Base grid → series i grid 변환
        other_coords = (T_inv @ base_coords.T).T[:, :3]  # (N, 3)

        # Series i voxel 인덱스
        oix = np.round(other_coords[:, 0] / osx).astype(int)
        oiy = np.round(other_coords[:, 1] / osy).astype(int)
        oiz = np.round(other_coords[:, 2] / osz).astype(int)

        valid = ((oix >= 0) & (oix < onx) &
                 (oiy >= 0) & (oiy < ony) &
                 (oiz >= 0) & (oiz < onz))

        # 범위 밖은 threshold 이하 값으로 (기여하지 않음)
        fill_val = np.int16(self.current_min_threshold - 100)
        result = np.full(lx * ly * lz, fill_val, dtype=np.int16)
        if valid.any():
            result[valid] = other_hu[oiz[valid], oiy[valid], oix[valid]]

        return result.reshape(lz, ly, lx)

    def _merge_and_fill_voxel(self, bone_entries):
        """여러 뼈의 voxel 영역을 합치고 closing으로 gap을 메운 뒤 re-mesh.

        ★ Cross-series 지원:
          모든 bone mesh는 이미 base grid 좌표에 있음 (fusion transform 적용됨).
          base grid를 작업 공간으로 사용하고, 다른 시리즈의 HU는
          base grid로 리샘플링하여 합성 볼륨을 만듦.

        처리 순서:
          1) Base grid에서 합성 HU 생성 (base + resampled other series)
          2) Mesh vertex (이미 base grid) → seed → interior fill
          3) Exclusion mask (다른 뼈 차단)
          4) 26-conn closing → gap 메움
          5) Contour → 후처리 (결과는 base grid 좌표)
        """
        # ── Base series를 작업 공간으로 사용 ──
        base_idx = self.base_series_index
        base_sd = self.all_series_data[base_idx]
        base_hu = base_sd['image_hu']
        base_spacing = base_sd['spacing']

        sz, sy, sx = base_spacing
        nz, ny, nx = base_hu.shape
        iters = int(self.merge_fill_iterations)

        # 관련된 시리즈 수집
        involved_series = set()
        for bone in bone_entries:
            sidx = bone.get('series_index')
            if sidx is not None:
                involved_series.add(sidx)

        # 모든 뼈의 mesh vertex (이미 base grid 좌표)
        all_pts = []
        for bone in bone_entries:
            all_pts.append(bone['mesh'].points.copy())
        combined_pts = np.vstack(all_pts)

        # 통합 bounding box (base grid 좌표 기준, 클램핑 없이 전체 범위)
        # Cross-series 뼈가 base grid 밖에 있을 수 있으므로 클램핑하지 않음.
        pad = iters + 8
        ix0 = int(combined_pts[:, 0].min() / sx) - pad
        ix1 = int(np.ceil(combined_pts[:, 0].max() / sx)) + pad + 1
        iy0 = int(combined_pts[:, 1].min() / sy) - pad
        iy1 = int(np.ceil(combined_pts[:, 1].max() / sy)) + pad + 1
        iz0 = int(combined_pts[:, 2].min() / sz) - pad
        iz1 = int(np.ceil(combined_pts[:, 2].max() / sz)) + pad + 1

        lx = ix1 - ix0
        ly = iy1 - iy0
        lz = iz1 - iz0
        if lx <= 0 or ly <= 0 or lz <= 0:
            print("[Merge & Fill] Invalid bounding box")
            return None

        # ── 합성 HU 볼륨 ──
        # Base HU로 초기화 (base grid 범위 밖은 threshold 이하로 채움)
        fill_val = np.int16(self.current_min_threshold - 100)
        local_hu = np.full((lz, ly, lx), fill_val, dtype=np.int16)

        # Base HU 중 겹치는 영역만 복사
        b_ix0 = max(0, ix0); b_ix1 = min(nx, ix1)
        b_iy0 = max(0, iy0); b_iy1 = min(ny, iy1)
        b_iz0 = max(0, iz0); b_iz1 = min(nz, iz1)
        if b_ix1 > b_ix0 and b_iy1 > b_iy0 and b_iz1 > b_iz0:
            # local 배열 내 대응 위치
            l_ix0 = b_ix0 - ix0; l_ix1 = b_ix1 - ix0
            l_iy0 = b_iy0 - iy0; l_iy1 = b_iy1 - iy0
            l_iz0 = b_iz0 - iz0; l_iz1 = b_iz1 - iz0
            local_hu[l_iz0:l_iz1, l_iy0:l_iy1, l_ix0:l_ix1] = \
                base_hu[b_iz0:b_iz1, b_iy0:b_iy1, b_ix0:b_ix1]

        local_bounds = (ix0, ix1, iy0, iy1, iz0, iz1)

        for sidx in involved_series:
            if sidx == base_idx:
                continue
            resampled = self._resample_series_to_base_local(
                sidx, local_bounds, base_spacing
            )
            if resampled is not None:
                # 각 voxel에서 가장 높은 HU를 사용 (뼈 데이터 보존)
                local_hu = np.maximum(local_hu, resampled)
                print(f"[Merge & Fill] Resampled series {sidx} → "
                      f"composite HU updated")

        local_mask = local_hu >= self.current_min_threshold
        lz, ly, lx = local_mask.shape

        struct26 = generate_binary_structure(3, 3)
        struct6 = generate_binary_structure(3, 1)  # 6-connectivity

        target_uids = {b.get('uid', '') for b in bone_entries}

        # ═══════════════════════════════════════════════════════════════
        # ★ Mesh interior 기반 새 알고리즘 ★
        # ═══════════════════════════════════════════════════════════════
        # CC label 기반의 기존 방식은 shared CC 문제와 미분리 뼈를 못 잡음.
        # 새 방식은 각 뼈의 mesh 자체를 voxelize하여 정확한 solid volume을 얻음.
        #
        # 1) 모든 visible 뼈 → mesh interior voxelize (HU CC 무관)
        # 2) target_solid = target 뼈들의 mesh interior 합집합
        # 3) non_target_solid = 다른 separated 뼈들의 mesh interior 합집합
        # 4) unclaimed = local_mask 중 어느 mesh interior에도 안 속하는 voxel
        #    → 미분리 뼈 혹은 분리 안 된 noise → 무조건 차단
        # 5) forbidden = non_target_solid ∪ unclaimed
        #
        # 이렇게 하면:
        #  - shared HU CC (touching bones)도 mesh가 정확히 구분
        #  - 미분리 뼈 X가 A와 같은 CC에 있어도 A의 mesh 내부엔 X가 없음 → X는 unclaimed
        # ═══════════════════════════════════════════════════════════════

        all_bones_info = []   # list of (uid, interior_mask, is_target)
        total_seed_vox = 0

        for bone in self.separated_bones:
            is_target = bone.get('uid') in target_uids
            # invisible others는 territory 계산에서도 제외
            if not is_target and not bone.get('visible', True):
                continue
            mesh = bone.get('mesh')
            if mesh is None or mesh.n_points == 0:
                continue

            interior = self._voxelize_mesh_interior(
                mesh, (lz, ly, lx), base_spacing, (iz0, iy0, ix0),
            )
            if not interior.any():
                continue

            all_bones_info.append((bone.get('uid'), interior, is_target))
            total_seed_vox += int(mesh.n_points)

        # 합집합 마스크 구축
        target_solid = np.zeros((lz, ly, lx), dtype=bool)
        non_target_solid = np.zeros((lz, ly, lx), dtype=bool)
        bone_volumes = []   # per-target mesh interior (bridge dilation용)

        for uid, interior, is_target in all_bones_info:
            if is_target:
                target_solid |= interior
                bone_volumes.append(interior)
            else:
                non_target_solid |= interior

        if not bone_volumes:
            print("[Merge & Fill] No target bone interiors found")
            return None

        # unclaimed: local_mask 중 어떤 mesh interior에도 안 속함
        # = 미분리 뼈 / noise / 분리되지 않은 구조물
        all_interior = target_solid | non_target_solid
        unclaimed = local_mask & ~all_interior

        # forbidden: non-target mesh interior + unclaimed solid
        # → 선택된 뼈의 mesh interior 외에는 모두 차단
        forbidden = non_target_solid | unclaimed

        # 안전: bone_volumes 각각에서도 forbidden 제거
        # (mesh interior가 살짝 다른 뼈로 새어 들어간 경우 대비)
        bone_volumes = [bv & ~forbidden for bv in bone_volumes]
        target_solid = target_solid & ~forbidden

        # 전체 합침
        bone_combined = np.zeros((lz, ly, lx), dtype=bool)
        for bv in bone_volumes:
            bone_combined |= bv

        n_series = len(involved_series)
        print(f"[Merge & Fill] {n_series} series, local ({lx}×{ly}×{lz}), "
              f"mesh_pts={total_seed_vox}, bones={len(bone_volumes)}, "
              f"target_solid={target_solid.sum()}, "
              f"non_target_solid={non_target_solid.sum()}, "
              f"unclaimed={unclaimed.sum()}, "
              f"bone={bone_combined.sum()}")

        # ── Gap 채우기 ──
        # 핵심: 각 뼈의 dilation을 **geodesic dilation**으로 수행.
        # scipy binary_dilation의 mask 파라미터를 사용하면 dilation이 forbidden 영역을
        # 통과할 수 없음 → 다른 뼈를 관통하는 wide elliptic bridge가 원천 차단됨.
        # 두 뼈 사이에 다른 뼈가 있으면 dilation은 그 뼈를 우회해서 만남 (anatomical).
        original_bone = bone_combined.copy()

        if len(bone_volumes) >= 2:
            # Geodesic dilation: forbidden 영역을 우회하면서만 dilation 진행
            allowed_mask = ~forbidden if forbidden.any() else None
            dilated_bones = []
            for bv in bone_volumes:
                if allowed_mask is not None:
                    dilated = binary_dilation(bv, structure=struct6,
                                              iterations=iters,
                                              mask=allowed_mask)
                else:
                    dilated = binary_dilation(bv, structure=struct6,
                                              iterations=iters)
                dilated_bones.append(dilated)

            # 두 뼈 이상의 dilation이 겹치는 영역 = bridge zone
            # geodesic dilation 덕분에 bridge는 자연스럽게 forbidden 영역을 우회한 경로상에 있음
            bridge = np.zeros((lz, ly, lx), dtype=bool)
            for i in range(len(dilated_bones)):
                for j in range(i + 1, len(dilated_bones)):
                    overlap = dilated_bones[i] & dilated_bones[j]
                    bridge |= overlap

            # 추가 안전: 5-voxel buffer로 forbidden 주변 soft-tissue gradient zone까지 차단
            # geodesic이 forbidden을 통과 못 해도, 그 옆 1-voxel HU gradient 영역은 채울 수 있음.
            if forbidden.any():
                gap_fill_excl = (
                    binary_dilation(forbidden, structure=struct26, iterations=5)
                    & ~original_bone
                )
            else:
                gap_fill_excl = forbidden

            gap_fill = bridge & ~original_bone & ~gap_fill_excl

            # gap voxels 중 HU가 유효한 것만 (연조직 침범 방지)
            hu_offset = getattr(self, 'vote_threshold', 20)
            hu_min = self.current_min_threshold - hu_offset
            gap_fill = gap_fill & (local_hu >= hu_min)

            # Voronoi territory: mesh 좌표 기반 strict 기하 제약
            # unclaimed solid도 extra other seed로 사용 → 미분리 구조 주변 soft tissue까지 차단
            target_territory, has_other = self._compute_voronoi_territory(
                target_uids, (lz, ly, lx), base_spacing, (iz0, iy0, ix0),
                extra_other_seed=non_target_solid | unclaimed,
            )
            if has_other:
                before_gap = int(gap_fill.sum())
                gap_fill = gap_fill & target_territory
                print(f"[Merge & Fill] Voronoi clip: gap {before_gap} → "
                      f"{int(gap_fill.sum())} vox")

            closed = original_bone | gap_fill
            # 최종 안전망: forbidden + Voronoi 양쪽 적용
            closed = closed & ~forbidden
            if has_other:
                # original_bone에도 Voronoi 적용 (mesh interior가 살짝 새어 들어간 경우)
                # 단, 각 target의 mesh interior 합집합은 보존 (Voronoi가 너무 깎으면 안 되므로)
                closed = (closed & target_territory) | (target_solid & ~forbidden)
        else:
            closed = original_bone & ~forbidden

        print(f"[Merge & Fill] After bridge fill(iters={iters}): "
              f"{closed.sum()} vox "
              f"(+{closed.sum() - original_bone.sum()} gap filled)")

        # ── Re-mesh (base grid 좌표계) ──
        # 원래 뼈 영역: HU >= thr인 voxel은 실제 HU 유지 → marching cubes가
        # sub-voxel 보간으로 원래 surface 위치를 복원함.
        # closed 안의 HU < thr voxel (mesh interior 중 sub-thr인 곳 + bridge):
        #   → thr+1 강제 (이 voxel들도 결과 mesh에 포함되도록)
        # closed 밖 voxel: thr 미만으로 강제 (surface 경계 제한)
        thr = self.current_min_threshold
        local_values = local_hu.copy().astype(np.int16)
        # closed 안에서 HU < thr인 voxel 전부 thr+1로 (gap + sub-thr 뼈 내부)
        add_to_bone = closed & (local_hu < thr)
        if add_to_bone.any():
            local_values[add_to_bone] = np.int16(thr + 1)
        # 비-뼈 영역: threshold 미만으로 강제
        non_bone = ~closed
        local_values[non_bone] = np.minimum(
            local_hu[non_bone], np.int16(thr - 1))

        local_grid = pv.ImageData(
            dimensions=(lx, ly, lz),
            spacing=(sx, sy, sz),
            origin=(ix0 * sx, iy0 * sy, iz0 * sz),
        )
        local_grid.point_data["values"] = local_values.ravel(order="C")

        try:
            new_mesh = local_grid.contour([thr], scalars="values")
        except Exception as e:
            print(f"[Merge & Fill] contour failed: {e}")
            return None

        if new_mesh is None or new_mesh.n_points == 0:
            return None

        new_mesh = self._close_surface(new_mesh)
        new_mesh = self._apply_smoothing(new_mesh)
        new_mesh = self._close_surface(new_mesh)

        # 결과 mesh는 이미 base grid 좌표 → 추가 변환 불필요
        return new_mesh
