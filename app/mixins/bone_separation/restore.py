"""Click-to-Restore mode + voxel-level restore + undo dispatch.

Methods grouped here:
  - on_restore_picking_toggled: enter/exit Restore Mode (registers 3D click)
  - _clear_restore_highlight, _on_restore_point_picked
  - _on_restore_confirm (Enter), _on_restore_cancel (Esc)
  - _reselect_bones_by_uid: helper to re-select after list refresh
  - _restore_bone_voxel: HU CC + region grow + HU-preserving re-mesh
  - on_restore_undo_clicked: dispatch regular restore-undo or _undo_merge
"""

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from scipy.ndimage import (binary_dilation, generate_binary_structure,
                           label as ndi_label)


class BoneRestoreMixin:
    # =================================================================
    # Click-to-Restore: 클릭→선택(하이라이트) → Enter→실행, Esc→취소
    # =================================================================

    def on_restore_picking_toggled(self, checked):
        """Restore Mode = 클릭 선택 모드. VTK observer를 등록/해제."""
        self.restore_picking_enabled = checked
        if checked:
            # Mutual exclusion
            if hasattr(self, 'pick_btn') and self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)
            if hasattr(self, 'landmark_pick_btn') and self.landmark_pick_btn.isChecked():
                self.landmark_pick_btn.setChecked(False)
            # VTK observer 등록 (이때만 클릭 선택 활성화)
            self._enable_bone_click_selection()
            self.restore_pick_btn.setText("Restore Mode: ON")
            print(f"[Restore] Mode ON — {len(self.separated_bones)} bone(s), "
                  f"iters={self.restore_iterations}")
            if hasattr(self, 'separation_status_label'):
                self.separation_status_label.setText(
                    "Restore: 클릭=선택, Ctrl+클릭=추가, Enter=복원, Esc=취소"
                )
        else:
            # VTK observer 해제
            self._disable_bone_click_selection()
            self.restore_pick_btn.setText("Restore Mode: OFF")
            self._clear_restore_highlight()
            # 리스트 선택도 해제
            self.bone_list_widget.blockSignals(True)
            self.bone_list_widget.clearSelection()
            self.bone_list_widget.blockSignals(False)
            self._on_bone_list_selection_changed()
            print("[Restore] Mode OFF")

    def _clear_restore_highlight(self):
        """선택 하이라이트 제거."""
        if self._restore_highlight_actor is not None:
            try:
                self.plotter.remove_actor(self._restore_highlight_actor)
            except Exception:
                pass
            self._restore_highlight_actor = None
        self._restore_selected_bone = None

    def _on_restore_point_picked(self, *args):
        """클릭 → 뼈 선택 + 하이라이트 (아직 실행하지 않음). 카메라 유지."""
        # ── 인자 파싱 (PyVista 버전 호환) ──
        picked_point = None
        if len(args) == 1:
            picked_point = args[0]
        elif len(args) >= 2:
            first = args[0]
            if hasattr(first, 'points'):
                try:
                    picked_point = first.points[int(args[1])]
                except Exception:
                    picked_point = first
            else:
                picked_point = first

        if not self.separated_bones or picked_point is None:
            return

        try:
            # 카메라 시점 저장
            cam_pos = self.plotter.camera_position

            pt = np.asarray(picked_point, dtype=float)
            if pt.ndim != 1 or pt.shape[0] < 3:
                return
            pt = pt[:3]
            if np.allclose(pt, 0.0):
                return

            # 가장 가까운 뼈 찾기
            best_bone = None
            best_dist = float('inf')
            for bone in self.separated_bones:
                if not bone.get('visible', True):
                    continue
                mesh = bone['mesh']
                if mesh is None or mesh.n_points == 0:
                    continue
                cid = mesh.find_closest_point(pt)
                dist = float(np.linalg.norm(mesh.points[cid] - pt))
                if dist < best_dist:
                    best_dist = dist
                    best_bone = bone

            if best_bone is None:
                return

            # 이전 하이라이트 제거
            self._clear_restore_highlight()

            # 새 하이라이트: 노란색 와이어프레임 (카메라 리셋 방지)
            self._restore_selected_bone = best_bone
            try:
                edges = best_bone['mesh'].extract_all_edges()
                self._restore_highlight_actor = self.plotter.add_mesh(
                    edges, color='yellow', line_width=2, opacity=0.9,
                    reset_camera=False,
                )
            except Exception:
                self._restore_highlight_actor = self.plotter.add_mesh(
                    best_bone['mesh'], style='wireframe',
                    color='yellow', line_width=1, opacity=0.6,
                    reset_camera=False,
                )

            # 카메라 시점 복원
            self.plotter.camera_position = cam_pos
            self.plotter.render()

            if hasattr(self, 'separation_status_label'):
                self.separation_status_label.setText(
                    f"▶ '{best_bone['name']}' 선택됨 — Enter: 복원, Esc: 취소"
                )
            print(f"[Restore] Selected: '{best_bone['name']}' (dist={best_dist:.1f}mm)")

        except Exception as e:
            print(f"[Restore] Selection error: {e}")
            import traceback
            traceback.print_exc()

    def _on_restore_confirm(self):
        """Enter 키: 리스트에서 선택된 모든 뼈에 복원 실행. 카메라 시점 유지."""
        if not self.restore_picking_enabled:
            return

        # 리스트에서 선택된 모든 뼈 수집
        selected_items = self.bone_list_widget.selectedItems()
        if not selected_items:
            return
        bones_to_restore = []
        for item in selected_items:
            uid = item.data(Qt.UserRole)
            b = self._bone_by_uid(uid)
            if b is not None:
                bones_to_restore.append(b)
        if not bones_to_restore:
            return

        try:
            cam_pos = self.plotter.camera_position
            n = len(bones_to_restore)
            print(f"[Restore] Executing on {n} bone(s), {self.restore_iterations} iters…")
            if hasattr(self, 'separation_status_label'):
                self.separation_status_label.setText(f"Restoring: {n} bone(s)…")
                try:
                    QApplication.processEvents()
                except Exception:
                    pass

            success_count = 0
            for bone in bones_to_restore:
                # Undo 스택: 복원 전 mesh 저장
                self._restore_undo_stack.append(
                    (bone['uid'], bone['mesh'].copy(deep=True))
                )
                if len(self._restore_undo_stack) > self._max_restore_undo:
                    self._restore_undo_stack.pop(0)

                new_mesh = self._restore_bone_voxel(bone)
                if new_mesh is None or new_mesh.n_points == 0:
                    print(f"[Restore] ✗ '{bone['name']}' — empty result")
                    # 해당 undo 엔트리 제거
                    if self._restore_undo_stack:
                        self._restore_undo_stack.pop()
                    continue

                bone['mesh'] = new_mesh
                bone['voxel_count'] = new_mesh.n_cells
                old_actor = bone.get('actor')
                if old_actor is not None:
                    try:
                        self.plotter.remove_actor(old_actor)
                    except Exception:
                        pass
                new_actor = self.plotter.add_mesh(
                    new_mesh, color=bone['color'],
                    specular=0.5, smooth_shading=True,
                    reset_camera=False,
                )
                bone['actor'] = new_actor
                if not bone.get('visible', True):
                    try:
                        new_actor.SetVisibility(False)
                    except Exception:
                        pass
                success_count += 1
                print(f"[Restore] ✓ '{bone['name']}' → {new_mesh.n_cells} cells")

            self.restore_undo_btn.setEnabled(bool(self._restore_undo_stack))
            # 하이라이트 갱신 (mesh가 바뀌었으므로)
            self._refresh_separation_list()
            # 리스트 선택 복원 (refresh가 clear하므로 다시 선택)
            self._reselect_bones_by_uid([b['uid'] for b in bones_to_restore])
            self._on_bone_list_selection_changed()

            self.plotter.camera_position = cam_pos
            self.plotter.render()
            if hasattr(self, 'separation_status_label'):
                self.separation_status_label.setText(
                    f"✓ {success_count}/{n}개 복원 완료 — 다음 작업 또는 Esc"
                )

        except Exception as e:
            print(f"[Restore] ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _reselect_bones_by_uid(self, uids):
        """주어진 uid 리스트의 뼈들을 리스트에서 선택."""
        uid_set = set(uids)
        self.bone_list_widget.blockSignals(True)
        try:
            self.bone_list_widget.clearSelection()
            for i in range(self.bone_list_widget.count()):
                item = self.bone_list_widget.item(i)
                if item.data(Qt.UserRole) in uid_set:
                    item.setSelected(True)
        finally:
            self.bone_list_widget.blockSignals(False)

    def _on_restore_cancel(self):
        """Escape 키: 리스트 선택 해제."""
        if not self.restore_picking_enabled:
            return
        if self.bone_list_widget.selectedItems():
            print("[Restore] Selection cancelled")
            self.bone_list_widget.blockSignals(True)
            self.bone_list_widget.clearSelection()
            self.bone_list_widget.blockSignals(False)
            self._on_bone_list_selection_changed()
            if hasattr(self, 'separation_status_label'):
                self.separation_status_label.setText(
                    "Restore: 클릭=선택, Ctrl+클릭=추가, Enter=복원, Esc=취소"
                )

    def _restore_bone_voxel(self, bone_entry):
        """Voxel-level 복원을 해당 뼈 영역에만 적용하고 re-mesh.

        뼈 격리 방식 (다른 뼈 exclusion):
          - Mesh vertex → seed → dilation으로 뼈 내부+여유 포함
          - 다른 모든 뼈의 vertex → exclusion mask (접근 금지 영역)
          - bone_local = threshold & seed_dilated & ~exclusion
          - 복원 후에도 exclusion으로 clipping → 다른 뼈로 절대 안 번짐

        이 방식은 타겟 뼈가 바깥으로 자유롭게 성장 가능하되,
        다른 뼈 영역만 차단하므로 3D 방향 제한이 없음.
        """
        image_hu, spacing, meta = self._get_bone_series_data(bone_entry)
        if image_hu is None:
            return None

        sz, sy, sx = spacing
        nz, ny, nx = image_hu.shape
        iters = int(self.restore_iterations)
        mesh = bone_entry['mesh']

        # Fusion 역변환: base frame → 원래 grid 좌표
        pts = self._bone_pts_to_grid_coords(bone_entry, mesh.points.copy())

        # Bounding box → voxel 인덱스 (넉넉한 padding)
        pad = iters + 5
        ix0 = max(0, int(pts[:, 0].min() / sx) - pad)
        ix1 = min(nx, int(np.ceil(pts[:, 0].max() / sx)) + pad + 1)
        iy0 = max(0, int(pts[:, 1].min() / sy) - pad)
        iy1 = min(ny, int(np.ceil(pts[:, 1].max() / sy)) + pad + 1)
        iz0 = max(0, int(pts[:, 2].min() / sz) - pad)
        iz1 = min(nz, int(np.ceil(pts[:, 2].max() / sz)) + pad + 1)

        # Local HU 영역 추출
        local_hu = image_hu[iz0:iz1, iy0:iy1, ix0:ix1]
        local_mask = local_hu >= self.current_min_threshold
        lz, ly, lx = local_mask.shape

        # ── Mesh vertex → local voxel seed ──
        vix = np.round(pts[:, 0] / sx).astype(int) - ix0
        viy = np.round(pts[:, 1] / sy).astype(int) - iy0
        viz = np.round(pts[:, 2] / sz).astype(int) - iz0

        valid = ((vix >= 0) & (vix < lx) &
                 (viy >= 0) & (viy < ly) &
                 (viz >= 0) & (viz < lz))
        vix, viy, viz = vix[valid], viy[valid], viz[valid]

        seed = np.zeros((lz, ly, lx), dtype=bool)
        if len(vix) > 0:
            seed[viz, viy, vix] = True

        if not seed.any():
            print("[Restore] No seed voxels — fallback to mesh fill_holes")
            return mesh.fill_holes(1e10).clean()

        # ── 다른 뼈 exclusion mask (접근 금지 영역) ──
        # HU CC + Voronoi 기반: local_hu와 threshold를 전달하여
        # 각 다른 뼈의 실제 solid 영역을 정확히 식별 (mesh shell 누출 방지)
        target_uid = bone_entry.get('uid', '')
        working_sidx = bone_entry.get('series_index')
        exclusion = self._build_other_bones_exclusion(
            {target_uid}, (lz, ly, lx), spacing, (iz0, iy0, ix0),
            working_series_idx=working_sidx,
            local_hu=local_hu, threshold=self.current_min_threshold,
        )

        struct26 = generate_binary_structure(3, 3)

        # ── 넉넉한 dilation으로 뼈 전체 커버 ──
        # 뼈 bounding box의 1/3 정도면 interior + closing margin 충분.
        # exclusion이 다른 뼈를 차단하므로 넉넉해도 안전.
        half_ext = max(5, min((ix1 - ix0) // 3,
                              (iy1 - iy0) // 3,
                              (iz1 - iz0) // 3))
        dilation_radius = min(half_ext, 25)
        seed_dilated = binary_dilation(seed, structure=struct26,
                                        iterations=dilation_radius)

        # ── 이 뼈만의 mask ──
        # seed 자체를 항상 포함 → 현재 mesh 표면이 절대 빠지지 않음
        # → 반복 클릭 시 누적됨 (새 mesh → 새 seed → 이전 결과 보존)
        bone_local = ((local_mask | seed) & seed_dilated & ~exclusion)

        # ── 고립 noise 제거: seed에 연결된 component만 보존 ──
        labeled_local, n_comp = ndi_label(bone_local, structure=struct26)
        if n_comp > 1:
            seed_labels = set(np.unique(labeled_local[seed]))
            seed_labels.discard(0)
            if seed_labels:
                bone_local = np.isin(labeled_local, list(seed_labels))

        print(f"[Restore] Local ({lx}×{ly}×{lz}), "
              f"seed={seed.sum()}, dilation_r={dilation_radius}, "
              f"exclusion={exclusion.sum()}, bone_local={bone_local.sum()} "
              f"({n_comp} CC → seed-connected)")

        # ── HU Region Growing (원본 CT 값 기반 복원) ──
        # 현재 threshold에서 hu_offset만큼 낮춘 값을 복원 threshold로 사용
        # → 원본 threshold로는 잡히지 않았지만 뼈일 가능성 높은 voxel을 복원
        original_count = bone_local.sum()
        hu_offset = getattr(self, 'vote_threshold', 50)
        restore_thr = max(0, self.current_min_threshold - hu_offset)

        print(f"[Restore] HU region growing: threshold={self.current_min_threshold} "
              f"→ restore_thr={restore_thr} (offset={hu_offset}), "
              f"max_iters={iters}")

        # ── grow_exclusion: prevents leakage through soft-tissue HU gradient ──
        # At restore_thr = threshold − 50 the HU gradient zone around every bone
        # is ~3-5 voxels thick and is NOT in the threshold-based exclusion mask.
        # Dilating that mask by 5 voxels closes the gap, while:
        #   & ~seed    → the target seed itself (bridge area etc.) is never blocked
        #   | ~seed_dilated → hard spatial cap: grow never escapes the bone's extent
        if exclusion.any():
            grow_excl = binary_dilation(exclusion, structure=struct26,
                                         iterations=5) & ~seed
        else:
            grow_excl = np.zeros((lz, ly, lx), dtype=bool)
        grow_excl |= ~seed_dilated          # spatial hard limit

        closed = self._hu_region_grow(bone_local, local_hu, grow_excl,
                                       restore_thr, iters)

        # Final safety clamp: keep result inside seed_dilated and outside other-bone CCs
        closed = closed & seed_dilated & ~exclusion

        # ── Voronoi territory 강제 제약 ──
        # closed의 모든 voxel은 'target mesh보다 다른 뼈 mesh에 더 가까울 수 없음'.
        # buffer/spatial limit로 못 잡는 케이스(가까운 뼈로 grow가 휘어 들어감)도
        # 기하학적으로 strict 차단. seed는 항상 보존하여 원본 표면 손실 방지.
        target_territory, has_other = self._compute_voronoi_territory(
            {target_uid}, (lz, ly, lx), spacing, (iz0, iy0, ix0),
            working_series_idx=working_sidx,
        )
        if has_other:
            before = int(closed.sum())
            closed = (closed & target_territory) | seed
            print(f"[Restore] Voronoi clip: {before} → {int(closed.sum())} vox")

        print(f"[Restore] After HU-grow: {closed.sum()} vox "
              f"(+{closed.sum() - original_count} filled)")

        # ── Local grid → contour (marching cubes) ──
        # HU 보존 방식 (Merge & Fill과 동일):
        #   - 원래 HU >= thr이면서 closed인 voxel: 원본 HU 유지
        #     → marching cubes가 sub-voxel 보간으로 원래 표면 위치 복원
        #     → 매 restore마다 표면이 voxel grid로 흔들리는 손상 방지
        #   - 새로 추가된 voxel(원래 HU < thr이지만 region grow가 채움):
        #     thr+1로 강제 → 새 경계가 정확히 closed 영역에 일치
        #   - 비-뼈 영역(~closed): thr 미만으로 clamp → 가짜 surface 방지
        thr = self.current_min_threshold
        local_values = local_hu.astype(np.int16, copy=True)

        original_bone = local_hu >= thr
        added = closed & ~original_bone
        if added.any():
            local_values[added] = np.int16(thr + 1)

        non_bone = ~closed
        if non_bone.any():
            local_values[non_bone] = np.minimum(
                local_hu[non_bone].astype(np.int16),
                np.int16(thr - 1),
            )

        local_grid = pv.ImageData(
            dimensions=(lx, ly, lz),
            spacing=(sx, sy, sz),
            origin=(ix0 * sx, iy0 * sy, iz0 * sz),
        )
        local_grid.point_data["values"] = local_values.ravel(order="C")

        try:
            new_mesh = local_grid.contour([thr], scalars="values")
        except Exception as e:
            print(f"[Restore] contour failed: {e}")
            return None

        if new_mesh is None or new_mesh.n_points == 0:
            return None

        # ── Mesh 후처리 (main pipeline과 동일) ──
        new_mesh = self._close_surface(new_mesh)
        new_mesh = self._apply_smoothing(new_mesh)
        new_mesh = self._close_surface(new_mesh)

        # Fusion: base frame으로 재변환
        result = self._grid_coords_to_bone_frame(bone_entry, new_mesh.points)
        if result is not new_mesh.points:
            new_mesh.points = result

        return new_mesh

    def on_restore_undo_clicked(self):
        """마지막 복원/합치기를 되돌림. 카메라 시점 유지."""
        if not self._restore_undo_stack:
            self.restore_undo_btn.setEnabled(False)
            return

        cam_pos = self.plotter.camera_position

        entry = self._restore_undo_stack.pop()
        if not self._restore_undo_stack:
            self.restore_undo_btn.setEnabled(False)

        # Merge undo: ('__merge__', merged_uid, [original_bone_snapshots])
        if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == '__merge__':
            _, merged_uid, original_snapshots = entry
            self._undo_merge(merged_uid, original_snapshots, cam_pos)
            return

        # Regular undo: (uid, old_mesh)
        uid, old_mesh = entry

        bone = self._bone_by_uid(uid)
        if bone is None:
            return

        bone['mesh'] = old_mesh
        bone['voxel_count'] = old_mesh.n_cells
        old_actor = bone.get('actor')
        if old_actor is not None:
            try:
                self.plotter.remove_actor(old_actor)
            except Exception:
                pass
        new_actor = self.plotter.add_mesh(
            old_mesh, color=bone['color'],
            specular=0.5, smooth_shading=True,
            reset_camera=False,
        )
        bone['actor'] = new_actor
        if not bone.get('visible', True):
            try:
                new_actor.SetVisibility(False)
            except Exception:
                pass
        self._refresh_separation_list()
        self.plotter.camera_position = cam_pos
        self.plotter.render()
