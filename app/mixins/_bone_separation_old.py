import os
import re
import time
import uuid
from datetime import datetime

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QItemSelectionModel
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QListWidgetItem,
    QMessageBox,
)
from scipy.ndimage import (binary_closing, binary_dilation,
                           binary_fill_holes, binary_opening,
                           distance_transform_edt,
                           generate_binary_structure,
                           label as ndi_label)

class BoneSeparationMixin:
    def _bone_color_palette(self, n):
        """Return n visually distinct RGB tuples in [0,1].
        Uses matplotlib's tab20 cyclically — 20 distinct hues, more than
        enough for typical anatomical regions.
        """
        try:
            import matplotlib.cm as mcm
            cmap = mcm.get_cmap('tab20', 20)
            return [tuple(float(c) for c in cmap(i % 20)[:3]) for i in range(max(n, 1))]
        except Exception:
            # Deterministic fallback: HSV wheel
            return [
                (0.5 + 0.5 * np.cos(2 * np.pi * i / max(n, 1)),
                 0.5 + 0.5 * np.cos(2 * np.pi * i / max(n, 1) + 2.094),
                 0.5 + 0.5 * np.cos(2 * np.pi * i / max(n, 1) + 4.189))
                for i in range(max(n, 1))
            ]


    def _clear_separated_actors(self):
        """Remove every separated-bone actor from the plotter and reset state.
        Leaves the base single-mesh actor untouched."""
        self._clear_list_highlight()
        for bone in self.separated_bones:
            actor = bone.get('actor')
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.separated_bones.clear()


    def _new_bone_uid(self):
        return str(uuid.uuid4())

    def _sanitize_bone_export_name(self, name):
        safe = re.sub(r'[^\w\-.]+', '_', str(name)).strip('._')
        return safe[:80] if safe else 'bone'

    def _bone_by_uid(self, uid):
        for bone in self.separated_bones:
            if bone.get('uid') == uid:
                return bone
        return None

    def _set_separation_tools_enabled(self, enabled):
        """Enable Phase 2 controls only while separated bones are on screen."""
        for attr in ('rename_bone_btn', 'merge_bones_btn', 'export_bones_stl_btn',
                     'bone_list_widget', 'toggle_vis_btn',
                     'restore_pick_btn', 'restore_iter_spinbox',
                     'vote_threshold_spinbox',
                     'merge_fill_btn', 'merge_fill_iter_spinbox'):
            if hasattr(self, attr):
                getattr(self, attr).setEnabled(bool(enabled))
        # Restore undo is managed separately (depends on stack, not just enabled)
        if not enabled:
            if hasattr(self, 'restore_undo_btn'):
                self.restore_undo_btn.setEnabled(False)
            if hasattr(self, 'restore_pick_btn') and self.restore_pick_btn.isChecked():
                self.restore_pick_btn.setChecked(False)

    def _bone_list_item_label(self, bone):
        vis = "●" if bone.get('visible', True) else "○"
        return f"{vis} {bone['name']}  ({bone['voxel_count']:,} vox)"

    def _refresh_separation_list(self):
        """Rebuild the bone QListWidget from self.separated_bones."""
        if not hasattr(self, 'bone_list_widget'):
            return
        self.bone_list_widget.blockSignals(True)
        self.bone_list_widget.clear()
        for bone in self.separated_bones:
            if 'uid' not in bone:
                bone['uid'] = self._new_bone_uid()
            item = QListWidgetItem(self._bone_list_item_label(bone))
            item.setFlags(
                Qt.ItemIsSelectable | Qt.ItemIsEnabled
            )
            item.setData(Qt.UserRole, bone['uid'])
            self.bone_list_widget.addItem(item)
        self.bone_list_widget.blockSignals(False)

    def _on_bone_list_item_changed(self, item):
        """Legacy — no longer used (checkboxes removed)."""
        pass

    def on_toggle_bone_visibility_clicked(self):
        """선택된 뼈의 visibility를 토글. Hide/Show 버튼 또는 H 키."""
        items = self.bone_list_widget.selectedItems()
        if not items:
            return
        cam_pos = self.plotter.camera_position
        for item in items:
            uid = item.data(Qt.UserRole)
            bone = self._bone_by_uid(uid)
            if bone is None:
                continue
            bone['visible'] = not bone.get('visible', True)
            actor = bone.get('actor')
            if actor is not None:
                try:
                    actor.SetVisibility(bool(bone['visible']))
                except Exception:
                    pass
            # 리스트 텍스트 업데이트 (● / ○)
            item.setText(self._bone_list_item_label(bone))
        self.plotter.camera_position = cam_pos
        self.plotter.render()

    # ── 3D bone click‑to‑select (VTK observer, landmarks.py 패턴) ──

    def _enable_bone_click_selection(self):
        """3D 뷰에서 뼈 클릭 → 리스트 선택 연동 활성화.
        PyVista wrapper → native VTK interactor 순으로 observer 등록."""
        if getattr(self, '_bone_click_press_obs', None) is not None:
            return  # 이미 활성화
        try:
            import vtk
            self._bone_cell_picker = vtk.vtkCellPicker()
            self._bone_cell_picker.SetTolerance(0.005)
        except Exception as e:
            print(f"[BoneClick] vtkCellPicker init failed: {e}")
            return

        self._bone_click_press_pos = None  # mouse‑down screen pos

        # ── press observer (드래그 vs 클릭 구분용) ──
        try:
            self._bone_click_press_obs = self.plotter.iren.add_observer(
                'LeftButtonPressEvent', self._on_3d_bone_press)
        except AttributeError:
            try:
                native = self.plotter.iren.interactor
                self._bone_click_press_obs = native.AddObserver(
                    'LeftButtonPressEvent', self._on_3d_bone_press)
            except Exception as e:
                print(f"[BoneClick] press observer failed: {e}")
                self._bone_click_press_obs = None

        # ── release observer (실제 선택 로직) ──
        try:
            self._bone_click_release_obs = self.plotter.iren.add_observer(
                'LeftButtonReleaseEvent', self._on_3d_bone_release)
        except AttributeError:
            try:
                native = self.plotter.iren.interactor
                self._bone_click_release_obs = native.AddObserver(
                    'LeftButtonReleaseEvent', self._on_3d_bone_release)
            except Exception as e:
                print(f"[BoneClick] release observer failed: {e}")
                self._bone_click_release_obs = None

        print(f"[BoneClick] Enabled  press={self._bone_click_press_obs}  "
              f"release={self._bone_click_release_obs}")

    def _disable_bone_click_selection(self):
        """3D 뷰 클릭 선택 비활성화."""
        for attr in ('_bone_click_press_obs', '_bone_click_release_obs'):
            obs_id = getattr(self, attr, None)
            if obs_id is not None:
                try:
                    self.plotter.iren.remove_observer(obs_id)
                except AttributeError:
                    try:
                        self.plotter.iren.interactor.RemoveObserver(obs_id)
                    except Exception:
                        pass
                except Exception:
                    pass
                setattr(self, attr, None)
        self._bone_click_press_pos = None

    def _get_native_iren(self, obj):
        """Return native VTK interactor for GetEventPosition() etc."""
        try:
            return self.plotter.iren.interactor
        except AttributeError:
            return obj  # observer passes interactor as first arg

    def _on_3d_bone_press(self, obj, event):
        """Mouse-down 위치 기록 (드래그 감지용)."""
        try:
            iren = self._get_native_iren(obj)
            self._bone_click_press_pos = iren.GetEventPosition()
        except Exception:
            self._bone_click_press_pos = None

    def _on_3d_bone_release(self, obj, event):
        """Mouse-up → 클릭인지 확인 후 뼈 선택.
        Observer는 Restore Mode 토글 시점에만 등록되므로 일반 가드만 체크."""
        if not self.bone_separation_enabled or not self.separated_bones:
            return

        try:
            iren = self._get_native_iren(obj)
            release_pos = iren.GetEventPosition()
        except Exception:
            return

        # ── 클릭 vs 드래그 구분 (5 px 이내만 클릭으로 인정) ──
        press = getattr(self, '_bone_click_press_pos', None)
        if press is not None:
            dx = abs(release_pos[0] - press[0])
            dy = abs(release_pos[1] - press[1])
            if dx + dy > 5:
                return  # 드래그(회전/팬) → 무시

        # ── Pick ──
        try:
            renderer = self.plotter.renderer
            self._bone_cell_picker.Pick(release_pos[0], release_pos[1],
                                        0, renderer)
            picked_actor = self._bone_cell_picker.GetActor()
            if picked_actor is None:
                return
        except Exception:
            return

        cam_pos = self.plotter.camera_position

        # ── picked actor ↔ bone 매칭 ──
        target_bone = None
        for bone in self.separated_bones:
            if not bone.get('visible', True):
                continue
            if bone.get('actor') is picked_actor:
                target_bone = bone
                break

        # fallback: actor 비교 실패 시 world position 기반 최근접 검색
        if target_bone is None:
            world_pos = np.array(self._bone_cell_picker.GetPickPosition())
            if np.allclose(world_pos, 0.0):
                return
            best_dist = float('inf')
            for bone in self.separated_bones:
                if not bone.get('visible', True):
                    continue
                mesh = bone.get('mesh')
                if mesh is None or mesh.n_points == 0:
                    continue
                cid = mesh.find_closest_point(world_pos)
                dist = float(np.linalg.norm(mesh.points[cid] - world_pos))
                if dist < best_dist:
                    best_dist = dist
                    target_bone = bone

        if target_bone is None:
            return

        # ── Ctrl 감지 (Qt 우선, VTK fallback) ──
        ctrl_held = False
        try:
            mods = QApplication.keyboardModifiers()
            ctrl_held = bool(mods & Qt.ControlModifier)
        except Exception:
            pass
        if not ctrl_held:
            try:
                iren = self._get_native_iren(obj)
                ctrl_held = bool(iren.GetControlKey())
            except Exception:
                pass

        # ── 리스트 선택 업데이트 ──
        # MultiSelection 모드에서 selectionModel.select(ClearAndSelect)는
        # 다른 항목을 해제 안 할 수 있음 → clearSelection() + setSelected() 사용
        target_uid = target_bone.get('uid')
        print(f"[BoneClick] hit={target_bone.get('name')}  ctrl={ctrl_held}")

        # block signals: 일괄 변경 후 시그널 1번만 발생시키기
        self.bone_list_widget.blockSignals(True)
        try:
            if not ctrl_held:
                # 일반 클릭: 전부 해제 후 target만 선택
                self.bone_list_widget.clearSelection()
                for i in range(self.bone_list_widget.count()):
                    item = self.bone_list_widget.item(i)
                    if item.data(Qt.UserRole) == target_uid:
                        item.setSelected(True)
                        break
            else:
                # Ctrl+클릭: target만 토글, 나머지 그대로
                for i in range(self.bone_list_widget.count()):
                    item = self.bone_list_widget.item(i)
                    if item.data(Qt.UserRole) == target_uid:
                        item.setSelected(not item.isSelected())
                        break
        finally:
            self.bone_list_widget.blockSignals(False)

        # blockSignals 동안 시그널이 발생 안 했으므로 수동 호출
        self._on_bone_list_selection_changed()
        self.plotter.camera_position = cam_pos

    def _clear_list_highlight(self):
        """리스트 선택 하이라이트 제거."""
        for actor in getattr(self, '_list_highlight_actors', []):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self._list_highlight_actors = []

    def _on_bone_list_selection_changed(self):
        """리스트에서 뼈를 선택하면 3D 뷰에서 하이라이트."""
        # 재진입 방지 (plotter.render() 중 재호출 차단)
        if getattr(self, '_in_highlight_update', False):
            return
        self._in_highlight_update = True
        try:
            cam_pos = self.plotter.camera_position
            self._clear_list_highlight()

            items = self.bone_list_widget.selectedItems()
            if not items:
                self.plotter.camera_position = cam_pos
                self.plotter.render()
                return

            for item in items:
                uid = item.data(Qt.UserRole)
                bone = self._bone_by_uid(uid)
                if bone is None or bone.get('mesh') is None:
                    continue
                if not bone.get('visible', True):
                    continue
                try:
                    highlight = self.plotter.add_mesh(
                        bone['mesh'],
                        color='yellow',
                        style='wireframe',
                        line_width=2,
                        opacity=0.5,
                        pickable=False,
                        reset_camera=False,
                    )
                    self._list_highlight_actors.append(highlight)
                except Exception:
                    pass

            self.plotter.camera_position = cam_pos
            self.plotter.render()
        finally:
            self._in_highlight_update = False

    def _on_bone_list_item_double_clicked(self, item):
        self._rename_bone_item(item)

    def _rename_bone_item(self, item):
        if item is None:
            return
        uid = item.data(Qt.UserRole)
        bone = self._bone_by_uid(uid)
        if bone is None:
            return
        text, ok = QInputDialog.getText(
            self,
            "Rename Bone",
            "Bone name:",
            text=bone['name'],
        )
        if not ok:
            return
        new_name = text.strip()
        if not new_name:
            return
        bone['name'] = new_name
        self.bone_list_widget.blockSignals(True)
        item.setText(self._bone_list_item_label(bone))
        self.bone_list_widget.blockSignals(False)

    def on_rename_bone_clicked(self):
        if not self.bone_separation_enabled:
            return
        item = self.bone_list_widget.currentItem()
        if item is None:
            QMessageBox.information(
                self, "Rename Bone", "Select a bone in the list first."
            )
            return
        self._rename_bone_item(item)

    def on_merge_bones_clicked(self):
        """Merge all selected list items into one bone mesh (undo 지원)."""
        if not self.bone_separation_enabled:
            return
        items = self.bone_list_widget.selectedItems()
        if len(items) < 2:
            QMessageBox.information(
                self,
                "Merge Bones",
                "Select at least two bones (Ctrl+click) in the list, then click Merge.",
            )
            return

        uids = [it.data(Qt.UserRole) for it in items]
        bones = [self._bone_by_uid(uid) for uid in uids]
        bones = [b for b in bones if b is not None]
        if len(bones) < 2:
            return

        try:
            merged_mesh = bones[0]['mesh'].copy(deep=True)
            for bone in bones[1:]:
                try:
                    merged_mesh = merged_mesh.merge(bone['mesh'])
                except Exception:
                    merged_mesh = merged_mesh + bone['mesh']
        except Exception as e:
            QMessageBox.critical(self, "Merge Failed", str(e))
            return

        if merged_mesh is None or merged_mesh.n_points == 0:
            QMessageBox.warning(self, "Merge Failed", "Merged mesh is empty.")
            return

        merged_names = [b['name'] for b in bones]
        default_name = " + ".join(merged_names[:3])
        if len(merged_names) > 3:
            default_name += f" (+{len(merged_names) - 3} more)"
        name, ok = QInputDialog.getText(
            self,
            "Merged Bone Name",
            "Name for the merged bone:",
            text=default_name,
        )
        if not ok:
            return
        merged_name = name.strip() or default_name
        total_voxels = sum(int(b.get('voxel_count', 0)) for b in bones)

        # 카메라 시점 저장
        cam_pos = self.plotter.camera_position

        # Undo: 원본 뼈들의 전체 상태를 저장
        # ('__merge__', merged_bone_uid, [original_bone_snapshots])
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

        # Remove old actors and entries
        uid_set = set(b['uid'] for b in bones)
        for bone in list(self.separated_bones):
            if bone.get('uid') in uid_set:
                actor = bone.get('actor')
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
        self.separated_bones = [
            b for b in self.separated_bones if b.get('uid') not in uid_set
        ]

        color = self._bone_color_palette(1)[0]
        actor = self.plotter.add_mesh(
            merged_mesh,
            color=color,
            specular=0.5,
            smooth_shading=True,
            reset_camera=False,
        )
        self.separated_bones.append({
            'uid': merged_uid,
            'id': 0,
            'mesh': merged_mesh,
            'actor': actor,
            'visible': True,
            'color': color,
            'voxel_count': total_voxels,
            'name': merged_name,
            'series_index': bones[0].get('series_index'),
        })

        # Undo 스택에 merge 항목 추가
        self._restore_undo_stack.append(
            ('__merge__', merged_uid, original_snapshots)
        )
        if len(self._restore_undo_stack) > self._max_restore_undo:
            self._restore_undo_stack = self._restore_undo_stack[-self._max_restore_undo:]
        self.restore_undo_btn.setEnabled(True)

        self._refresh_separation_list()
        # 카메라 시점 복원
        self.plotter.camera_position = cam_pos
        self.separation_status_label.setText(
            f"Merged {len(bones)} bones → \"{merged_name}\" "
            f"({len(self.separated_bones)} total)"
        )
        self.plotter.render()

    def on_export_separated_bones_stl(self):
        """Export each visible separated bone as an individual STL file."""
        if not self.bone_separation_enabled or not self.separated_bones:
            QMessageBox.information(
                self, "Export Bones",
                "Run Separate Bones first."
            )
            return

        visible = [b for b in self.separated_bones if b.get('visible', True)]
        if not visible:
            QMessageBox.information(
                self, "Export Bones", "No visible bones to export."
            )
            return

        base_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Folder for Separated Bones"
        )
        if not base_dir:
            return

        name = "UnknownPatient"
        date_str = "UnknownDate"
        if self.current_meta_info:
            name = self.current_meta_info.get('patient_name', name)
            date_str = self.current_meta_info.get('study_date', date_str)
        name = "".join(c for c in str(name) if c.isalnum() or c in (' ', '-', '_')).strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(
            base_dir,
            f"{name}_{date_str}_{timestamp}_Bones_Export",
        )
        os.makedirs(export_dir, exist_ok=True)

        used_names = set()
        exported = []
        try:
            for bone in visible:
                base = self._sanitize_bone_export_name(bone['name'])
                fname = base
                n = 2
                while fname.lower() in used_names:
                    fname = f"{base}_{n}"
                    n += 1
                used_names.add(fname.lower())
                path = os.path.join(export_dir, f"{fname}.stl")
                bone['mesh'].save(path)
                exported.append((
                    bone['name'], fname, path, bone.get('voxel_count', 0)
                ))

            report_path = os.path.join(export_dir, "bones_export_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("--- Separated Bones STL Export ---\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Count: {len(exported)}\n\n")
                for disp, fname, path, vox in exported:
                    f.write(f"{disp}\n  file: {os.path.basename(path)}\n")
                    f.write(f"  voxels: {vox}\n\n")
                # 작업 이력
                undo_stack = getattr(self, '_restore_undo_stack', [])
                if undo_stack:
                    f.write("--- Operation History (undo stack) ---\n")
                    for j, entry in enumerate(undo_stack):
                        if (isinstance(entry, tuple) and len(entry) == 3
                                and entry[0] == '__merge__'):
                            names = [s['name'] for s in entry[2]]
                            f.write(f"  {j+1}. Merge: {', '.join(names)}\n")
                        else:
                            uid, _ = entry
                            bone = self._bone_by_uid(uid)
                            bname = bone['name'] if bone else uid[:8]
                            f.write(f"  {j+1}. Restore: {bname}\n")
                    f.write("\n")

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(exported)} bone(s) to:\n{export_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def on_sep_stage_a_toggled(self, state):
        self.separation_apply_stage_a = (state == Qt.Checked)

    def on_min_bone_vox_changed(self, value):
        self.min_bone_voxels = int(value)


    def _compute_single_separated_bone_entries(self):
        """Split the base mesh into separated bones by connected component.

        Uses _compute_masked_values exactly like update_base_mesh, so every
        voxel-level option (closing, particle removal, threshold) applies
        identically. Bones that are physically disconnected in the resulting
        mesh become separate objects; touching bones stay as one CC.
        """
        if self.volume_grid is None or self.current_image_hu is None:
            return [], 0.0

        t0 = time.time()

        # Build pre-smoothing base mesh: same mask as update_base_mesh, but
        # stop before smoothing so each CC can be smoothed independently.
        values = self._compute_masked_values(self.current_image_hu)
        self.volume_grid.point_data["masked"] = values
        raw_base = self.volume_grid.contour(
            [self.current_min_threshold], scalars="masked"
        )
        raw_base = self._close_surface(raw_base)

        if raw_base is None or raw_base.n_points == 0:
            return [], time.time() - t0

        if (hasattr(self, 'crop_checkbox') and self.crop_checkbox.isChecked() and
                self.cropping_bounds is not None):
            try:
                raw_base = raw_base.clip_box(self.cropping_bounds, invert=False)
            except Exception as e:
                print(f"[Separation] crop on base mesh failed: {e}")

        if raw_base is None or raw_base.n_points == 0:
            return [], time.time() - t0

        entries = self._split_mesh_into_bone_entries(
            raw_base, series_index=None, name_prefix="Bone "
        )
        return entries, time.time() - t0


    def _split_mesh_into_bone_entries(self, raw_base, series_index=None,
                                      name_prefix="Bone "):
        """Run connectivity on a pre-smoothing mesh and return bone entries.

        Each entry stores the pre-smoothing CC slice as raw_mesh and the
        smoothed/stage-A version as mesh. min_bone_voxels is reinterpreted
        as a minimum cell count to filter out micro-fragments.
        """
        try:
            labeled = raw_base.connectivity(extraction_mode='all')
        except Exception as e:
            print(f"[Separation] connectivity failed: {e}")
            return []

        rids = labeled.cell_data.get('RegionId')
        if rids is None:
            return []
        rids_arr = np.asarray(rids)
        unique_ids, counts = np.unique(rids_arr, return_counts=True)
        if len(unique_ids) == 0:
            return []

        min_cells = max(1, int(getattr(self, 'min_bone_voxels', 1)))

        # Sort by size descending so larger bones get lower IDs
        order = np.argsort(-counts)
        unique_ids = unique_ids[order]
        counts = counts[order]

        keep_mask = counts >= min_cells
        if not keep_mask.any():
            keep_mask[0] = True  # fallback: at least keep the largest

        entries = []
        bone_num = 0
        for rid, count, keep in zip(unique_ids, counts, keep_mask):
            if not keep:
                continue
            try:
                cells_to_remove = np.where(rids_arr != rid)[0]
                sub = labeled.remove_cells(cells_to_remove)
                if not isinstance(sub, pv.PolyData):
                    sub = sub.extract_surface()
            except Exception as e:
                print(f"[Separation] extract region {rid} failed: {e}")
                continue

            for arr in ('RegionId',):
                if arr in sub.point_data:
                    del sub.point_data[arr]
                if arr in sub.cell_data:
                    del sub.cell_data[arr]

            if sub is None or sub.n_points == 0:
                continue

            raw_mesh = sub.copy(deep=True)
            mesh = self._apply_smoothing(sub)
            if mesh is None or mesh.n_points == 0:
                mesh = raw_mesh.copy(deep=True)
            mesh = self._close_surface(mesh)
            if getattr(self, 'separation_apply_stage_a', False):
                stage_a = self._apply_stage_a(mesh)
                if stage_a is not None and stage_a.n_points > 0:
                    mesh = stage_a
            if mesh is None or mesh.n_points == 0:
                continue

            bone_num += 1
            entries.append({
                'name': f"{name_prefix}{bone_num}",
                'mesh': mesh,
                'raw_mesh': raw_mesh,
                'voxel_count': int(count),
                'series_index': series_index,
                'label_id': bone_num,
            })
        return entries


    def _compute_fusion_separated_bone_entries(self):
        """Per-series mesh-CC separation, transformed into the base grid frame.

        For each included series: build the same pre-smoothing mesh that
        _update_fused_meshes would use, transform into the base series grid,
        clip to crop box, then split by connected component.
        """
        if not self.all_series_data:
            return [], 0.0

        t0 = time.time()
        base_idx = max(0, min(self.base_series_index, len(self.all_series_data) - 1))
        base_meta = self.all_series_data[base_idx].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_base_inv = np.linalg.inv(T_base) if T_base is not None else None

        crop_active = (
            self.cropping_bounds is not None and
            hasattr(self, 'crop_checkbox') and
            self.crop_checkbox.isChecked()
        )

        if len(self.fusion_include_flags) != len(self.all_series_data):
            self.fusion_include_flags = [True] * len(self.all_series_data)

        entries = []
        for i, sd in enumerate(self.all_series_data):
            if not self.fusion_include_flags[i]:
                continue

            image_hu = sd['image_hu']
            spacing = sd['spacing']
            meta = sd.get('meta') or {}
            T_i = self._series_grid_to_lps_matrix(meta)

            if i != base_idx and T_i is None:
                print(f"[Separation] Series {i} missing DICOM geometry; skipped.")
                continue

            nz, ny, nx = image_hu.shape
            sz, sy, sx = spacing
            grid_i = pv.ImageData(
                dimensions=(nx, ny, nz),
                spacing=(sx, sy, sz),
            )
            try:
                values = self._compute_masked_values(image_hu)
            except Exception as e:
                print(f"[Separation] Series {i} mask failed: {e}")
                continue
            grid_i.point_data["masked"] = values
            try:
                raw_base = grid_i.contour(
                    [self.current_min_threshold], scalars="masked"
                )
            except Exception as e:
                print(f"[Separation] Series {i} contour failed: {e}")
                continue
            raw_base = self._close_surface(raw_base)
            if raw_base is None or raw_base.n_points == 0:
                continue

            if i != base_idx and T_base_inv is not None and T_i is not None:
                T_composite = T_base_inv @ T_i
                raw_base.transform(T_composite, inplace=True)

            if crop_active and self.cropping_bounds is not None:
                try:
                    raw_base = raw_base.clip_box(self.cropping_bounds, invert=False)
                except Exception:
                    pass
                if raw_base is None or raw_base.n_points == 0:
                    continue

            desc = str(meta.get('series_description', f'Series {i}'))[:24]
            sub_idx = meta.get('sub_idx', '')
            series_tag = f"[{i}]"
            if sub_idx != '' and sub_idx is not None:
                series_tag += f".{sub_idx}"

            series_entries = self._split_mesh_into_bone_entries(
                raw_base,
                series_index=i,
                name_prefix=f"{series_tag} {desc} — B",
            )
            entries.extend(series_entries)

        return entries, time.time() - t0


    def _hide_combined_mesh_for_separation(self):
        """Hide the normal combined-mesh actors before showing separated bones."""
        self._presep_fusion_visibility = []
        if self.fusion_enabled:
            for actor in self.fusion_actors:
                try:
                    self._presep_fusion_visibility.append(bool(actor.GetVisibility()))
                    actor.SetVisibility(False)
                except Exception:
                    self._presep_fusion_visibility.append(True)
            if self.current_mesh_actor is not None:
                try:
                    self.current_mesh_actor.SetVisibility(False)
                except Exception:
                    pass
        else:
            if self.current_mesh_actor is not None:
                try:
                    self._presep_base_actor_was_visible = bool(
                        self.current_mesh_actor.GetVisibility()
                    )
                    self.current_mesh_actor.SetVisibility(False)
                except Exception:
                    self._presep_base_actor_was_visible = True


    def _restore_combined_mesh_after_separation(self):
        """Restore visibility of fusion or single combined mesh actors."""
        if self.fusion_enabled:
            for actor, vis in zip(self.fusion_actors, self._presep_fusion_visibility):
                try:
                    actor.SetVisibility(bool(vis))
                except Exception:
                    pass
            self._presep_fusion_visibility = []
            if self.current_mesh_actor is not None:
                try:
                    self.current_mesh_actor.SetVisibility(False)
                except Exception:
                    pass
        else:
            if self.current_mesh_actor is not None:
                try:
                    self.current_mesh_actor.SetVisibility(
                        self._presep_base_actor_was_visible
                    )
                except Exception:
                    pass


    def _show_separated_bones(self, entries, took):
        """Add separated-bone actors to the plotter from pre-built entries."""
        self._clear_separated_actors()
        self._hide_combined_mesh_for_separation()

        colors = self._bone_color_palette(len(entries))
        for entry, color in zip(entries, colors):
            mesh = entry['mesh']
            actor = self.plotter.add_mesh(
                mesh,
                color=color,
                specular=0.5,
                smooth_shading=True,
            )
            self.separated_bones.append({
                'uid': self._new_bone_uid(),
                'id': entry.get('label_id', 0),
                'mesh': mesh,
                'raw_mesh': entry.get('raw_mesh'),
                'actor': actor,
                'visible': True,
                'color': color,
                'voxel_count': entry['voxel_count'],
                'name': entry['name'],
                'series_index': entry.get('series_index'),
            })

        self.bone_separation_enabled = True
        self.clear_separation_btn.setEnabled(True)
        self._set_separation_tools_enabled(True)
        # 클릭 선택은 Restore Mode 토글 시점에 활성화됨 (여기서는 등록 안 함)
        mode = "fusion" if self.fusion_enabled else "single"
        self.separation_status_label.setText(
            f"Separated into {len(entries)} bone(s) ({mode}, {took:.1f}s, "
            f"min cells={self.min_bone_voxels:,})"
        )
        self._refresh_separation_list()
        try:
            self.plotter.update()
        except Exception:
            pass


    def on_separate_bones_clicked(self):
        """Run separation on the active series (single) or all included fusion
        series, then show per-bone meshes with individual visibility toggles."""
        if not self.all_series_data and self.current_image_hu is None:
            QMessageBox.information(
                self, "Bone Separation", "Load a patient first."
            )
            return
        if self.fusion_enabled and not self.all_series_data:
            QMessageBox.information(
                self, "Bone Separation", "Load a patient first."
            )
            return
        if (not self.fusion_enabled and
                (self.volume_grid is None or self.current_image_hu is None)):
            QMessageBox.information(
                self, "Bone Separation", "Load a patient first."
            )
            return

        self.separation_status_label.setText("Separating…")
        self.separate_btn.setEnabled(False)
        try:
            QApplication.processEvents()
        except Exception:
            pass

        try:
            if self.fusion_enabled:
                self.separation_status_label.setText(
                    "Separating included fusion series…"
                )
                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                entries, took = self._compute_fusion_separated_bone_entries()
            else:
                entries, took = self._compute_single_separated_bone_entries()

            if not entries:
                self.separation_status_label.setText(
                    "No bones found. Try smaller min voxels, lower erosion, "
                    "or include more series in fusion."
                )
                return

            self.separation_status_label.setText(
                f"Building {len(entries)} mesh(es)…"
            )
            try:
                QApplication.processEvents()
            except Exception:
                pass

            self._show_separated_bones(entries, took)
        finally:
            self.separate_btn.setEnabled(True)


    def _reapply_mesh_ops_to_separated_bones(self):
        """Re-apply smoothing + Stage A to each separated bone from its stored
        raw_mesh, without re-running the voxel-level separation pipeline.

        Called when mesh-level settings change (smoothing mode, Stage A
        cleanup) while bone_separation_enabled is True.
        """
        if not self.separated_bones:
            return
        for bone in self.separated_bones:
            raw = bone.get('raw_mesh')
            if raw is None or raw.n_points == 0:
                continue
            mesh = self._apply_smoothing(raw.copy(deep=True))
            if mesh is None or mesh.n_points == 0:
                mesh = raw.copy(deep=True)
            mesh = self._close_surface(mesh)
            if getattr(self, 'separation_apply_stage_a', False):
                result = self._apply_stage_a(mesh)
                if result is not None and result.n_points > 0:
                    mesh = result
            bone['mesh'] = mesh
            old_actor = bone.get('actor')
            if old_actor is not None:
                try:
                    self.plotter.remove_actor(old_actor)
                except Exception:
                    pass
            new_actor = self.plotter.add_mesh(
                mesh,
                color=bone.get('color', (1.0, 1.0, 1.0)),
                specular=0.5,
                smooth_shading=True,
            )
            bone['actor'] = new_actor
            if not bone.get('visible', True):
                try:
                    new_actor.SetVisibility(False)
                except Exception:
                    pass
        try:
            self.plotter.update()
        except Exception:
            pass

    def on_clear_separation_clicked(self):
        """Remove per-bone actors and restore the combined mesh view."""
        self._disable_bone_click_selection()
        self._clear_separated_actors()
        self.bone_separation_enabled = False
        self.clear_separation_btn.setEnabled(False)
        self.separation_status_label.setText("Not separated.")
        self._refresh_separation_list()
        self._set_separation_tools_enabled(False)
        self._restore_combined_mesh_after_separation()
        try:
            self.plotter.update()
        except Exception:
            pass


    def _auto_clear_separation_on_remesh(self, reason):
        """Helper used by render paths to drop a stale separation when the
        underlying single mesh is about to be rebuilt (threshold/smoothing/
        series/patient change). Safe to call before UI exists."""
        if not getattr(self, 'bone_separation_enabled', False):
            return
        self._clear_separated_actors()
        self.bone_separation_enabled = False
        if hasattr(self, 'clear_separation_btn'):
            self.clear_separation_btn.setEnabled(False)
        if hasattr(self, 'separation_status_label'):
            self.separation_status_label.setText(
                f"Cleared ({reason}). Click 'Separate Bones' to redo."
            )
        self._set_separation_tools_enabled(False)
        if hasattr(self, 'bone_list_widget'):
            self._refresh_separation_list()
        # Clear restore state
        self._restore_undo_stack = []
        if hasattr(self, 'restore_undo_btn'):
            self.restore_undo_btn.setEnabled(False)

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
                self.separation_status_label.setText(f"복원 중: {n}개 뼈…")
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

    def _get_bone_series_data(self, bone_entry):
        """뼈 entry에 해당하는 HU 배열, spacing, meta를 반환."""
        series_idx = bone_entry.get('series_index')
        if (series_idx is not None
                and hasattr(self, 'all_series_data')
                and series_idx < len(self.all_series_data)):
            sd = self.all_series_data[series_idx]
            return sd['image_hu'], sd['spacing'], sd.get('meta') or {}
        return self.current_image_hu, self.current_spacing, self.current_meta_info or {}

    def _bone_pts_to_grid_coords(self, bone_entry, pts):
        """Fusion 모드에서 mesh 좌표를 원래 시리즈 grid 좌표로 역변환.

        Single 모드나 base series면 변환 없이 그대로 반환.
        """
        series_idx = bone_entry.get('series_index')
        if (not getattr(self, 'fusion_enabled', False)
                or series_idx is None
                or series_idx == self.base_series_index):
            return pts

        _, _, meta = self._get_bone_series_data(bone_entry)
        base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(meta)

        if T_base is None or T_i is None:
            return pts

        T_composite = np.linalg.inv(T_base) @ T_i
        T_inv = np.linalg.inv(T_composite)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (T_inv @ pts_h.T).T[:, :3]

    def _grid_coords_to_bone_frame(self, bone_entry, pts):
        """원래 시리즈 grid 좌표를 fusion base frame으로 변환 (역변환의 반대)."""
        series_idx = bone_entry.get('series_index')
        if (not getattr(self, 'fusion_enabled', False)
                or series_idx is None
                or series_idx == self.base_series_index):
            return pts

        _, _, meta = self._get_bone_series_data(bone_entry)
        base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(meta)

        if T_base is None or T_i is None:
            return pts

        T_composite = np.linalg.inv(T_base) @ T_i
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (T_composite @ pts_h.T).T[:, :3]

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

        closed = self._hu_region_grow(bone_local, local_hu, exclusion,
                                       restore_thr, iters)

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

    def _undo_merge(self, merged_uid, original_snapshots, cam_pos):
        """Merge 작업을 되돌림: 합쳐진 뼈를 제거하고 원본 뼈들을 복원."""
        # 합쳐진 뼈 제거
        merged_bone = self._bone_by_uid(merged_uid)
        if merged_bone is not None:
            actor = merged_bone.get('actor')
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.separated_bones = [
                b for b in self.separated_bones if b.get('uid') != merged_uid
            ]

        # 원본 뼈들 복원
        for snap in original_snapshots:
            mesh = snap['mesh']
            color = snap.get('color', (1, 1, 1))
            actor = self.plotter.add_mesh(
                mesh, color=color,
                specular=0.5, smooth_shading=True,
                reset_camera=False,
            )
            restored = {
                'uid': snap['uid'],
                'id': snap.get('id', 0),
                'mesh': mesh,
                'actor': actor,
                'visible': snap.get('visible', True),
                'color': color,
                'voxel_count': snap.get('voxel_count', mesh.n_cells),
                'name': snap['name'],
                'series_index': snap.get('series_index'),
            }
            if snap.get('raw_mesh') is not None:
                restored['raw_mesh'] = snap['raw_mesh']
            if not restored['visible']:
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass
            self.separated_bones.append(restored)

        self._refresh_separation_list()
        self.plotter.camera_position = cam_pos
        self.plotter.render()
        if hasattr(self, 'separation_status_label'):
            names = [s['name'] for s in original_snapshots]
            self.separation_status_label.setText(
                f"Undo merge → {len(names)}개 뼈 복원: {', '.join(names[:3])}"
                + (f" (+{len(names)-3})" if len(names) > 3 else "")
            )

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

        # ── 다른 뼈 exclusion mask ──
        # HU CC + Voronoi 기반 (composite local_hu 사용 → cross-series 정확)
        target_uids = {b.get('uid', '') for b in bone_entries}
        exclusion = self._build_other_bones_exclusion(
            target_uids, (lz, ly, lx), base_spacing, (iz0, iy0, ix0),
            local_hu=local_hu, threshold=self.current_min_threshold,
        )

        # ── 각 뼈의 solid volume: local_mask에서 mesh seed에 연결된 CC ──
        # mesh surface points → seed voxels, 그 seed에 연결된
        # HU≥threshold connected component = 실제 뼈 볼륨
        struct6 = generate_binary_structure(3, 1)  # 6-connectivity
        bone_volumes = []  # 각 뼈의 solid mask
        total_seed_vox = 0

        # local_mask의 connected components (한 번만 계산)
        labeled_hu, n_hu_comp = ndi_label(local_mask, structure=struct26)

        for pts in all_pts:
            vix = np.round(pts[:, 0] / sx).astype(int) - ix0
            viy = np.round(pts[:, 1] / sy).astype(int) - iy0
            viz = np.round(pts[:, 2] / sz).astype(int) - iz0

            valid = ((vix >= 0) & (vix < lx) &
                     (viy >= 0) & (viy < ly) &
                     (viz >= 0) & (viz < lz))
            vix, viy, viz = vix[valid], viy[valid], viz[valid]

            if len(vix) > 0:
                total_seed_vox += len(vix)
                # seed가 속한 HU connected component labels 수집
                seed_labels = set(labeled_hu[viz, viy, vix])
                seed_labels.discard(0)

                if seed_labels:
                    bone_vol = np.isin(labeled_hu, list(seed_labels))
                else:
                    # seed가 threshold 미만인 경우: seed 자체만 사용
                    bone_vol = np.zeros((lz, ly, lx), dtype=bool)
                    bone_vol[viz, viy, vix] = True

                bone_vol = bone_vol & ~exclusion
                bone_volumes.append(bone_vol)

        if not bone_volumes:
            print("[Merge & Fill] No bone volumes found")
            return None

        # 전체 합침
        bone_combined = np.zeros((lz, ly, lx), dtype=bool)
        for bv in bone_volumes:
            bone_combined |= bv

        n_series = len(involved_series)
        print(f"[Merge & Fill] {n_series} series, local ({lx}×{ly}×{lz}), "
              f"seed={total_seed_vox}, bones={len(bone_volumes)}, "
              f"exclusion={exclusion.sum()}, bone={bone_combined.sum()}")

        # ── Gap 채우기: 양쪽 뼈의 dilation이 겹치는 부분만 bridge ──
        # 각 뼈를 iters만큼 dilate → 두 뼈의 dilated가 겹치는 영역 = gap
        # 이 방식은 뼈 외곽을 변형하지 않고 사이 공간만 채움
        original_bone = bone_combined.copy()

        if len(bone_volumes) >= 2:
            # 각 뼈를 dilate
            dilated_bones = []
            for bv in bone_volumes:
                dilated = binary_dilation(bv, structure=struct6,
                                          iterations=iters)
                dilated_bones.append(dilated)

            # 두 뼈 이상의 dilation이 겹치는 영역 = bridge zone
            # (어떤 두 뼈의 dilated가 겹치면 그곳이 gap)
            bridge = np.zeros((lz, ly, lx), dtype=bool)
            for i in range(len(dilated_bones)):
                for j in range(i + 1, len(dilated_bones)):
                    overlap = dilated_bones[i] & dilated_bones[j]
                    bridge |= overlap

            # bridge에서 원래 뼈, exclusion 제외 → 순수 gap voxels
            gap_fill = bridge & ~original_bone & ~exclusion

            # gap voxels 중 HU가 유효한 것만 (연조직 침범 방지)
            hu_offset = getattr(self, 'vote_threshold', 20)
            hu_min = self.current_min_threshold - hu_offset
            gap_fill = gap_fill & (local_hu >= hu_min)

            closed = original_bone | gap_fill
        else:
            closed = original_bone

        print(f"[Merge & Fill] After bridge fill(iters={iters}): "
              f"{closed.sum()} vox "
              f"(+{closed.sum() - original_bone.sum()} gap filled)")

        # ── Re-mesh (base grid 좌표계) ──
        # 원래 뼈 영역: 실제 HU 그대로 사용 → marching cubes가
        # sub-voxel 보간으로 원래 surface 위치를 복원함.
        # gap-fill 영역: thr+1 강제 (새로 추가된 bridge material)
        # 비-뼈 영역: thr 미만으로 강제 (surface 경계 제한)
        thr = self.current_min_threshold
        local_values = local_hu.copy().astype(np.int16)
        # gap fill voxels: threshold 이상 강제
        gap_mask = closed & ~original_bone
        if gap_mask.any():
            local_values[gap_mask] = np.int16(thr + 1)
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
