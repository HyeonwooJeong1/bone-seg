"""Bone list widget + selection + visibility + rename + 3D click selection.

Methods grouped here:
  - color palette / uid / sanitize helpers
  - separation-tools enable/disable
  - bone list refresh, item label
  - visibility toggle
  - 3D click-to-select (vtkCellPicker + press/release observers)
  - list selection highlight (yellow wireframe)
  - rename (double-click and button)
"""

import re
import uuid

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QInputDialog,
    QListWidgetItem,
    QMessageBox,
)


class BoneListUIMixin:
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
        """Toggle visibility of the selected bone(s). Hide/Show button or 'H' key.

        The selection is cleared afterward so you don't have to click a second
        time to deselect before choosing another bone.
        """
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
            # Update list text (● / ○)
            item.setText(self._bone_list_item_label(bone))
        # Auto-deselect so a second click isn't needed (also clears highlight).
        self.bone_list_widget.clearSelection()
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
        """Mouse-up → if it was a click (not a drag), select the picked bone."""
        if not self.bone_separation_enabled or not self.separated_bones:
            return
        # If landmark picking is active, let that mode handle the click instead.
        if getattr(self, 'landmark_picking_enabled', False):
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
