import os
from datetime import datetime

import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.constants import BASE_DATA_DIR, SESSION_FORMAT, SESSION_VERSION

# Canonical names of AI-segmented bones. Only these (or bones tagged
# source=='ai') are saved/restored; legacy HU/threshold bones are dropped.
_AI_BONE_NAMES = {
    "Femur_L", "Femur_R", "Hip_L", "Hip_R", "Sacrum", "Patella_L",
    "Patella_R", "Tibia_L", "Tibia_R", "Fibula_L", "Fibula_R",
    "Talus_L", "Talus_R", "Calcaneus_L", "Calcaneus_R", "Tarsals_L",
    "Tarsals_R", "Metatarsals_L", "Metatarsals_R", "Phalanges_L", "Phalanges_R",
}


def _is_ai_bone(bone):
    """True if a bone dict / saved state belongs to the AI segmentation."""
    return (bone.get('source') == 'ai'
            or bone.get('name', '') in _AI_BONE_NAMES)


class SessionIoMixin:
    def on_save_session_clicked(self):
        if not self.all_series_data:
            QMessageBox.warning(self, "No Session", "Load a patient before saving a session.")
            return

        meta = self.current_meta_info or {}
        name = "".join(c for c in str(meta.get('patient_name', 'patient'))
                       if c.isalnum() or c in (' ', '-', '_')).strip() or 'patient'
        date_str = str(meta.get('study_date', ''))
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        default_name = f"{name}_{date_str}_{timestamp}_session.json".replace("__", "_")

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Session",
            default_name,
            "Session file (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith('.json'):
            path += '.json'

        try:
            # mesh 파일들을 저장할 디렉토리
            mesh_dir = path.rsplit('.', 1)[0] + '_meshes'
            state = self._collect_session_state(mesh_dir=mesh_dir)
            from app.constants import APP_VERSION
            state['app_version'] = APP_VERSION   # dev version that created this session
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Session Saved",
                                    f"Session saved to:\n{path}")
        except Exception as e:
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Save Failed", f"Could not save session:\n{e}")


    def on_load_session_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            "",
            "Session file (*.json)",
        )
        if not path:
            return
        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not read session:\n{e}")
            return

        if state.get('format') != SESSION_FORMAT:
            QMessageBox.warning(self, "Wrong Format",
                                "This file does not look like a Stanford Medicine session.")
            return
        if int(state.get('version', 0)) > SESSION_VERSION:
            QMessageBox.warning(
                self, "Newer Format",
                f"This session was saved by a newer app version "
                f"(v{state.get('version')}). Some fields may be ignored."
            )

        try:
            self._apply_session_state(state, session_path=path)
            ver = str(state.get('app_version', '')) or '—'
            QMessageBox.information(self, "Session Loaded",
                                    f"Restored session (app {ver}) from:\n{path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Apply Failed",
                                 f"Session loaded but could not be fully applied:\n{e}")


    def _save_mesh_to_dir(self, mesh, mesh_dir, filename):
        """mesh를 mesh_dir에 .vtk로 저장하고, 상대 경로를 반환."""
        if mesh is None or mesh.n_points == 0:
            return None
        os.makedirs(mesh_dir, exist_ok=True)
        fpath = os.path.join(mesh_dir, filename)
        mesh.save(fpath)
        return filename  # JSON에는 상대 경로만 기록

    def _collect_session_state(self, mesh_dir=None):
        """Snapshot every user-controllable parameter into a JSON-safe dict."""
        meta = self.current_meta_info or {}

        # Cropping bounds: a 6-tuple is enough to reconstruct everything
        # (clip_box, last_box_bounds, and the box widget).
        crop_bounds = None
        if self.last_box_bounds is not None:
            crop_bounds = [float(v) for v in self.last_box_bounds]

        # Camera state — list of three (x, y, z) tuples
        try:
            cam = self.plotter.camera_position
            camera = [[float(v) for v in row] for row in cam]
        except Exception:
            camera = None

        landmarks = []
        for e in self.landmark_data:
            item = {
                'name': e['name'],
                'grid': [float(v) for v in e['grid']],
                'memo': str(e.get('memo', '')),
            }
            if e.get('lps') is not None:
                item['lps'] = [float(v) for v in e['lps']]
            if e.get('ras') is not None:
                item['ras'] = [float(v) for v in e['ras']]
            landmarks.append(item)

        state = {
            'format': SESSION_FORMAT,
            'version': SESSION_VERSION,
            'saved_at': datetime.now().isoformat(timespec='seconds'),

            'patient': {
                'patient_id': str(self.patient_combo.currentText()),
                'patient_name': str(meta.get('patient_name', '')),
                'study_date': str(meta.get('study_date', '')),
                'series_uid': str(meta.get('series_uid', '')),
                'slice_thickness': meta.get('slice_thickness', None),
                'series_description': str(meta.get('series_description', '')),
            },

            'render': {
                'min_threshold': int(self.current_min_threshold),
                'smooth_index': int(self.smooth_combo.currentIndex()),
                'smooth_text': str(self.smooth_combo.currentText()),
            },

            'particle_removal': {
                'voxel_enabled': bool(self.particle_removal_enabled),
                # v2: Stage C is morphological opening
                'opening_iterations': int(self.opening_iterations),
                'opening_connectivity': int(self.opening_connectivity),
                'mesh_cleanup_enabled': bool(self.mesh_cleanup_enabled),
                'keep_largest_only': bool(self.keep_largest_only),
                'min_fragment_faces': int(self.min_fragment_faces),
            },

            'cropping': {
                'enabled': bool(self.crop_checkbox.isChecked()),
                'hide_handles': bool(self.hide_handles_checkbox.isChecked()),
                'bounds': crop_bounds,
            },

            'landmarks': {
                'counter': int(self.landmark_counter),
                'coord_system': str(self.landmark_coord_system),
                'points': landmarks,
            },

            'view': {
                'show_axes': bool(self.axes_checkbox.isChecked()),
            },
            'fusion': {
                'enabled': bool(self.fusion_enabled),
                'base_series_index': int(self.base_series_index),
                'mako_only': bool(self.mako_only_mode),
                'include_flags': [bool(x) for x in self.fusion_include_flags],
            },
            'camera': camera,
        }

        # ---- Bone separation state (v5) ----
        if (getattr(self, 'bone_separation_enabled', False)
                and self.separated_bones and mesh_dir is not None):
            bones_state = []
            for i, bone in enumerate(self.separated_bones):
                # Only persist AI-segmented bones; skip legacy HU/ivory bones.
                if not _is_ai_bone(bone):
                    continue
                # Persist the pristine full surface (raw_mesh) — the crop is a
                # view-only clip that must not be baked into the saved mesh.
                full_mesh = bone.get('raw_mesh') or bone.get('mesh')
                mesh_file = self._save_mesh_to_dir(
                    full_mesh, mesh_dir, f"bone_{i:03d}.vtk")
                raw_file = self._save_mesh_to_dir(
                    full_mesh, mesh_dir, f"bone_{i:03d}_raw.vtk")
                bones_state.append({
                    'uid': bone.get('uid', ''),
                    'id': bone.get('id', 0),
                    'name': bone.get('name', ''),
                    'visible': bone.get('visible', True),
                    'color': list(bone.get('color', (1, 1, 1))),
                    'voxel_count': int(bone.get('voxel_count', 0)),
                    'series_index': bone.get('series_index'),
                    'source': bone.get('source', ''),   # 'ai' for AI-segmented bones
                    'mesh_file': mesh_file,
                    'raw_mesh_file': raw_file,
                })

            # Undo stack
            undo_state = []
            undo_stack = getattr(self, '_restore_undo_stack', [])
            for j, entry in enumerate(undo_stack):
                if (isinstance(entry, tuple) and len(entry) == 3
                        and entry[0] == '__merge__'):
                    # Merge undo: ('__merge__', merged_uid, [snapshots])
                    _, merged_uid, snapshots = entry
                    snap_state = []
                    for k, snap in enumerate(snapshots):
                        sfile = self._save_mesh_to_dir(
                            snap.get('mesh'), mesh_dir,
                            f"undo_{j:03d}_snap_{k:03d}.vtk")
                        rfile = self._save_mesh_to_dir(
                            snap.get('raw_mesh'), mesh_dir,
                            f"undo_{j:03d}_snap_{k:03d}_raw.vtk")
                        snap_state.append({
                            'uid': snap.get('uid', ''),
                            'id': snap.get('id', 0),
                            'name': snap.get('name', ''),
                            'visible': snap.get('visible', True),
                            'color': list(snap.get('color', (1, 1, 1))),
                            'voxel_count': int(snap.get('voxel_count', 0)),
                            'series_index': snap.get('series_index'),
                            'mesh_file': sfile,
                            'raw_mesh_file': rfile,
                        })
                    undo_state.append({
                        'type': 'merge',
                        'merged_uid': merged_uid,
                        'snapshots': snap_state,
                    })
                else:
                    # Regular undo: (uid, old_mesh)
                    uid, old_mesh = entry
                    ufile = self._save_mesh_to_dir(
                        old_mesh, mesh_dir, f"undo_{j:03d}.vtk")
                    undo_state.append({
                        'type': 'restore',
                        'uid': uid,
                        'mesh_file': ufile,
                    })

            state['bone_separation'] = {
                'enabled': True,
                'restore_method': 'hu_grow',
                'restore_iterations': int(getattr(self, 'restore_iterations', 10)),
                'hu_offset': int(getattr(self, 'vote_threshold', 50)),
                'merge_fill_iterations': int(getattr(self, 'merge_fill_iterations', 5)),
                'bones': bones_state,
                'undo_stack': undo_state,
            }

        return state


    def _apply_session_state(self, state, session_path=None):
        """Restore all user-controllable parameters from a state dict."""
        self._loading_session = True
        try:
            # ---- 1) Ensure the right patient is loaded ----
            target_patient = state.get('patient', {}).get('patient_id', '')
            current_patient = self.patient_combo.currentText()
            need_load = False
            if target_patient and target_patient != current_patient:
                idx = self.patient_combo.findText(target_patient)
                if idx < 0:
                    raise RuntimeError(
                        f"Patient '{target_patient}' not found in {BASE_DATA_DIR}/"
                    )
                self.patient_combo.setCurrentIndex(idx)
                need_load = True
            elif not self.all_series_data:
                need_load = True
            if need_load:
                # on_load_clicked() runs synchronously and auto-selects series 0
                self.on_load_clicked()
                if not self.all_series_data:
                    raise RuntimeError("Patient load failed; aborting session restore.")

            # ---- 2) Pick the matching series ----
            target_uid = state.get('patient', {}).get('series_uid', '')
            target_st = state.get('patient', {}).get('slice_thickness', None)
            picked = -1
            for i, sd in enumerate(self.all_series_data):
                m = sd.get('meta', {})
                if str(m.get('series_uid', '')) == target_uid:
                    if target_st is None or (
                        m.get('slice_thickness') is not None
                        and abs(float(m['slice_thickness']) - float(target_st)) < 1e-6
                    ):
                        picked = i
                        break
            if picked < 0:
                # Could not find exact UID+ST match; keep current series but warn
                print("[Session] Warning: series UID not found, using current series")
            else:
                if self.series_combo.currentIndex() != picked:
                    self.series_combo.setCurrentIndex(picked)
                    # on_series_switched also rebuilds the base mesh

            # ---- 3) Restore render parameters (signals blocked) ----
            render = state.get('render', {})
            min_th = int(render.get('min_threshold', self.current_min_threshold))
            self._set_widget_value(self.min_slider, min_th)
            self._set_widget_value(self.min_spinbox, min_th)
            self.current_min_threshold = min_th

            smooth_idx = int(render.get('smooth_index', 0))
            if 0 <= smooth_idx < self.smooth_combo.count():
                self.smooth_combo.blockSignals(True)
                self.smooth_combo.setCurrentIndex(smooth_idx)
                self.smooth_combo.blockSignals(False)

            # ---- 4) Particle removal ----
            pr = state.get('particle_removal', {})
            self.particle_removal_enabled = bool(pr.get('voxel_enabled',
                                                        self.particle_removal_enabled))
            # v2 keys; if loading an old v1 session these are absent and the
            # current defaults (iterations=1, connectivity=1) are kept. The
            # legacy 'min_particle_volume_mm3' field, if present, is ignored.
            self.opening_iterations = int(pr.get('opening_iterations',
                                                 self.opening_iterations))
            self.opening_iterations = max(0, min(5, self.opening_iterations))
            self.opening_connectivity = int(pr.get('opening_connectivity',
                                                   self.opening_connectivity))
            if self.opening_connectivity not in (1, 2, 3):
                self.opening_connectivity = 1
            self.mesh_cleanup_enabled = bool(pr.get('mesh_cleanup_enabled',
                                                    self.mesh_cleanup_enabled))
            self.keep_largest_only = bool(pr.get('keep_largest_only',
                                                 self.keep_largest_only))
            self.min_fragment_faces = int(pr.get('min_fragment_faces',
                                                 self.min_fragment_faces))

            self._set_widget_checked(self.voxel_cleanup_checkbox,
                                     self.particle_removal_enabled)
            self.opening_iter_spinbox.setEnabled(self.particle_removal_enabled)
            self.opening_conn_combo.setEnabled(self.particle_removal_enabled)
            self._set_widget_value(self.opening_iter_spinbox,
                                   int(self.opening_iterations))
            # Map connectivity back to combo index
            self.opening_conn_combo.blockSignals(True)
            for i in range(self.opening_conn_combo.count()):
                if self.opening_conn_combo.itemData(i) == self.opening_connectivity:
                    self.opening_conn_combo.setCurrentIndex(i)
                    break
            self.opening_conn_combo.blockSignals(False)

            self._set_widget_checked(self.mesh_cleanup_checkbox,
                                     self.mesh_cleanup_enabled)
            self.keep_largest_checkbox.setEnabled(self.mesh_cleanup_enabled)
            self._set_widget_checked(self.keep_largest_checkbox,
                                     self.keep_largest_only)
            self.fragment_faces_spinbox.setEnabled(
                self.mesh_cleanup_enabled and not self.keep_largest_only
            )
            self._set_widget_value(self.fragment_faces_spinbox,
                                   self.min_fragment_faces)

            # ---- 4b) Fusion ----
            fusion_state = state.get('fusion', {})
            self.fusion_enabled = bool(fusion_state.get('enabled', self.fusion_enabled))
            self.base_series_index = int(
                fusion_state.get('base_series_index', self.base_series_index)
            )
            self.base_series_index = max(
                0,
                min(self.base_series_index, max(0, len(self.all_series_data) - 1)),
            )
            self.mako_only_mode = bool(fusion_state.get('mako_only', self.mako_only_mode))
            saved_flags = fusion_state.get('include_flags', None)
            if isinstance(saved_flags, list) and len(saved_flags) == len(self.all_series_data):
                self.fusion_include_flags = [bool(x) for x in saved_flags]
            else:
                # Fall back to auto-detect or all-on
                if self.mako_only_mode:
                    self.fusion_include_flags = self._autodetect_mako_flags()
                else:
                    self.fusion_include_flags = [True] * len(self.all_series_data)
            self._set_widget_checked(self.fusion_checkbox, self.fusion_enabled)
            self._set_widget_checked(self.mako_only_checkbox, self.mako_only_mode)
            self._refresh_series_include_list()
            # Apply mutual-exclusion enabled state. Cropping and landmark
            # picking remain available in fusion mode; click-to-remove and
            # bone separation are single-series-only.
            if hasattr(self, 'pick_btn'):
                self.pick_btn.setEnabled(not self.fusion_enabled)

            # ---- 5) Cropping ----
            crop = state.get('cropping', {})
            crop_enabled = bool(crop.get('enabled', False))
            crop_bounds = crop.get('bounds', None)
            # Reset existing widgets first
            self.plotter.clear_box_widgets()
            if crop_bounds and len(crop_bounds) == 6:
                bounds_tuple = tuple(float(v) for v in crop_bounds)
                self.cropping_bounds = pv.Box(bounds=bounds_tuple)
                self.last_box_bounds = bounds_tuple
            else:
                self.cropping_bounds = None
                self.last_box_bounds = None
            self._set_widget_checked(self.crop_checkbox, crop_enabled)
            if crop_enabled and self.volume_grid is not None:
                box_init = (self.last_box_bounds
                            if self.last_box_bounds is not None
                            else self.volume_grid.bounds)
                self.plotter.add_box_widget(
                    self.on_box_cropped,
                    bounds=box_init,
                    factor=1.0,
                )
                self.hide_handles_checkbox.setEnabled(True)
                hide_handles = bool(crop.get('hide_handles', False))
                self._set_widget_checked(self.hide_handles_checkbox, hide_handles)
                if hide_handles:
                    self.plotter.clear_box_widgets()
            else:
                self.hide_handles_checkbox.setEnabled(False)
                self._set_widget_checked(self.hide_handles_checkbox, False)

            # ---- 6) Rebuild mesh with all restored parameters ----
            # Release the guard so this final rebuild actually runs.
            self._loading_session = False
            self.update_base_mesh()

            # ---- 7) View toggles ----
            view = state.get('view', {})
            self._set_widget_checked(self.axes_checkbox,
                                     bool(view.get('show_axes', True)))

            # ---- 8) Landmarks (after mesh exists so sphere radius is sensible) ----
            self._clear_measurement_visualization()
            # Wipe any existing landmarks
            for actor in self.landmark_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.landmark_actors.clear()
            self.landmark_data.clear()
            self.landmark_counter = 0

            lm_section = state.get('landmarks', {})
            self.landmark_counter = int(lm_section.get('counter', 0))
            coord_sys = str(lm_section.get('coord_system', 'grid'))
            # Update combo to match (without triggering a redundant refresh)
            self.landmark_coord_combo.blockSignals(True)
            for i in range(self.landmark_coord_combo.count()):
                if self.landmark_coord_combo.itemData(i) == coord_sys:
                    self.landmark_coord_combo.setCurrentIndex(i)
                    self.landmark_coord_system = coord_sys
                    break
            self.landmark_coord_combo.blockSignals(False)

            radius = self._estimate_landmark_radius()
            for lm in lm_section.get('points', []):
                grid = np.asarray(lm.get('grid', [0, 0, 0]), dtype=float)
                entry = {
                    'name': str(lm.get('name', f"L{len(self.landmark_data)+1}")),
                    'grid': grid,
                    'lps': (np.asarray(lm['lps'], dtype=float)
                            if lm.get('lps') is not None else self._grid_to_lps(grid)),
                    'ras': None,
                    'memo': str(lm.get('memo', '')),
                }
                if lm.get('ras') is not None:
                    entry['ras'] = np.asarray(lm['ras'], dtype=float)
                elif entry['lps'] is not None:
                    entry['ras'] = self._lps_to_ras(entry['lps'])
                self.landmark_data.append(entry)

                sphere = pv.Sphere(radius=radius, center=grid)
                actor = self.plotter.add_mesh(
                    sphere,
                    color='red',
                    specular=0.3,
                    smooth_shading=True,
                    pickable=False,
                    name=f"landmark_{len(self.landmark_data) - 1}",
                )
                self.landmark_actors.append(actor)
            # If counter is missing/stale, ensure it covers all loaded points
            if self.landmark_counter < len(self.landmark_data):
                self.landmark_counter = len(self.landmark_data)
            self._refresh_landmark_table()

            # ---- 9) Camera last, so framing matches the saved view ----
            cam = state.get('camera')
            if cam and len(cam) == 3:
                try:
                    self.plotter.camera_position = [
                        tuple(float(v) for v in row) for row in cam
                    ]
                    self._camera_initialized = True
                except Exception as e:
                    print(f"[Session] Camera restore failed: {e}")

            # ---- 10) Bone separation (v5) ----
            sep = state.get('bone_separation')
            if sep and sep.get('enabled') and sep.get('bones') and session_path:
                self._restore_bone_separation_state(sep, session_path)

            self.plotter.update()
        finally:
            self._loading_session = False


    def _set_widget_value(self, widget, value):
        """Set value on a slider/spinbox without firing signals."""
        widget.blockSignals(True)
        try:
            widget.setValue(value)
        finally:
            widget.blockSignals(False)


    def _set_widget_checked(self, widget, checked):
        """Set checkbox state without firing signals."""
        widget.blockSignals(True)
        try:
            widget.setChecked(bool(checked))
        finally:
            widget.blockSignals(False)

    # ===== Bone Separation Session Restore =====

    def _load_mesh_from_dir(self, mesh_dir, filename):
        """mesh_dir에서 .vtk 파일 로드. 없으면 None."""
        if not filename:
            return None
        fpath = os.path.join(mesh_dir, filename)
        if not os.path.isfile(fpath):
            return None
        try:
            return pv.read(fpath)
        except Exception as e:
            print(f"[Session] Failed to load mesh {fpath}: {e}")
            return None

    def _restore_bone_separation_state(self, sep, session_path):
        """세션에서 뼈 분리 상태 복원 (bones + undo stack)."""
        mesh_dir = session_path.rsplit('.', 1)[0] + '_meshes'
        if not os.path.isdir(mesh_dir):
            print(f"[Session] Mesh directory not found: {mesh_dir}")
            return

        # 기존 분리 상태 정리
        self._clear_separated_actors()

        # 설정 복원
        self.restore_method = 'hu_grow'
        self.restore_iterations = int(sep.get('restore_iterations', 10))
        self.vote_threshold = int(sep.get('hu_offset', sep.get('vote_threshold', 50)))
        self.merge_fill_iterations = int(sep.get('merge_fill_iterations', 5))

        # UI 위젯 동기화
        if hasattr(self, 'restore_iter_spinbox'):
            self._set_widget_value(self.restore_iter_spinbox,
                                   self.restore_iterations)
        if hasattr(self, 'vote_threshold_spinbox'):
            self._set_widget_value(self.vote_threshold_spinbox,
                                   self.vote_threshold)
        if hasattr(self, 'merge_fill_iter_spinbox'):
            self._set_widget_value(self.merge_fill_iter_spinbox,
                                   self.merge_fill_iterations)

        # 뼈 복원 — AI로 분리한 뼈만 불러온다 (옛 HU/threshold 분리 뼈는 제외).
        for bone_state in sep.get('bones', []):
            if not _is_ai_bone(bone_state):
                continue
            mesh = self._load_mesh_from_dir(mesh_dir,
                                            bone_state.get('mesh_file'))
            if mesh is None or mesh.n_points == 0:
                continue

            raw_mesh = self._load_mesh_from_dir(mesh_dir,
                                                bone_state.get('raw_mesh_file'))
            color = tuple(bone_state.get('color', [1, 1, 1]))
            visible = bone_state.get('visible', True)

            actor = self.plotter.add_mesh(
                mesh, color=color,
                specular=0.5, smooth_shading=True,
                reset_camera=False,
            )
            if not visible:
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass

            bone = {
                'uid': bone_state.get('uid', self._new_bone_uid()),
                'id': bone_state.get('id', 0),
                'mesh': mesh,
                'actor': actor,
                'visible': visible,
                'color': color,
                'voxel_count': int(bone_state.get('voxel_count', mesh.n_cells)),
                'name': bone_state.get('name', 'Bone'),
                'series_index': bone_state.get('series_index'),
                'source': 'ai',
            }
            if raw_mesh is not None:
                bone['raw_mesh'] = raw_mesh
            self.separated_bones.append(bone)

        # Undo stack 복원
        self._restore_undo_stack = []
        for undo_entry in sep.get('undo_stack', []):
            if undo_entry.get('type') == 'merge':
                # Merge undo
                snapshots = []
                for snap_state in undo_entry.get('snapshots', []):
                    smesh = self._load_mesh_from_dir(
                        mesh_dir, snap_state.get('mesh_file'))
                    if smesh is None:
                        continue
                    raw = self._load_mesh_from_dir(
                        mesh_dir, snap_state.get('raw_mesh_file'))
                    snapshots.append({
                        'uid': snap_state.get('uid', ''),
                        'id': snap_state.get('id', 0),
                        'mesh': smesh,
                        'visible': snap_state.get('visible', True),
                        'color': tuple(snap_state.get('color', [1, 1, 1])),
                        'voxel_count': int(snap_state.get('voxel_count', 0)),
                        'name': snap_state.get('name', ''),
                        'series_index': snap_state.get('series_index'),
                        'raw_mesh': raw,
                    })
                if snapshots:
                    self._restore_undo_stack.append(
                        ('__merge__', undo_entry.get('merged_uid', ''),
                         snapshots)
                    )
            else:
                # Regular restore undo
                umesh = self._load_mesh_from_dir(
                    mesh_dir, undo_entry.get('mesh_file'))
                if umesh is not None:
                    self._restore_undo_stack.append(
                        (undo_entry.get('uid', ''), umesh)
                    )

        # UI 활성화
        self.bone_separation_enabled = True
        self._set_separation_tools_enabled(True)
        if hasattr(self, 'restore_undo_btn'):
            self.restore_undo_btn.setEnabled(bool(self._restore_undo_stack))
        if hasattr(self, 'clear_separation_btn'):
            self.clear_separation_btn.setEnabled(True)
        self._refresh_separation_list()

        # Restored bones are AI bones → hide the ivory threshold volume so only
        # the coloured AI bones remain (same as a fresh AI-first load), and turn
        # on 3D click-to-select.
        if self.separated_bones:
            self.ai_segmentation_active = True
            if hasattr(self, '_hide_volume_for_ai'):
                self._hide_volume_for_ai()
            if hasattr(self, '_enable_bone_click_selection'):
                self._enable_bone_click_selection()
        if hasattr(self, '_update_info_panel'):
            self._update_info_panel()
        try:
            self.plotter.render()
        except Exception:
            pass

        n_bones = len(self.separated_bones)
        n_undo = len(self._restore_undo_stack)
        print(f"[Session] Restored {n_bones} bones, {n_undo} undo entries")
        if hasattr(self, 'separation_status_label'):
            self.separation_status_label.setText(
                f"Session restored: {n_bones} bone(s), {n_undo} undo"
            )

    # ===== Hover Preview (yellow sphere follows mouse) =====

