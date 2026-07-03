"""Connected-component bone separation (single + fusion) and presentation.

Methods grouped here:
  - on_sep_stage_a_toggled, on_min_bone_vox_changed: setting handlers
  - _compute_single_separated_bone_entries: split base mesh by CC
  - _split_mesh_into_bone_entries: shared CC-extraction kernel
  - _compute_fusion_separated_bone_entries: per-series CC, transformed
  - _hide_combined_mesh_for_separation / _restore_combined_mesh_after_separation
  - _show_separated_bones: add per-bone actors
  - on_separate_bones_clicked: top-level UI handler
  - _reapply_mesh_ops_to_separated_bones: re-smooth from raw_mesh
  - on_clear_separation_clicked, _auto_clear_separation_on_remesh
"""

import time

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox


class BoneSeparateMixin:
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
