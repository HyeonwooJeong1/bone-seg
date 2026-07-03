import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QCheckBox, QLabel

from app.constants import MAKO_KEYWORDS

class FusionMixin:
    def _autodetect_mako_flags(self):
        """Return a list[bool] matching all_series_data, with True for series
        whose description matches any of MAKO_KEYWORDS. Falls back to
        all-True when no series matches (don't strand the user with nothing)."""
        flags = []
        any_hit = False
        for sd in self.all_series_data:
            desc = (sd.get('meta', {}) or {}).get('series_description', '') or ''
            d_low = str(desc).lower()
            hit = any(k in d_low for k in MAKO_KEYWORDS)
            flags.append(hit)
            any_hit = any_hit or hit
        if not any_hit:
            # No series carries a Mako keyword: include everything as fallback.
            flags = [True] * len(self.all_series_data)
        return flags


    def _refresh_series_include_list(self):
        """Rebuild the dynamic per-series include checkbox list to match
        the currently loaded patient + fusion_include_flags + mako_only_mode."""
        # Clear previous widgets
        for cb in self.series_include_checkboxes:
            try:
                cb.setParent(None)
                cb.deleteLater()
            except Exception:
                pass
        self.series_include_checkboxes.clear()
        # Drop any stragglers in the layout (separators etc.)
        while self.series_include_list_layout.count():
            item = self.series_include_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        if not self.all_series_data:
            empty = QLabel("<i>Load a patient to see series.</i>")
            empty.setStyleSheet("color: #666;")
            self.series_include_list_layout.addWidget(empty)
            return

        # Ensure flags array length matches the series list
        if len(self.fusion_include_flags) != len(self.all_series_data):
            if self.mako_only_mode:
                self.fusion_include_flags = self._autodetect_mako_flags()
            else:
                self.fusion_include_flags = [True] * len(self.all_series_data)

        for i, sd in enumerate(self.all_series_data):
            m = sd.get('meta', {}) or {}
            desc = m.get('series_description', 'N/A')
            st = m.get('slice_thickness', '?')
            nslices = sd['image_hu'].shape[0]
            sub_idx = m.get('sub_idx', 0)
            sub_part = f" #{sub_idx}" if sub_idx else ""
            zmin = m.get('z_min')
            zmax = m.get('z_max')
            z_part = (
                f"  z[{zmin:.0f}…{zmax:.0f}]"
                if zmin is not None and zmax is not None and zmin != zmax
                else ""
            )
            cb = QCheckBox(
                f"[{i}] {desc}{sub_part} — ST {st}mm ({nslices} sl.){z_part}"
            )
            cb.setChecked(bool(self.fusion_include_flags[i]))
            cb.setProperty("series_index", i)
            cb.stateChanged.connect(self._on_series_include_toggled)
            self.series_include_list_layout.addWidget(cb)
            self.series_include_checkboxes.append(cb)


    def _on_series_include_toggled(self, _state):
        """User flipped one of the per-series include checkboxes."""
        sender = self.sender()
        if sender is None:
            return
        idx = sender.property("series_index")
        if idx is None or idx < 0 or idx >= len(self.fusion_include_flags):
            return
        was = self.fusion_include_flags[idx]
        now = bool(sender.isChecked())
        if was == now:
            return
        self.fusion_include_flags[idx] = now

        # If the user disabled the current base series, pick another included
        # one to keep landmark / coord semantics sane.
        if not now and idx == self.base_series_index:
            new_base = next(
                (j for j, f in enumerate(self.fusion_include_flags) if f),
                None,
            )
            if new_base is not None and new_base != self.base_series_index:
                self._switching_series = True
                self.series_combo.setCurrentIndex(new_base)
                self._switching_series = False
                self.base_series_index = new_base
                self._invalidate_landmarks_on_base_change()
                # Force ImageData rebuild for new base
                series = self.all_series_data[new_base]
                self.current_image_hu = series['image_hu']
                self.current_spacing = series['spacing']
                self.current_meta_info = series['meta']
                self.volume_grid = self._build_image_data(
                    self.current_image_hu, self.current_spacing
                )

        if self.fusion_enabled and self.all_series_data:
            self._camera_initialized = False
            self.update_base_mesh()

        # 2D slice viewer should add/remove the toggled series' panel.
        if hasattr(self, '_refresh_slice_viewer'):
            self._refresh_slice_viewer()


    def on_mako_only_toggled(self, state):
        """Apply / un-apply the auto-detect Mako filter."""
        self.mako_only_mode = (state == Qt.Checked)
        if self.mako_only_mode:
            self.fusion_include_flags = self._autodetect_mako_flags()
        else:
            self.fusion_include_flags = [True] * len(self.all_series_data)
        # Update each checkbox's state in-place (no rebuild needed)
        for cb in self.series_include_checkboxes:
            idx = cb.property("series_index")
            if idx is None or idx < 0 or idx >= len(self.fusion_include_flags):
                continue
            cb.blockSignals(True)
            cb.setChecked(bool(self.fusion_include_flags[idx]))
            cb.blockSignals(False)
        if self.fusion_enabled and self.all_series_data:
            self._camera_initialized = False
            self.update_base_mesh()

        # Mako filter may have toggled multiple series at once — refresh 2D.
        if hasattr(self, '_refresh_slice_viewer'):
            self._refresh_slice_viewer()


    def on_fusion_toggled(self, state):
        """Switch between fusion (all series at once) and single-series mode."""
        self.fusion_enabled = (state == Qt.Checked)

        # Only click-to-remove and bone-separation are single-series-only.
        # Cropping and landmark picking work in fusion mode.
        in_fusion = self.fusion_enabled
        if in_fusion:
            if hasattr(self, 'pick_btn') and self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)
        if hasattr(self, 'pick_btn'):
            self.pick_btn.setEnabled(not in_fusion)

        # series_combo selects either (a) the active series in single mode or
        # (b) the base reference frame in fusion mode. Always allow both.
        if hasattr(self, 'series_combo'):
            current_idx = self.series_combo.currentIndex()
            self.base_series_index = max(0, current_idx)

        # Landmarks placed in single mode use the previously active series' grid,
        # which is identical to the base series' grid in fusion mode (because the
        # base series is the one anchored at identity). So as long as the user
        # doesn't change base_series_index, landmarks stay valid.

        self._camera_initialized = False
        if self.all_series_data:
            self.update_base_mesh()

        # Switching between single (1 panel) and fusion (N panels) modes
        # changes the 2D viewer layout entirely.
        if hasattr(self, '_refresh_slice_viewer'):
            self._refresh_slice_viewer()


    def on_series_switched(self, index):
        """Series-combo handler.

        - Single mode: swap the active series (rebuild volume_grid, contour, etc.)
        - Fusion mode:  the selected entry becomes the BASE reference frame.
                        All series are re-transformed into the new base's grid.
        """
        if self._switching_series or index < 0 or index >= len(self.all_series_data):
            return

        series = self.all_series_data[index]
        self.current_image_hu = series['image_hu']
        self.current_spacing = series['spacing']
        self.current_meta_info = series['meta']

        # Rebuild the VTK volume grid for this series (used by single-mode and
        # by fusion when this series is the base).
        self.volume_grid = self._build_image_data(
            self.current_image_hu, self.current_spacing
        )

        # In fusion mode this combo selects the base reference frame.
        if self.fusion_enabled:
            if self.base_series_index != index:
                # Base frame changed → landmark grid coords are no longer meaningful.
                self._invalidate_landmarks_on_base_change()
            self.base_series_index = index

        # Reset camera so new volume is framed properly
        self._camera_initialized = False

        self.update_base_mesh()

        # Reset crop widget if active (single mode only — disabled while fusing)
        if not self.fusion_enabled and self.crop_checkbox.isChecked():
            self.plotter.clear_box_widgets()
            self.plotter.add_box_widget(self.on_box_cropped, bounds=self.volume_grid.bounds)

        # Re-compute LPS/RAS for any existing landmarks using the new series geometry.
        # Grid coords stay valid (same physical voxel grid) and sphere actors are
        # unchanged. Only the DICOM-coord representations need refreshing.
        if hasattr(self, 'landmark_data') and self.landmark_data:
            for entry in self.landmark_data:
                entry['lps'] = self._grid_to_lps(entry['grid'])
                entry['ras'] = self._lps_to_ras(entry['lps'])
            self._refresh_landmark_table()

        # Mirror the new active series in the 2D slice viewer (if built).
        if hasattr(self, '_refresh_slice_viewer'):
            self._refresh_slice_viewer()

        # Update info window
        meta = self.current_meta_info
        if meta:
            name = meta.get('patient_name', 'N/A')
            desc = meta.get('series_description', 'N/A')
            date_str = meta.get('study_date', 'N/A')
            modality = meta.get('modality', 'N/A')
            uid = meta.get('series_uid', 'N/A')
            st = meta.get('slice_thickness', 'N/A')
            if date_str and len(date_str) == 8:
                date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            self.setWindowTitle(f"{name} — {desc} (ST:{st}mm) — {date_str}")
            self.info_text.setHtml(f"""
            <h3>Patient Information</h3>
            <ul>
                <li><b>Patient Name:</b> {name}</li>
                <li><b>Study Date:</b> {date_str}</li>
                <li><b>Modality:</b> {modality}</li>
            </ul>
            <hr>
            <h3>Scan Information</h3>
            <ul>
                <li><b>Series Description:</b> {desc}</li>
                <li><b>Slice Thickness:</b> {st} mm</li>
                <li><b>Series UID:</b> {uid}</li>
                <li><b>Volume Size:</b> {self.current_image_hu.shape} (Z, Y, X)</li>
                <li><b>Physical Spacing:</b> {self.current_spacing[0]:.2f} x {self.current_spacing[1]:.2f} x {self.current_spacing[2]:.2f} mm</li>
            </ul>
            """)


    def _print_fusion_diagnostics(self):
        """Dump per-series geometry summary to the console. Useful for
        understanding why fused meshes line up (or don't)."""
        print("\n========= [Fusion diagnostics] =========")
        print(f"Total series in patient: {len(self.all_series_data)}")
        print(f"Base series index:       {self.base_series_index}")
        for i, sd in enumerate(self.all_series_data):
            m = sd.get('meta', {}) or {}
            ipp = m.get('ipp_first')
            rd = m.get('row_dir')
            cd = m.get('col_dir')
            nd = m.get('normal_dir')
            sh = sd['image_hu'].shape
            sp = sd['spacing']
            star = " *" if i == self.base_series_index else "  "
            print(
                f"{star}[{i}] desc='{m.get('series_description','N/A')}' "
                f"ST={m.get('slice_thickness','?')}mm  "
                f"shape={sh}  spacing(z,y,x)={tuple(round(float(v),3) for v in sp)}"
            )
            if ipp is not None and rd is not None:
                print(f"      IPP_first   = {[round(float(v),2) for v in ipp]}")
                print(f"      row_dir     = {[round(float(v),3) for v in rd]}")
                print(f"      col_dir     = {[round(float(v),3) for v in cd]}")
                print(f"      normal_dir  = {[round(float(v),3) for v in nd]}")
            else:
                print("      (no DICOM geometry — will be SKIPPED in fusion)")
        print("========================================\n")


    def _update_fused_meshes(self):
        """Build & render meshes for every series, aligned in the base series'
        grid coordinate frame. 각 시리즈의 mesh를 mesh-level CC로 뼈 분리 표시.
        """
        # 기존 actors 정리
        self._clear_separated_actors()
        for actor in self.fusion_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.fusion_actors.clear()
        self.fusion_meshes.clear()
        if self.current_mesh_actor is not None:
            try:
                self.plotter.remove_actor(self.current_mesh_actor)
            except Exception:
                pass
            self.current_mesh_actor = None

        if not self.all_series_data:
            return

        if not getattr(self, '_fusion_diag_printed', False):
            self._print_fusion_diagnostics()
            self._fusion_diag_printed = True

        if len(self.series_volume_grids) != len(self.all_series_data):
            self.series_volume_grids = [None] * len(self.all_series_data)

        base_idx = max(0, min(self.base_series_index, len(self.all_series_data) - 1))
        base_meta = self.all_series_data[base_idx]['meta']
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_base_inv = np.linalg.inv(T_base) if T_base is not None else None

        crop_active = (
            self.cropping_bounds is not None and self.crop_checkbox.isChecked()
        )

        if len(self.fusion_include_flags) != len(self.all_series_data):
            self.fusion_include_flags = [True] * len(self.all_series_data)

        any_rendered = False
        base_mesh_for_picking = None
        # 전체 시리즈에 걸쳐 bone entry를 누적 (고유 색상 부여)
        all_bone_entries = []

        for i, sd in enumerate(self.all_series_data):
            if not self.fusion_include_flags[i]:
                continue

            image_hu = sd['image_hu']
            spacing = sd['spacing']
            meta = sd['meta']

            grid = self.series_volume_grids[i]
            if grid is None:
                grid = self._build_image_data(image_hu, spacing)
                self.series_volume_grids[i] = grid

            grid.point_data["masked"] = self._compute_masked_values(image_hu)

            try:
                mesh = grid.contour(
                    [self.current_min_threshold], scalars="masked"
                )
            except Exception as e:
                print(f"[Fusion] Series {i} contour failed: {e}")
                continue

            if mesh is None or mesh.n_points == 0:
                continue

            mesh = self._close_surface(mesh)
            mesh = self._apply_smoothing(mesh)
            mesh = self._close_surface(mesh)
            mesh = self._apply_stage_a(mesh)
            if mesh is None or mesh.n_points == 0:
                continue

            # Transform into base frame
            if i != base_idx and T_base_inv is not None:
                T_i = self._series_grid_to_lps_matrix(meta)
                if T_i is not None:
                    T_composite = T_base_inv @ T_i
                    mesh.transform(T_composite, inplace=True)
                else:
                    print(f"[Fusion] Series {i} missing DICOM geometry; skipped.")
                    continue

            if crop_active:
                try:
                    mesh = mesh.clip_box(self.cropping_bounds, invert=False)
                except Exception:
                    pass
                if mesh is None or mesh.n_points == 0:
                    continue

            self.fusion_meshes.append((i, mesh))
            any_rendered = True

            if i == base_idx:
                base_mesh_for_picking = mesh

            # ── Mesh-level CC 뼈 분리 ──
            desc = str(meta.get('series_description', f'S{i}'))[:16]
            try:
                labeled = mesh.connectivity(extraction_mode='all')
                rids = labeled.cell_data.get('RegionId')
            except Exception as e:
                print(f"[Fusion] Series {i} connectivity failed: {e}")
                rids = None

            if rids is not None and len(rids) > 0:
                rids_arr = np.asarray(rids)
                unique_ids, counts = np.unique(rids_arr, return_counts=True)
                min_cells = max(1, int(getattr(self, 'min_bone_voxels', 1)))

                for rid, count in zip(unique_ids, counts):
                    if count < min_cells:
                        continue
                    try:
                        cells_idx = np.where(rids_arr == rid)[0]
                        sub = labeled.extract_cells(cells_idx)
                        if not isinstance(sub, pv.PolyData):
                            sub = sub.extract_surface()
                        for arr in ('RegionId',):
                            if arr in sub.point_data:
                                del sub.point_data[arr]
                            if arr in sub.cell_data:
                                del sub.cell_data[arr]
                    except Exception:
                        continue
                    if sub is None or sub.n_points == 0:
                        continue
                    all_bone_entries.append({
                        'mesh': sub,
                        'cell_count': int(count),
                        'series_index': i,
                        'series_desc': desc,
                    })
            else:
                # CC 실패 시 시리즈 전체를 하나의 bone으로
                all_bone_entries.append({
                    'mesh': mesh,
                    'cell_count': mesh.n_cells,
                    'series_index': i,
                    'series_desc': desc,
                })

        # ── 모든 시리즈의 뼈를 크기순으로 색상 배정 + 표시 ──
        all_bone_entries.sort(key=lambda x: -x['cell_count'])
        colors = self._bone_color_palette(len(all_bone_entries))

        for bone_num, (entry, color) in enumerate(zip(all_bone_entries, colors), start=1):
            sub = entry['mesh']
            actor = self.plotter.add_mesh(
                sub, color=color, specular=0.5, smooth_shading=True
            )
            self.fusion_actors.append(actor)
            self.separated_bones.append({
                'uid': self._new_bone_uid(),
                'id': bone_num,
                'mesh': sub,
                'raw_mesh': sub.copy(deep=True),
                'actor': actor,
                'visible': True,
                'color': color,
                'voxel_count': entry['cell_count'],
                'name': f"[{entry['series_desc']}] Bone {bone_num}",
                'series_index': entry['series_index'],
            })

        # base_mesh for landmark/hover/click-to-remove
        if base_mesh_for_picking is not None:
            self.base_mesh = base_mesh_for_picking
        elif self.fusion_meshes:
            self.base_mesh = self.fusion_meshes[0][1]

        # UI 상태
        self.bone_separation_enabled = True
        if hasattr(self, 'clear_separation_btn'):
            self.clear_separation_btn.setEnabled(True)
        self._set_separation_tools_enabled(True)
        if hasattr(self, 'separation_status_label'):
            self.separation_status_label.setText(
                f"{len(all_bone_entries)} bone(s) across "
                f"{len(self.fusion_meshes)} series"
            )
        self._refresh_separation_list()

        if not self._camera_initialized and any_rendered:
            self.plotter.reset_camera()
            self._camera_initialized = True

        self.plotter.update()


    def _invalidate_landmarks_on_base_change(self):
        """When the user switches the base series in fusion mode, existing
        landmark grid coordinates lose meaning (they were placed in the previous
        base series' grid frame). Wipe them with a friendly notice.
        """
        if not self.landmark_data:
            return
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Landmarks cleared",
                "Switching the base series invalidates existing landmark grid\n"
                "coordinates. Landmarks have been cleared."
            )
        except Exception:
            pass
        for actor in self.landmark_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.landmark_actors.clear()
        self.landmark_data.clear()
        self.landmark_counter = 0
        if hasattr(self, '_refresh_landmark_table'):
            self._refresh_landmark_table()
        if hasattr(self, '_clear_measurement_visualization'):
            self._clear_measurement_visualization()


