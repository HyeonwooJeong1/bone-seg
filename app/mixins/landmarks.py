import os
import time
from datetime import datetime

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
)

class LandmarksMixin:
    def on_landmark_picking_toggled(self, checked):
        self.landmark_picking_enabled = checked
        if checked:
            # Mutual exclusion: turn off other picking modes
            if hasattr(self, 'pick_btn') and self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)
            if hasattr(self, 'restore_pick_btn') and self.restore_pick_btn.isChecked():
                self.restore_pick_btn.setChecked(False)
            self.landmark_pick_btn.setText("Pick Landmark: ON")
            self._enable_volume_landmark_picking()
        else:
            self.landmark_pick_btn.setText("Pick Landmark: OFF")
            self._disable_volume_landmark_picking()

    # ──────────────────────────────────────────────────────────────────
    # Volume-based 3D landmark picking (vtkVolumePicker)
    #
    # There's no surface mesh to cell-pick anymore — the bone is a volume
    # whose non-bone voxels are fully transparent. vtkVolumePicker casts a
    # ray and returns the point where it crosses the opacity isosurface
    # (= the bone surface), which is exactly what we want.
    # ──────────────────────────────────────────────────────────────────
    def _enable_volume_landmark_picking(self):
        iren_wrapper = self.plotter.iren
        try:
            self._lm_native_iren = iren_wrapper.interactor
        except AttributeError:
            self._lm_native_iren = iren_wrapper

        # Two pickers: vtkVolumePicker (opacity isosurface = bone surface)
        # with a vtkCellPicker fallback. Some VTK builds give a flaky
        # vtkVolumePicker result with vtkSmartVolumeMapper, so we keep a
        # backup that also handles volumes.
        if getattr(self, '_lm_volume_picker', None) is None:
            vp = vtk.vtkVolumePicker()
            vp.SetTolerance(0.005)
            try:
                vp.SetVolumeOpacityIsovalue(0.5)
                vp.PickCroppingPlanesOff()
            except Exception:
                pass
            self._lm_volume_picker = vp
        if getattr(self, '_lm_cell_picker', None) is None:
            cp = vtk.vtkCellPicker()
            cp.SetTolerance(0.005)
            self._lm_cell_picker = cp

        # The volume actors must be pickable for either picker to hit them.
        for actor in getattr(self, 'volume_actors', []) or []:
            try:
                actor.SetPickable(True)
            except Exception:
                pass

        # Yellow hover sphere. reset_camera=False is CRITICAL: the sphere
        # geometry sits at (0,0,0) but the cropped bone volume is far from
        # the origin, so a default add_mesh would reframe the camera to
        # include (0,0,0) — that's the "view resets every time" bug.
        if self._hover_sphere_actor is None:
            radius = self._estimate_landmark_radius()
            sphere_mesh = pv.Sphere(radius=radius, center=(0.0, 0.0, 0.0))
            # OPAQUE + high ambient: a translucent sphere blends into the
            # opaque volume and effectively disappears (that was the
            # "preview doesn't show" bug). Bright lime distinguishes the
            # hover preview from the red placed landmarks.
            self._hover_sphere_actor = self.plotter.add_mesh(
                sphere_mesh, color=(0.1, 1.0, 0.1), opacity=1.0,
                ambient=0.7, diffuse=0.5, specular=0.2,
                smooth_shading=True, pickable=False, reset_camera=False,
                name="hover_preview_sphere",
            )
        try:
            self._hover_sphere_actor.SetVisibility(False)
        except Exception:
            pass

        # Use the PyVista wrapper's add_observer (proven by the old hover
        # preview) so the events actually fire.
        add_obs = getattr(iren_wrapper, 'add_observer', None)
        if add_obs is None:
            add_obs = self._lm_native_iren.AddObserver
        self._lm_obs_ids = [
            add_obs('LeftButtonPressEvent',   self._lm_on_press),
            add_obs('LeftButtonReleaseEvent', self._lm_on_release),
            add_obs('MouseMoveEvent',         self._lm_on_move),
        ]

    def _disable_volume_landmark_picking(self):
        iren_wrapper = self.plotter.iren
        rem = getattr(iren_wrapper, 'remove_observer', None)
        for oid in getattr(self, '_lm_obs_ids', []) or []:
            try:
                if rem is not None:
                    rem(oid)
                else:
                    self._lm_native_iren.RemoveObserver(oid)
            except Exception:
                pass
        self._lm_obs_ids = []
        if self._hover_sphere_actor is not None:
            try:
                self._hover_sphere_actor.SetVisibility(False)
                self.plotter.render()
            except Exception:
                pass

    def _lm_pick_volume_world(self):
        """Pick the bone surface under the cursor → world point, or None.

        Tries vtkVolumePicker (opacity isosurface) first, then a plain
        vtkCellPicker as a fallback.
        """
        native = getattr(self, '_lm_native_iren', None)
        if native is None:
            return None
        try:
            x, y = native.GetEventPosition()
        except Exception:
            return None
        renderer = self.plotter.renderer
        for picker in (getattr(self, '_lm_volume_picker', None),
                       getattr(self, '_lm_cell_picker', None)):
            if picker is None:
                continue
            try:
                if not picker.Pick(x, y, 0, renderer):
                    continue
                pos = np.asarray(picker.GetPickPosition(), dtype=float)
                if pos.size == 3 and np.all(np.isfinite(pos)):
                    return pos
            except Exception:
                continue
        return None

    def _lm_on_press(self, obj, evt):
        if not self.landmark_picking_enabled:
            return
        try:
            self._lm_press_pos = self._lm_native_iren.GetEventPosition()
        except Exception:
            self._lm_press_pos = None

    def _lm_on_release(self, obj, evt):
        if not self.landmark_picking_enabled:
            return
        press = getattr(self, '_lm_press_pos', None)
        self._lm_press_pos = None
        if press is None:
            return
        try:
            rel = self._lm_native_iren.GetEventPosition()
        except Exception:
            return
        # Click vs rotate-drag: only place when the mouse barely moved.
        if abs(rel[0] - press[0]) > 5 or abs(rel[1] - press[1]) > 5:
            return
        pt = self._lm_pick_volume_world()
        if pt is not None:
            self.on_landmark_picked(pt)
        else:
            print("[Landmark] click did not hit the bone volume")

    def _lm_on_move(self, obj, evt):
        if not self.landmark_picking_enabled or self._hover_sphere_actor is None:
            return
        # Throttle the whole handler (pick + render) to ~60Hz.
        now = time.time()
        if now - getattr(self, '_hover_last_render', 0.0) < \
                getattr(self, '_hover_min_interval', 1.0 / 60.0):
            return
        self._hover_last_render = now
        pt = self._lm_pick_volume_world()
        try:
            if pt is not None:
                self._hover_sphere_actor.SetPosition(
                    float(pt[0]), float(pt[1]), float(pt[2])
                )
                self._hover_sphere_actor.SetVisibility(True)
            else:
                self._hover_sphere_actor.SetVisibility(False)
            self.plotter.render()
        except Exception:
            pass


    def on_landmark_picked(self, picked_point):
        """Add the picked surface point as a landmark (in grid coords)."""
        if picked_point is None:
            return
        try:
            grid_pt = np.asarray(picked_point, dtype=float).ravel()
            if grid_pt.size != 3 or not np.all(np.isfinite(grid_pt)):
                return

            lps_pt = self._grid_to_lps(grid_pt)
            ras_pt = self._lps_to_ras(lps_pt) if lps_pt is not None else None

            self.landmark_counter += 1
            entry = {
                'name': f"L{self.landmark_counter}",
                'grid': grid_pt,
                'lps': lps_pt,
                'ras': ras_pt,
                'memo': '',
            }
            self.landmark_data.append(entry)

            # Visualize as a small bright-red sphere. High ambient makes it
            # glow against the ivory bone so it stays easy to spot from any
            # angle / lighting; reset_camera=False keeps the user's view.
            radius = self._estimate_landmark_radius()
            sphere = pv.Sphere(radius=radius, center=grid_pt)
            actor = self.plotter.add_mesh(
                sphere,
                color=(1.0, 0.05, 0.05),
                ambient=0.7,
                diffuse=0.5,
                specular=0.2,
                smooth_shading=True,
                pickable=False,  # so future clicks pass through to the bone
                reset_camera=False,
                name=f"landmark_{len(self.landmark_data)-1}",
            )
            self.landmark_actors.append(actor)

            self._refresh_landmark_table()
            self.plotter.update()
        except Exception as e:
            print(f"[Landmark] Failed to add point: {e}")


    def _estimate_landmark_radius(self):
        """Pick a sphere radius proportional to the currently rendered scene.

        In fusion mode the base_mesh may only cover a small ROI (e.g. the
        pelvis) while the user can also hover over distant series (knee,
        ankle). Use the aggregate scene bounds so the sphere stays visible
        everywhere.
        """
        try:
            b = self._full_render_bounds()
            if b is None and self.base_mesh is not None and self.base_mesh.n_points > 0:
                b = self.base_mesh.bounds
            if b is not None:
                diag = float(np.sqrt(
                    (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
                ))
                return max(0.5, diag * self.landmark_radius_factor)
        except Exception:
            pass
        return 1.5  # mm, sensible default

    # ----- Coordinate conversions -----

    def _grid_to_lps(self, grid_pt):
        """Convert a PyVista grid coord (mm, origin (0,0,0)) to DICOM LPS (mm).

        Math: LPS = IPP_first + X * row_dir + Y * col_dir + Z * normal_dir
        Returns None if DICOM geometry is missing.
        """
        meta = self.current_meta_info
        if not meta:
            return None
        try:
            ipp = np.array(meta.get('ipp_first'), dtype=float)
            row_dir = np.array(meta.get('row_dir'), dtype=float)
            col_dir = np.array(meta.get('col_dir'), dtype=float)
            normal_dir = np.array(meta.get('normal_dir'), dtype=float)
            if ipp.size != 3 or row_dir.size != 3 or col_dir.size != 3 or normal_dir.size != 3:
                return None
        except Exception:
            return None
        x, y, z = float(grid_pt[0]), float(grid_pt[1]), float(grid_pt[2])
        lps = ipp + x * row_dir + y * col_dir + z * normal_dir
        return lps


    def _lps_to_ras(self, lps_pt):
        """LPS ↔ RAS is a simple flip of the L and P axes."""
        if lps_pt is None:
            return None
        return np.array([-lps_pt[0], -lps_pt[1], lps_pt[2]])


    def _has_dicom_geometry(self):
        meta = self.current_meta_info or {}
        return all(k in meta for k in ('ipp_first', 'row_dir', 'col_dir', 'normal_dir'))


    def _coord_for_display(self, entry):
        """Get the (X, Y, Z) tuple to show in the table for this entry."""
        sys_ = self.landmark_coord_system
        pt = entry.get(sys_)
        if pt is None:
            return None
        return tuple(float(v) for v in pt)

    # ----- Table rendering / editing -----

    def _coord_labels(self):
        """Return (axis labels, header units) for the current coord system."""
        if self.landmark_coord_system == 'lps':
            return ("L (mm)", "P (mm)", "S (mm)")
        if self.landmark_coord_system == 'ras':
            return ("R (mm)", "A (mm)", "S (mm)")
        return ("X (mm)", "Y (mm)", "Z (mm)")


    def _update_landmark_table_header(self):
        ax = self._coord_labels()
        self.landmark_table.setHorizontalHeaderLabels(
            ["#", "Name", ax[0], ax[1], ax[2], ""]
        )


    def _refresh_landmark_table(self):
        """Rebuild the table from self.landmark_data."""
        self._suppress_table_signal = True
        try:
            self._update_landmark_table_header()
            self.landmark_table.setRowCount(len(self.landmark_data))
            for i, entry in enumerate(self.landmark_data):
                # # column
                idx_item = QTableWidgetItem(str(i + 1))
                idx_item.setTextAlignment(Qt.AlignCenter)
                idx_item.setFlags(idx_item.flags() & ~Qt.ItemIsEditable)
                self.landmark_table.setItem(i, 0, idx_item)

                # Name column (editable)
                name_item = QTableWidgetItem(str(entry.get('name', '')))
                name_item.setTextAlignment(Qt.AlignCenter)
                self.landmark_table.setItem(i, 1, name_item)

                # X / Y / Z columns (read-only) for the currently selected system
                pt = self._coord_for_display(entry)
                for c in range(3):
                    if pt is None:
                        item = QTableWidgetItem("N/A")
                        item.setToolTip(
                            "DICOM geometry (IPP/IOP) unavailable for this series.\n"
                            "Re-cache the patient with the updated loader to enable LPS/RAS."
                        )
                    else:
                        item = QTableWidgetItem(f"{pt[c]:.2f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.landmark_table.setItem(i, 2 + c, item)

                # Delete button
                del_btn = QPushButton("✕")
                del_btn.setToolTip("Delete this landmark")
                del_btn.clicked.connect(lambda _checked=False, row=i: self.remove_landmark(row))
                self.landmark_table.setCellWidget(i, 5, del_btn)
        finally:
            self._suppress_table_signal = False

        # Keep the 2D slice viewer's landmark markers in sync. This is
        # the single point that catches every landmark add / delete /
        # clear / rename path because they all funnel through here.
        if hasattr(self, '_refresh_2d_landmarks'):
            try:
                self._refresh_2d_landmarks()
            except Exception as e:
                print(f"[landmarks] 2D marker refresh failed: {e}")


    def _on_landmark_table_item_changed(self, item):
        """Handle inline edits in the Name column (column index 1)."""
        if self._suppress_table_signal:
            return
        if item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self.landmark_data):
            new_name = item.text().strip()
            if not new_name:
                # Disallow empty names — revert to old value
                self._suppress_table_signal = True
                try:
                    item.setText(str(self.landmark_data[row].get('name', '')))
                finally:
                    self._suppress_table_signal = False
                return
            self.landmark_data[row]['name'] = new_name


    def on_landmark_coord_changed(self, _index):
        data = self.landmark_coord_combo.currentData()
        if data:
            self.landmark_coord_system = data
            self._refresh_landmark_table()


    def remove_landmark(self, row):
        if row < 0 or row >= len(self.landmark_data):
            return
        # Clear any active measurement — the indices it refers to are about to shift
        self._clear_measurement_visualization()
        actor = self.landmark_actors.pop(row)
        try:
            self.plotter.remove_actor(actor)
        except Exception:
            pass
        self.landmark_data.pop(row)
        self._refresh_landmark_table()
        self.plotter.update()


    def on_clear_landmarks_clicked(self):
        if not self.landmark_data:
            return
        confirm = QMessageBox.question(
            self,
            "Clear All Landmarks",
            f"Remove all {len(self.landmark_data)} landmark point(s)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        self._clear_measurement_visualization()
        for actor in self.landmark_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.landmark_actors.clear()
        self.landmark_data.clear()
        self.landmark_counter = 0
        self._refresh_landmark_table()
        if hasattr(self, 'measurement_label'):
            self.measurement_label.setText(
                "<i>Select 2 rows for distance, 3 rows for angle (Ctrl+Click).</i>"
            )
        self.plotter.update()

    # ----- Export -----

    def on_landmark_export_clicked(self):
        if not self.landmark_data:
            QMessageBox.information(self, "No Landmarks", "There are no landmarks to export.")
            return

        # Default filename: PatientName_StudyDate_landmarks
        meta = self.current_meta_info or {}
        name = "".join(c for c in str(meta.get('patient_name', 'patient'))
                       if c.isalnum() or c in (' ', '-', '_')).strip() or 'patient'
        date_str = str(meta.get('study_date', ''))
        default_name = f"{name}_{date_str}_landmarks".rstrip('_')

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Landmarks",
            default_name,
            "CSV file (*.csv);;JSON file (*.json)",
        )
        if not path:
            return

        # Decide format by extension (fall back to filter)
        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = '.json' if 'JSON' in selected_filter else '.csv'
            path += ext

        try:
            if ext == '.csv':
                self._export_landmarks_csv(path)
            elif ext == '.json':
                self._export_landmarks_json(path)
            else:
                QMessageBox.warning(self, "Unknown Format",
                                    f"Unsupported extension: {ext}\nUse .csv or .json.")
                return
            QMessageBox.information(self, "Export Complete",
                                    f"Saved {len(self.landmark_data)} landmark(s) to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write file:\n{e}")


    def _export_landmarks_csv(self, path):
        import csv
        has_dicom = self._has_dicom_geometry()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            header = ["index", "name", "grid_X", "grid_Y", "grid_Z"]
            if has_dicom:
                header += ["LPS_L", "LPS_P", "LPS_S", "RAS_R", "RAS_A", "RAS_S"]
            w.writerow(header)
            for i, e in enumerate(self.landmark_data, start=1):
                row = [i, e['name'],
                       f"{e['grid'][0]:.4f}", f"{e['grid'][1]:.4f}", f"{e['grid'][2]:.4f}"]
                if has_dicom:
                    lps = e.get('lps')
                    ras = e.get('ras')
                    if lps is not None and ras is not None:
                        row += [f"{lps[0]:.4f}", f"{lps[1]:.4f}", f"{lps[2]:.4f}",
                                f"{ras[0]:.4f}", f"{ras[1]:.4f}", f"{ras[2]:.4f}"]
                    else:
                        row += ["", "", "", "", "", ""]
                w.writerow(row)

    # ===== Measurements (distance / angle) =====
    @staticmethod

    def _distance(a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    @staticmethod

    def _angle_at_vertex(p_left, p_vertex, p_right):
        """Return the angle at p_vertex (in degrees) of the polyline p_left – p_vertex – p_right."""
        v1 = np.asarray(p_left) - np.asarray(p_vertex)
        v2 = np.asarray(p_right) - np.asarray(p_vertex)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos_t = float(np.dot(v1, v2) / (n1 * n2))
        cos_t = max(-1.0, min(1.0, cos_t))
        return float(np.degrees(np.arccos(cos_t)))


    def _clear_measurement_visualization(self):
        for actor in self.measurement_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.measurement_actors.clear()


    def _update_landmark_selection_color(self, selected_rows):
        """Recolor landmark spheres: selected → yellow, others → red."""
        sel = set(selected_rows)
        for i, actor in enumerate(getattr(self, 'landmark_actors', [])):
            if actor is None:
                continue
            try:
                col = (1.0, 1.0, 0.0) if i in sel else (1.0, 0.05, 0.05)
                actor.GetProperty().SetColor(*col)
            except Exception:
                pass

    def _on_landmark_selection_changed(self):
        if self._suppress_table_signal:
            return
        sel_model = self.landmark_table.selectionModel()
        if sel_model is None:
            return
        rows = sorted({idx.row() for idx in sel_model.selectedRows()})
        # Filter out invalid rows defensively
        rows = [r for r in rows if 0 <= r < len(self.landmark_data)]

        # Highlight the selected landmark spheres (yellow) so you can see which
        # point is selected; unselected ones go back to red.
        self._update_landmark_selection_color(rows)
        if hasattr(self, "_update_info_panel"):
            self._update_info_panel()

        self._clear_measurement_visualization()

        if len(rows) < 2:
            self.measurement_label.setText(
                "<i>Select 2 rows for distance, 3 rows for angle (Ctrl+Click).</i>"
            )
            self.plotter.update()
            return

        # Distance and angle are invariant under rigid transforms, so we always
        # compute on grid coordinates — they exist for every landmark.
        pts = [np.asarray(self.landmark_data[r]['grid'], dtype=float) for r in rows]
        names = [str(self.landmark_data[r]['name']) for r in rows]

        if len(rows) == 2:
            d = self._distance(pts[0], pts[1])
            self.measurement_label.setText(
                f"<b>Distance</b> &nbsp; {names[0]} ↔ {names[1]}: "
                f"<span style='color:#0a7'><b>{d:.2f} mm</b></span>"
            )
            self._draw_segment(pts[0], pts[1], color='lime', label=f"{d:.1f} mm")

        elif len(rows) == 3:
            # Use selection order to define vertex = middle (by row index).
            p_left, p_vertex, p_right = pts[0], pts[1], pts[2]
            n_left, n_vertex, n_right = names[0], names[1], names[2]
            ang = self._angle_at_vertex(p_left, p_vertex, p_right)
            d1 = self._distance(p_left, p_vertex)
            d2 = self._distance(p_vertex, p_right)
            self.measurement_label.setText(
                f"<b>Angle</b> at <i>{n_vertex}</i> "
                f"({n_left}—{n_vertex}—{n_right}): "
                f"<span style='color:#c50'><b>{ang:.2f}°</b></span><br>"
                f"&nbsp;&nbsp;<small>{n_left}—{n_vertex}: {d1:.2f} mm "
                f"&nbsp;|&nbsp; {n_vertex}—{n_right}: {d2:.2f} mm</small>"
            )
            self._draw_segment(p_left, p_vertex, color='lime', label=f"{d1:.1f}")
            self._draw_segment(p_vertex, p_right, color='lime', label=f"{d2:.1f}")
            self._draw_angle_marker(p_left, p_vertex, p_right, ang)

        else:
            self.measurement_label.setText(
                f"<i>Selected {len(rows)} rows. Pick exactly 2 (distance) "
                f"or 3 (angle).</i>"
            )

        self.plotter.update()


    def _draw_segment(self, p1, p2, color='lime', label=None):
        """Draw a single line segment with an optional midpoint label."""
        try:
            p1 = np.asarray(p1, dtype=float)
            p2 = np.asarray(p2, dtype=float)
            line_actor = self.plotter.add_lines(
                np.array([p1, p2]), color=color, width=3,
            )
            self.measurement_actors.append(line_actor)
            if label:
                mid = (p1 + p2) / 2.0
                try:
                    lbl_actor = self.plotter.add_point_labels(
                        np.array([mid]),
                        [label],
                        text_color=color,
                        point_color=color,
                        point_size=1,
                        font_size=14,
                        shape=None,
                        always_visible=True,
                        render_points_as_spheres=False,
                    )
                    self.measurement_actors.append(lbl_actor)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Measurement] segment draw failed: {e}")


    def _draw_angle_marker(self, p_left, p_vertex, p_right, angle_deg):
        """Draw a small arc at the vertex + an angle label."""
        try:
            v1 = np.asarray(p_left) - np.asarray(p_vertex)
            v2 = np.asarray(p_right) - np.asarray(p_vertex)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-6 or n2 < 1e-6:
                return
            u1 = v1 / n1
            u2 = v2 / n2

            # Arc radius: 20% of the shorter segment, capped so it does not dwarf the scene
            radius = min(n1, n2) * 0.2

            n_samples = 24
            angle_rad = float(np.radians(angle_deg))
            arc_pts = []
            if angle_rad < 1e-6:
                arc_pts = [np.asarray(p_vertex) + radius * u1] * (n_samples + 1)
            else:
                sin_total = np.sin(angle_rad)
                for i in range(n_samples + 1):
                    t = i / n_samples
                    a = np.sin((1 - t) * angle_rad) / sin_total
                    b = np.sin(t * angle_rad) / sin_total
                    direction = a * u1 + b * u2
                    arc_pts.append(np.asarray(p_vertex) + radius * direction)
            arc_pts = np.asarray(arc_pts)

            # Build line segments connecting consecutive arc points
            segments = np.empty((2 * (len(arc_pts) - 1), 3), dtype=float)
            for i in range(len(arc_pts) - 1):
                segments[2 * i] = arc_pts[i]
                segments[2 * i + 1] = arc_pts[i + 1]
            arc_actor = self.plotter.add_lines(segments, color='magenta', width=2)
            self.measurement_actors.append(arc_actor)

            # Angle label at arc midpoint
            mid_arc = arc_pts[len(arc_pts) // 2]
            try:
                lbl_actor = self.plotter.add_point_labels(
                    np.array([mid_arc]),
                    [f"{angle_deg:.1f}°"],
                    text_color='magenta',
                    point_color='magenta',
                    point_size=1,
                    font_size=14,
                    shape=None,
                    always_visible=True,
                    render_points_as_spheres=False,
                )
                self.measurement_actors.append(lbl_actor)
            except Exception:
                pass
        except Exception as e:
            print(f"[Measurement] angle marker failed: {e}")


    def _export_landmarks_json(self, path):
        import json
        meta = self.current_meta_info or {}
        has_dicom = self._has_dicom_geometry()
        payload = {
            "exported_at": datetime.now().isoformat(timespec='seconds'),
            "patient": {
                "name": str(meta.get('patient_name', 'N/A')),
                "study_date": str(meta.get('study_date', 'N/A')),
                "modality": str(meta.get('modality', 'N/A')),
                "series_description": str(meta.get('series_description', 'N/A')),
                "series_uid": str(meta.get('series_uid', 'N/A')),
                "slice_thickness_mm": meta.get('slice_thickness', None),
                "patient_position": str(meta.get('patient_position', 'N/A')),
            },
            "coordinate_systems": {
                "grid": "PyVista ImageData (mm). Origin = (0,0,0).",
                "lps": ("DICOM patient Left/Posterior/Superior (mm). "
                        "Computed via IPP + X*row_dir + Y*col_dir + Z*normal_dir."),
                "ras": "Right/Anterior/Superior (mm). LPS with X and Y negated.",
            },
            "dicom_geometry_available": has_dicom,
            "spacing_mm": list(self.current_spacing) if self.current_spacing else None,
            "landmarks": [],
        }
        for i, e in enumerate(self.landmark_data, start=1):
            item = {
                "index": i,
                "name": e['name'],
                "grid": [float(v) for v in e['grid']],
            }
            if e.get('lps') is not None:
                item['lps'] = [float(v) for v in e['lps']]
            if e.get('ras') is not None:
                item['ras'] = [float(v) for v in e['ras']]
            payload['landmarks'].append(item)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ===== Session Save / Load =====

    def _enable_hover_preview(self):
        """Activate the hover preview sphere + mouse-move observer."""
        # Lazy-create the picker
        if self._hover_picker is None:
            self._hover_picker = vtk.vtkCellPicker()
            self._hover_picker.SetTolerance(0.005)

        # Lazy-create the preview sphere centered at the origin.
        # We translate the actor via SetPosition() each mouse move so we never
        # rebuild the geometry — that keeps hover updates cheap.
        if self._hover_sphere_actor is None:
            radius = self._estimate_landmark_radius()
            sphere_mesh = pv.Sphere(radius=radius, center=(0.0, 0.0, 0.0))
            self._hover_sphere_actor = self.plotter.add_mesh(
                sphere_mesh,
                color="yellow",
                opacity=0.65,
                specular=0.3,
                smooth_shading=True,
                pickable=False,
                name="hover_preview_sphere",
            )
        # Start hidden until the mouse actually moves over the bone surface
        try:
            self._hover_sphere_actor.SetVisibility(False)
        except Exception:
            pass

        # Register MouseMoveEvent observer (idempotent)
        if self._hover_observer_id is None:
            try:
                self._hover_observer_id = self.plotter.iren.add_observer(
                    "MouseMoveEvent", self._on_hover_mouse_move
                )
            except AttributeError:
                # Fallback: register directly on the underlying VTK interactor
                try:
                    native_iren = self.plotter.iren.interactor
                    self._hover_observer_id = native_iren.AddObserver(
                        "MouseMoveEvent", self._on_hover_mouse_move
                    )
                except Exception as e:
                    print(f"[Hover] Failed to register observer: {e}")
                    self._hover_observer_id = None


    def _disable_hover_preview(self):
        """Remove the mouse-move observer and hide the preview sphere."""
        if self._hover_observer_id is not None:
            try:
                self.plotter.iren.remove_observer(self._hover_observer_id)
            except AttributeError:
                try:
                    self.plotter.iren.interactor.RemoveObserver(self._hover_observer_id)
                except Exception:
                    pass
            except Exception:
                pass
            self._hover_observer_id = None

        if self._hover_sphere_actor is not None:
            try:
                self._hover_sphere_actor.SetVisibility(False)
                self.plotter.render()
            except Exception:
                pass


    def _destroy_hover_preview(self):
        """Fully tear down the hover sphere (used on patient reload)."""
        self._disable_hover_preview()
        if self._hover_sphere_actor is not None:
            try:
                self.plotter.remove_actor(self._hover_sphere_actor)
            except Exception:
                pass
            self._hover_sphere_actor = None


    def _on_hover_mouse_move(self, obj, event_name):
        """vtk MouseMoveEvent handler — projects cursor onto bone surface."""
        if not self.landmark_picking_enabled:
            return
        if self._hover_picker is None or self._hover_sphere_actor is None:
            return

        # Build the set of valid bone-mesh actors. In fusion mode there is no
        # single current_mesh_actor — instead every series contributes its own
        # actor to self.fusion_actors. We want the hover sphere to follow the
        # cursor over any of them.
        target_actors = set()
        if self.bone_separation_enabled and self.separated_bones:
            for bone in self.separated_bones:
                if not bone.get('visible', True):
                    continue
                actor = bone.get('actor')
                if actor is not None:
                    target_actors.add(actor)
        else:
            if self.current_mesh_actor is not None:
                target_actors.add(self.current_mesh_actor)
            for a in self.fusion_actors:
                if a is not None:
                    target_actors.add(a)
        if not target_actors:
            return

        try:
            # Get screen-space cursor position from the native VTK interactor
            try:
                iren = self.plotter.iren.interactor
            except AttributeError:
                iren = obj  # observer passes the interactor itself
            x, y = iren.GetEventPosition()
            renderer = self.plotter.renderer

            hit = self._hover_picker.Pick(x, y, 0, renderer)
            show_sphere = False
            if hit and self._hover_picker.GetCellId() != -1:
                picked_actor = self._hover_picker.GetActor()
                # Only react when the ray hit a bone-mesh actor (single or fused)
                if picked_actor in target_actors:
                    pt = self._hover_picker.GetPickPosition()
                    self._hover_sphere_actor.SetPosition(pt[0], pt[1], pt[2])
                    show_sphere = True

            was_visible = bool(self._hover_sphere_actor.GetVisibility())
            if show_sphere != was_visible:
                self._hover_sphere_actor.SetVisibility(show_sphere)

            # Throttle render to keep cursor tracking smooth without flooding GPU
            now = time.time()
            if show_sphere or was_visible:
                if now - self._hover_last_render >= self._hover_min_interval:
                    self._hover_last_render = now
                    self.plotter.render()
        except Exception:
            # Swallow per-frame errors to avoid spamming the console while moving fast
            pass


