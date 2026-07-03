import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt

class ParticleRemovalMixin:
    def on_smooth_changed(self, index):
        if self.volume_grid is not None:
            self.update_base_mesh()

    # ===== Particle Removal Handlers =====

    def on_voxel_cleanup_toggled(self, state):
        self.particle_removal_enabled = (state == Qt.Checked)
        self.opening_iter_spinbox.setEnabled(self.particle_removal_enabled)
        self.opening_conn_combo.setEnabled(self.particle_removal_enabled)
        if self.volume_grid is not None:
            self.update_base_mesh()

    def on_opening_iter_changed(self, value):
        self.opening_iterations = int(value)
        if self.particle_removal_enabled and self.volume_grid is not None:
            self.update_base_mesh()

    def on_opening_conn_changed(self, index):
        data = self.opening_conn_combo.itemData(index)
        if data is None:
            return
        self.opening_connectivity = int(data)
        if self.particle_removal_enabled and self.opening_iterations > 0 \
                and self.volume_grid is not None:
            self.update_base_mesh()

    def on_mesh_cleanup_toggled(self, state):
        self.mesh_cleanup_enabled = (state == Qt.Checked)
        self.keep_largest_checkbox.setEnabled(self.mesh_cleanup_enabled)
        self.fragment_faces_spinbox.setEnabled(
            self.mesh_cleanup_enabled and not self.keep_largest_only
        )
        if self.volume_grid is not None:
            self.update_base_mesh()

    def on_keep_largest_toggled(self, state):
        self.keep_largest_only = (state == Qt.Checked)
        self.fragment_faces_spinbox.setEnabled(
            self.mesh_cleanup_enabled and not self.keep_largest_only
        )
        if self.mesh_cleanup_enabled and self.volume_grid is not None:
            self.update_base_mesh()

    def on_fragment_faces_changed(self, value):
        self.min_fragment_faces = int(value)
        if self.mesh_cleanup_enabled and not self.keep_largest_only and self.volume_grid is not None:
            self.update_base_mesh()

    def on_closing_toggled(self, state):
        self.closing_enabled = (state == Qt.Checked)
        self.closing_iter_spinbox.setEnabled(self.closing_enabled)
        if self.volume_grid is not None:
            self.update_base_mesh()

    def on_closing_iter_changed(self, value):
        self.closing_iterations = int(value)
        if self.closing_enabled and self.volume_grid is not None:
            self.update_base_mesh()

    def on_fill_holes_toggled(self, state):
        self.mesh_fill_holes_enabled = (state == Qt.Checked)
        self.fill_holes_spinbox.setEnabled(self.mesh_fill_holes_enabled)
        if self.volume_grid is not None:
            self.update_base_mesh()

    def on_fill_holes_size_changed(self, value):
        self.mesh_fill_holes_size = int(value)
        if self.mesh_fill_holes_enabled and self.volume_grid is not None:
            self.update_base_mesh()

    def on_picking_toggled(self, checked):
        self.picking_enabled = checked
        if checked:
            # Mutual exclusion: turn off other picking modes
            if hasattr(self, 'landmark_pick_btn') and self.landmark_pick_btn.isChecked():
                self.landmark_pick_btn.setChecked(False)
            if hasattr(self, 'restore_pick_btn') and self.restore_pick_btn.isChecked():
                self.restore_pick_btn.setChecked(False)
            self.pick_btn.setText("Click-to-Remove: ON")
            self.plotter.enable_point_picking(
                callback=self.on_point_picked,
                show_message="Click on a particle to remove it (toggle off to rotate freely)",
                use_picker='cell',
                show_point=False,
                pickable_window=False,
                left_clicking=True,
            )
        else:
            self.pick_btn.setText("Click-to-Remove: OFF")
            try:
                self.plotter.disable_picking()
            except Exception:
                pass


    def on_point_picked(self, picked_point):
        """Remove the connected component nearest to picked_point from base_mesh."""
        if self.base_mesh is None or picked_point is None:
            return
        if self.base_mesh.n_points == 0:
            return

        try:
            # Snapshot for undo
            self._manual_undo_stack.append(self.base_mesh.copy(deep=True))
            if len(self._manual_undo_stack) > self._max_undo:
                self._manual_undo_stack.pop(0)
            self.undo_btn.setEnabled(True)

            # Label all connected regions
            labeled = self.base_mesh.connectivity(extraction_mode='all')
            if 'RegionId' not in labeled.point_data:
                self._manual_undo_stack.pop()
                return

            # Find which region contains the point closest to the picked location
            closest_pt_id = labeled.find_closest_point(np.asarray(picked_point))
            target_region = int(labeled.point_data['RegionId'][closest_pt_id])

            # Remove cells belonging to the picked region
            region_ids_cell = labeled.cell_data['RegionId']
            cells_to_remove = np.where(region_ids_cell == target_region)[0]
            if len(cells_to_remove) == 0:
                self._manual_undo_stack.pop()
                if not self._manual_undo_stack:
                    self.undo_btn.setEnabled(False)
                return

            cleaned = labeled.remove_cells(cells_to_remove)
            if not isinstance(cleaned, pv.PolyData):
                cleaned = cleaned.extract_surface()
            for arr_name in ('RegionId',):
                if arr_name in cleaned.point_data:
                    del cleaned.point_data[arr_name]
                if arr_name in cleaned.cell_data:
                    del cleaned.cell_data[arr_name]

            self.base_mesh = cleaned
            self.update_rendered_mesh()
        except Exception as e:
            print(f"[Click-to-Remove] Failed: {e}")
            if self._manual_undo_stack:
                self._manual_undo_stack.pop()
            if not self._manual_undo_stack:
                self.undo_btn.setEnabled(False)


    def on_undo_clicked(self):
        if not self._manual_undo_stack:
            self.undo_btn.setEnabled(False)
            return
        self.base_mesh = self._manual_undo_stack.pop()
        if not self._manual_undo_stack:
            self.undo_btn.setEnabled(False)
        self.update_rendered_mesh()


    def _clear_undo_stack(self):
        self._manual_undo_stack.clear()
        if hasattr(self, 'undo_btn'):
            self.undo_btn.setEnabled(False)

    # ===== Landmark Picking Handlers =====


