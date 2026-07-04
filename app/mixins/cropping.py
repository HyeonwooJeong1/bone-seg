import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt

class CroppingMixin:
    def _full_render_bounds(self):
        """Aggregate bounds for whatever is currently on screen.

        - Fusion mode: union of every visible fusion-mesh's bounds, expressed
          in the base series' grid frame (which is the frame the crop widget
          lives in, since all clipping happens there).
        - Single mode: the active volume grid's bounds.
        Returns None when nothing is rendered yet.
        """
        # AI-first mode: fit the box to the union of the AI bones' full surfaces
        # (they're already in the base-grid frame the crop widget lives in).
        if getattr(self, 'ai_segmentation_active', False) and getattr(self, 'separated_bones', None):
            bl = []
            for bone in self.separated_bones:
                m = bone.get('raw_mesh') or bone.get('mesh')
                if m is not None and m.n_points > 0:
                    bl.append(m.bounds)
            if bl:
                return (min(b[0] for b in bl), max(b[1] for b in bl),
                        min(b[2] for b in bl), max(b[3] for b in bl),
                        min(b[4] for b in bl), max(b[5] for b in bl))
        if self.fusion_enabled:
            bounds_list = []
            # fusion_meshes stores (series_index, mesh) tuples — unpack carefully
            for entry in self.fusion_meshes:
                try:
                    mesh = entry[1] if isinstance(entry, tuple) else entry
                    if mesh is not None and mesh.n_points > 0:
                        bounds_list.append(mesh.bounds)
                except Exception:
                    pass
            if not bounds_list:
                # Fall back to the base volume grid while meshes haven't been
                # built yet (e.g. crop toggled before first render).
                if self.volume_grid is not None:
                    return tuple(self.volume_grid.bounds)
                return None
            xs0 = [b[0] for b in bounds_list]
            xs1 = [b[1] for b in bounds_list]
            ys0 = [b[2] for b in bounds_list]
            ys1 = [b[3] for b in bounds_list]
            zs0 = [b[4] for b in bounds_list]
            zs1 = [b[5] for b in bounds_list]
            return (min(xs0), max(xs1), min(ys0), max(ys1), min(zs0), max(zs1))
        if self.volume_grid is not None:
            return tuple(self.volume_grid.bounds)
        return None


    def _apply_crop_to_ai_bones(self):
        """Clip every AI-segmented bone against the crop box (view-only).

        Each bone keeps its pristine full surface in `raw_mesh`; the visible
        actor and the `mesh` used for the selection highlight are derived from
        it live, mirroring how the volume/fusion crop re-clips on every change.
        Returns True if AI bones were handled (so the caller can skip the
        volume/fusion renderer)."""
        bones = getattr(self, 'separated_bones', None)
        if not bones:
            return False
        crop_active = (
            getattr(self, 'cropping_bounds', None) is not None
            and getattr(self, 'crop_checkbox', None) is not None
            and self.crop_checkbox.isChecked()
        )
        for bone in bones:
            full = bone.get('raw_mesh') or bone.get('mesh')
            if full is None:
                continue
            disp, empty = full, False
            if crop_active:
                try:
                    clipped = full.clip_box(self.cropping_bounds, invert=False)
                    if clipped is not None and clipped.n_points > 0:
                        disp = clipped
                    else:
                        empty = True  # bone lies entirely outside the box
                except Exception:
                    disp = full
            # `mesh` tracks what's on screen (highlight wireframe uses it).
            bone['mesh'] = full if empty else disp
            actor = bone.get('actor')
            if actor is None:
                continue
            try:
                if not empty:
                    actor.mapper.dataset = disp
                actor.SetVisibility(bool(bone.get('visible', True)) and not empty)
            except Exception:
                pass
        # Re-draw the yellow selection wireframe against the clipped meshes.
        if (hasattr(self, '_on_bone_list_selection_changed')
                and hasattr(self, 'bone_list_widget')
                and self.bone_list_widget.selectedItems()):
            try:
                self._on_bone_list_selection_changed()
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass
        return True

    def _rerender_for_crop(self):
        """Re-run whichever renderer owns the scene so crop changes show up."""
        # AI-first mode: the visible geometry is the AI bones, not the volume.
        if getattr(self, 'ai_segmentation_active', False) and self._apply_crop_to_ai_bones():
            return
        if self.fusion_enabled:
            self._update_fused_meshes()
        else:
            self.update_rendered_mesh()


    def on_crop_toggled(self, state):
        if state == Qt.Checked:
            self.hide_handles_checkbox.setEnabled(True)
            full_bounds = self._full_render_bounds()
            if full_bounds is not None:
                # Restore the previous box position (if any) so toggling off/on
                # preserves the user's crop region. factor=1.0 prevents
                # PyVista from inflating the box by 25%.
                restore_bounds = (
                    self.last_box_bounds
                    if self.last_box_bounds is not None
                    else full_bounds
                )
                self.plotter.add_box_widget(
                    self.on_box_cropped,
                    bounds=restore_bounds,
                    factor=1.0,
                )
        else:
            self.hide_handles_checkbox.setEnabled(False)
            self.hide_handles_checkbox.setChecked(False)
            self.plotter.clear_box_widgets()
            # Intentionally KEEP self.cropping_bounds and self.last_box_bounds
            # so that re-enabling the crop restores the same region.
            # The renderer skips applying the crop while the checkbox is off.
        self._rerender_for_crop()


    def on_crop_reset_clicked(self):
        """Explicitly reset the crop box to full bounds and clear cropping."""
        self.cropping_bounds = None
        self.last_box_bounds = None
        # If the crop is currently active, rebuild the widget at full bounds
        if self.crop_checkbox.isChecked():
            full_bounds = self._full_render_bounds()
            if full_bounds is not None:
                self.plotter.clear_box_widgets()
                self.plotter.add_box_widget(
                    self.on_box_cropped,
                    bounds=full_bounds,
                    factor=1.0,
                )
        self._rerender_for_crop()


    def on_hide_handles_toggled(self, state):
        if state == Qt.Checked:
            # Hide the box widget visually, but keep the cropping bounds
            self.plotter.clear_box_widgets()
        else:
            # Re-show the box widget at the SAME position it was before hiding
            if self.crop_checkbox.isChecked():
                full_bounds = self._full_render_bounds()
                if full_bounds is not None:
                    restore_bounds = (
                        self.last_box_bounds
                        if self.last_box_bounds is not None
                        else full_bounds
                    )
                    # factor=1.0 prevents PyVista's default 25% padding
                    self.plotter.add_box_widget(
                        self.on_box_cropped,
                        bounds=restore_bounds,
                        factor=1.0,
                    )


    def on_box_cropped(self, box_polydata):
        # box_polydata is a PolyData object from the widget callback
        self.cropping_bounds = box_polydata
        self.last_box_bounds = box_polydata.bounds  # Store 6-value tuple for restoring later
        self._rerender_for_crop()
    

