"""2D Axial Slice Viewer — MicroDicom-style companion to the 3D view.

Renders the active series' HU volumes as a **unified z-ordered stack**:
all slices from every included series are concatenated in ascending
patient-z order so the user scrolls through "the whole leg" continuously,
exactly like scrolling a single CT in MicroDicom. A single slider
controls the unified index; the series name + local slice number are
shown in the title so the user always knows which series they're on.

Optional 3D companion views (both default OFF — toggle in the 2D window):
  - **Slice indicator**: a red translucent quad at the current slice's
    physical location in the 3D plotter. Cached polydata is mutated in
    place to avoid actor churn / camera resets during drag-scroll.
  - **Hide above current slice**: a VTK clipping plane on every bone
    mesh actor's mapper that hides everything above the current slice
    in patient z. Pure GPU clipping — no mesh recomputation, no smoothing
    redo, so it's fast even on dense fused meshes.

Implementation: matplotlib FigureCanvas (NOT a second QtInteractor) so we
don't fight for the WGL OpenGL context with the main 3D plotter on
Windows. Same pattern as the existing Scout Viewer.
"""

import time

import numpy as np
import pyvista as pv
import vtk
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# Standard radiologic window/level presets (Width, Level) in HU.
WL_PRESETS = {
    'Bone':        (1500,  300),
    'Soft Tissue': ( 400,   40),
    'Lung':        (1500, -600),
    'Brain':       (  80,   40),
}


class SliceViewerMixin:
    # ──────────────────────────────────────────────────────────────────
    # Window construction (called once from MainWindow.__init__)
    # ──────────────────────────────────────────────────────────────────
    def _build_slice_window(self):
        """Construct the 2D slice viewer window. Hidden by default."""
        self.slice_window = QMainWindow(self)
        self.slice_window.setWindowTitle("2D Slice Viewer (Axial)")
        self.slice_window.resize(720, 860)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Single matplotlib subplot — the unified stack shows one slice
        # at a time, transitioning between series as the slider moves.
        self.slice_fig = Figure(figsize=(6, 6), facecolor='black')
        self.slice_canvas = FigureCanvas(self.slice_fig)
        # Give the canvas keyboard focus on click/wheel so matplotlib's
        # own modifier tracking has a chance to populate event.key. We
        # also check Qt's global modifier state in _slice_mpl_scroll as a
        # robust fallback for cases where the canvas hasn't taken focus.
        self.slice_canvas.setFocusPolicy(Qt.WheelFocus)
        self.slice_ax = self.slice_fig.add_subplot(111)
        self._style_axes(self.slice_ax)
        self.slice_fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
        self.slice_im = None
        layout.addWidget(self.slice_canvas, stretch=1)

        # Status line: unified position + per-series breakdown + W/L.
        self.slice_status_label = QLabel("Load a patient to view slices.")
        self.slice_status_label.setStyleSheet(
            "QLabel { padding: 4px; background: #222; color: #ddd; "
            "font-family: monospace; }"
        )
        layout.addWidget(self.slice_status_label)

        # Unified stack slider (0 .. total_slices - 1).
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self._on_slice_slider_changed)
        slider_row.addWidget(self.slice_slider, stretch=1)
        layout.addLayout(slider_row)

        # Pick-landmark-on-2D toggle. When ON, left-click places a
        # landmark instead of scrolling slices. The landmark is added to
        # the central landmark table + a sphere appears in the 3D view.
        pick_row = QHBoxLayout()
        self.slice_pick_landmark_btn = QPushButton("Pick Landmark on 2D: OFF")
        self.slice_pick_landmark_btn.setCheckable(True)
        self.slice_pick_landmark_btn.setToolTip(
            "Click on the 2D slice to place a landmark.\n"
            "The same point is also added to the 3D view + landmark table."
        )
        self.slice_pick_landmark_btn.toggled.connect(self._on_2d_pick_toggled)
        pick_row.addWidget(self.slice_pick_landmark_btn)

        self.slice_reset_zoom_btn = QPushButton("Reset Zoom")
        self.slice_reset_zoom_btn.setToolTip("Reset 2D zoom/pan to fit the slice.")
        self.slice_reset_zoom_btn.clicked.connect(self._on_reset_zoom_clicked)
        pick_row.addWidget(self.slice_reset_zoom_btn)
        layout.addLayout(pick_row)

        # 3D companion toggles (both default OFF for performance).
        self.slice_3d_indicator_checkbox = QCheckBox("Show current slice in 3D view")
        self.slice_3d_indicator_checkbox.setChecked(self.slice_show_3d_indicator)
        self.slice_3d_indicator_checkbox.setToolTip(
            "Overlay a semi-transparent red rectangle in the 3D plotter at\n"
            "the physical location of the currently displayed 2D slice."
        )
        self.slice_3d_indicator_checkbox.stateChanged.connect(self._on_3d_indicator_toggled)
        layout.addWidget(self.slice_3d_indicator_checkbox)

        self.slice_hide_above_checkbox = QCheckBox(
            "Hide bone above current slice (3D clip)"
        )
        self.slice_hide_above_checkbox.setChecked(self.slice_hide_above_enabled)
        self.slice_hide_above_checkbox.setToolTip(
            "Clip every bone mesh in the 3D view at the current slice's\n"
            "patient-z position. GPU clipping plane — does NOT recompute\n"
            "the mesh, so smoothing / particle removal / bone separation\n"
            "are all preserved. Fast enough for live drag scrolling."
        )
        self.slice_hide_above_checkbox.stateChanged.connect(self._on_hide_above_toggled)
        layout.addWidget(self.slice_hide_above_checkbox)

        # W/L preset combo (shared)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.wl_preset_combo = QComboBox()
        self.wl_preset_combo.addItems(list(WL_PRESETS.keys()) + ['Custom'])
        self.wl_preset_combo.setCurrentText('Bone')
        self.wl_preset_combo.currentIndexChanged.connect(self._on_wl_preset_changed)
        preset_row.addWidget(self.wl_preset_combo, stretch=1)
        layout.addLayout(preset_row)

        # Window Width slider
        ww_row = QHBoxLayout()
        ww_row.addWidget(QLabel("Width:"))
        self.ww_slider = QSlider(Qt.Horizontal)
        self.ww_slider.setRange(1, 4000)
        self.ww_slider.setValue(self.window_width)
        self.ww_slider.valueChanged.connect(self._on_ww_changed)
        ww_row.addWidget(self.ww_slider, stretch=1)
        layout.addLayout(ww_row)

        # Window Level slider
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Level:"))
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(-1000, 3000)
        self.wl_slider.setValue(self.window_level)
        self.wl_slider.valueChanged.connect(self._on_wl_changed)
        wl_row.addWidget(self.wl_slider, stretch=1)
        layout.addLayout(wl_row)

        help_label = QLabel(
            "<i>Left-drag: scroll · Right-drag: W/L · Wheel: ±1 slice · "
            "Ctrl+Wheel: zoom</i>"
        )
        help_label.setStyleSheet("QLabel { color: #888; padding: 2px; }")
        layout.addWidget(help_label)

        self.slice_window.setCentralWidget(central)

        self.slice_canvas.mpl_connect('button_press_event',   self._slice_mpl_press)
        self.slice_canvas.mpl_connect('button_release_event', self._slice_mpl_release)
        self.slice_canvas.mpl_connect('motion_notify_event',  self._slice_mpl_motion)
        self.slice_canvas.mpl_connect('scroll_event',         self._slice_mpl_scroll)

        # The 3D landmark table fires this signal whenever the user picks
        # different rows. We piggy-back on it to update the 2D measurement
        # overlay alongside the existing 3D one. Multiple slots on the same
        # signal coexist; we don't replace the existing handler.
        if hasattr(self, 'landmark_table') and self.landmark_table is not None:
            self.landmark_table.itemSelectionChanged.connect(
                self._refresh_2d_measurement
            )

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor('black')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ──────────────────────────────────────────────────────────────────
    # Public action
    # ──────────────────────────────────────────────────────────────────
    def open_slice_viewer(self):
        if self.current_image_hu is None:
            QMessageBox.information(
                self, "No Data", "Load a patient and pick a series first."
            )
            return
        self.slice_window.show()
        self.slice_window.raise_()
        self.slice_window.activateWindow()
        if not self.slice_stack:
            self._refresh_slice_viewer()
        else:
            self.slice_canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────
    # Mouse handlers
    # ──────────────────────────────────────────────────────────────────
    def _slice_mpl_press(self, event):
        if not self.slice_stack:
            return
        if event.x is None or event.y is None:
            return
        if event.button == 1:
            if self.slice_landmark_pick_enabled:
                # In pick mode left-click places a landmark instead of
                # starting a slice drag.
                self._handle_2d_landmark_click(event)
                return
            self._slice_drag_mode = 'slice'
            self._slice_drag_start = (event.x, event.y)
            self._slice_drag_start_idx = self.slice_unified_idx
        elif event.button == 3:
            self._slice_drag_mode = 'wl'
            self._slice_drag_start = (event.x, event.y)
            self._slice_drag_start_ww = self.window_width
            self._slice_drag_start_wl = self.window_level

    def _slice_mpl_release(self, event):
        self._slice_drag_mode = None

    def _slice_mpl_motion(self, event):
        if not self._slice_drag_mode:
            return
        if event.x is None or event.y is None:
            return

        dx = event.x - self._slice_drag_start[0]
        dy = event.y - self._slice_drag_start[1]

        if self._slice_drag_mode == 'slice':
            total = len(self.slice_stack)
            if total <= 0:
                return
            new_idx = self._slice_drag_start_idx + int(dy / 5.0)
            new_idx = int(np.clip(new_idx, 0, total - 1))
            if new_idx != self.slice_unified_idx:
                self._set_unified_slice(new_idx)
        elif self._slice_drag_mode == 'wl':
            new_ww = int(np.clip(self._slice_drag_start_ww + dx * 4, 1, 4000))
            new_wl = int(np.clip(self._slice_drag_start_wl + dy * 4, -1000, 3000))
            self.window_width = new_ww
            self.window_level = new_wl
            self.ww_slider.blockSignals(True)
            self.ww_slider.setValue(new_ww)
            self.ww_slider.blockSignals(False)
            self.wl_slider.blockSignals(True)
            self.wl_slider.setValue(new_wl)
            self.wl_slider.blockSignals(False)
            self._set_preset_combo_to_custom()
            self._update_clim_only()

    def _slice_mpl_scroll(self, event):
        if not self.slice_stack:
            return
        # Ctrl + wheel = zoom around the cursor (canvas-only, no 3D effect).
        #
        # We can't rely on matplotlib's event.key here: on the Qt5Agg
        # backend the modifier state is tracked via the canvas's own
        # keyPressEvent, which only fires while the canvas has keyboard
        # focus. Scrolling the wheel without first clicking on the canvas
        # leaves event.key == None even when Ctrl is held. Qt tracks the
        # modifier state globally though, so check that directly.
        mods = QApplication.keyboardModifiers()
        ctrl_held = (
            bool(mods & Qt.ControlModifier)
            or (event.key and 'control' in event.key)
            or (event.key and 'ctrl' in event.key)
        )
        if ctrl_held:
            self._zoom_2d(event)
            return
        total = len(self.slice_stack)
        if event.button == 'up':
            new_idx = min(self.slice_unified_idx + 1, total - 1)
        else:
            new_idx = max(self.slice_unified_idx - 1, 0)
        if new_idx != self.slice_unified_idx:
            self._set_unified_slice(new_idx)

    def _zoom_2d(self, event):
        """Cursor-centred zoom by adjusting the axes' xlim/ylim.

        Matplotlib's y-axis is inverted on imshow (top-down) because we
        passed extent=(0, w, h, 0); cursor-anchored rescaling works the
        same way as long as we use the actual current limits.
        """
        if event.xdata is None or event.ydata is None:
            return
        ax = self.slice_ax
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # 'up' = zoom in (shrink view extent), 'down' = zoom out.
        scale = 0.8 if event.button == 'up' else 1.25
        x_c, y_c = event.xdata, event.ydata
        new_xlim = (x_c - (x_c - cur_xlim[0]) * scale,
                    x_c + (cur_xlim[1] - x_c) * scale)
        new_ylim = (y_c - (y_c - cur_ylim[0]) * scale,
                    y_c + (cur_ylim[1] - y_c) * scale)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.slice_canvas.draw_idle()

    def _on_reset_zoom_clicked(self):
        """Reset the zoom/pan to fit the current slice."""
        if self.slice_im is None or not self.slice_stack:
            return
        series_idx, _ = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        spacing = series['spacing']
        sx = float(spacing[2])
        sy = float(spacing[1])
        nz, ny, nx = series['image_hu'].shape
        self.slice_ax.set_xlim(0.0, nx * sx)
        self.slice_ax.set_ylim(ny * sy, 0.0)  # imshow origin='upper' convention
        self.slice_canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────
    # Qt control handlers
    # ──────────────────────────────────────────────────────────────────
    def _on_slice_slider_changed(self, value):
        if not self.slice_stack:
            return
        self.slice_unified_idx = int(value)
        self._render_unified_slice(force_render=True)

    def _on_ww_changed(self, value):
        self.window_width = int(value)
        self._set_preset_combo_to_custom()
        self._update_clim_only(force_render=True)

    def _on_wl_changed(self, value):
        self.window_level = int(value)
        self._set_preset_combo_to_custom()
        self._update_clim_only(force_render=True)

    def _on_wl_preset_changed(self, index):
        name = self.wl_preset_combo.currentText()
        if name not in WL_PRESETS:
            return
        ww, wl = WL_PRESETS[name]
        self.window_width = ww
        self.window_level = wl
        self.ww_slider.blockSignals(True)
        self.ww_slider.setValue(ww)
        self.ww_slider.blockSignals(False)
        self.wl_slider.blockSignals(True)
        self.wl_slider.setValue(wl)
        self.wl_slider.blockSignals(False)
        self._update_clim_only(force_render=True)

    def _on_2d_pick_toggled(self, checked):
        """Toggle 'click-to-place-landmark' mode on the 2D canvas."""
        self.slice_landmark_pick_enabled = bool(checked)
        if self.slice_landmark_pick_enabled:
            self.slice_pick_landmark_btn.setText("Pick Landmark on 2D: ON")
            self.slice_canvas.setCursor(Qt.CrossCursor)
        else:
            self.slice_pick_landmark_btn.setText("Pick Landmark on 2D: OFF")
            self.slice_canvas.setCursor(Qt.ArrowCursor)
            self._slice_drag_mode = None

    def _on_3d_indicator_toggled(self, state):
        self.slice_show_3d_indicator = (state == Qt.Checked)
        if self.slice_show_3d_indicator:
            self._update_3d_slice_indicator()
        else:
            self._remove_3d_slice_indicator()
        self._render_plotter()

    def _on_hide_above_toggled(self, state):
        self.slice_hide_above_enabled = (state == Qt.Checked)
        if self.slice_hide_above_enabled:
            self._ensure_slice_clip_plane()
            self._update_slice_clip_plane_origin()
            self._apply_clip_to_all_actors()
            self._update_clip_cap_mesh()
        else:
            self._remove_clip_from_all_actors()
            self._remove_clip_cap_mesh()
        self._render_plotter()

    def _set_preset_combo_to_custom(self):
        if self.wl_preset_combo.currentText() != 'Custom':
            self.wl_preset_combo.blockSignals(True)
            self.wl_preset_combo.setCurrentText('Custom')
            self.wl_preset_combo.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────
    # Stack assembly
    # ──────────────────────────────────────────────────────────────────
    def _active_series_indices(self):
        if not self.all_series_data:
            return []
        if self.fusion_enabled and self.fusion_include_flags:
            return [i for i, f in enumerate(self.fusion_include_flags) if f]
        idx = max(0, min(self.base_series_index, len(self.all_series_data) - 1))
        return [idx]

    def _series_lps_z_origin(self, series_idx):
        series = self.all_series_data[series_idx]
        meta = series.get('meta', {}) or {}
        z_min = meta.get('z_min')
        if z_min is not None:
            try:
                return float(z_min)
            except (TypeError, ValueError):
                pass
        ipp = meta.get('ipp_first')
        if ipp is not None and len(ipp) == 3:
            try:
                return float(ipp[2])
            except (TypeError, ValueError):
                pass
        return float(series_idx) * 10000.0

    def _build_slice_stack(self):
        active = self._active_series_indices()
        if not active:
            return []
        sorted_active = sorted(active, key=self._series_lps_z_origin)
        stack = []
        for series_idx in sorted_active:
            nz = self.all_series_data[series_idx]['image_hu'].shape[0]
            for local in range(nz):
                stack.append((series_idx, local))
        return stack

    # ──────────────────────────────────────────────────────────────────
    # Refresh — full rebuild
    # ──────────────────────────────────────────────────────────────────
    def _refresh_slice_viewer(self):
        if self.slice_fig is None:
            return

        old_stack_pos = self._current_stack_pos()
        self.slice_stack = self._build_slice_stack()
        total = len(self.slice_stack)

        if total <= 0:
            self.slice_ax.clear()
            self._style_axes(self.slice_ax)
            self.slice_im = None
            self._last_rendered_series_idx = None
            self.slice_unified_idx = 0
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, 0)
            self.slice_slider.setValue(0)
            self.slice_slider.blockSignals(False)
            self._remove_3d_slice_indicator()
            self._remove_clip_from_all_actors()
            self._update_slice_status_label()
            self.slice_canvas.draw_idle()
            return

        new_pos = 0
        if old_stack_pos is not None:
            for i, entry in enumerate(self.slice_stack):
                if entry == old_stack_pos:
                    new_pos = i
                    break
            else:
                new_pos = total // 2
        else:
            new_pos = total // 2
        self.slice_unified_idx = int(np.clip(new_pos, 0, total - 1))

        self.slice_slider.blockSignals(True)
        self.slice_slider.setRange(0, total - 1)
        self.slice_slider.setValue(self.slice_unified_idx)
        self.slice_slider.blockSignals(False)

        self._last_rendered_series_idx = None
        # Mesh rebuild may have produced fresh actors — re-attach clip if on.
        if self.slice_hide_above_enabled:
            self._ensure_slice_clip_plane()
            self._apply_clip_to_all_actors()
        self._render_unified_slice(force_render=True)

    def _current_stack_pos(self):
        if not self.slice_stack:
            return None
        if 0 <= self.slice_unified_idx < len(self.slice_stack):
            return self.slice_stack[self.slice_unified_idx]
        return None

    # ──────────────────────────────────────────────────────────────────
    # Refresh — hot path
    # ──────────────────────────────────────────────────────────────────
    def _set_unified_slice(self, new_idx):
        self.slice_unified_idx = int(new_idx)
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(self.slice_unified_idx)
        self.slice_slider.blockSignals(False)
        self._render_unified_slice()

    def _render_unified_slice(self, force_render=False):
        if not self.slice_stack:
            return
        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        image_hu = series['image_hu']
        spacing = series['spacing']
        meta = series.get('meta', {}) or {}
        nz, ny, nx = image_hu.shape
        sx = float(spacing[2])
        sy = float(spacing[1])

        slice_2d = image_hu[local_idx]
        lo, hi = self._wl_to_clim()

        if (self._last_rendered_series_idx != series_idx
                or self.slice_im is None):
            # Series transition — extent/aspect may differ, recreate imshow.
            self.slice_ax.clear()
            self._style_axes(self.slice_ax)
            desc = str(meta.get('series_description', f'Series {series_idx}'))
            self.slice_ax.set_title(
                f"[{series_idx}] {desc}", color='#ddd', fontsize=10
            )
            self.slice_im = self.slice_ax.imshow(
                slice_2d,
                cmap='gray',
                vmin=lo, vmax=hi,
                extent=(0.0, nx * sx, ny * sy, 0.0),
                aspect='equal',
                interpolation='bilinear',
            )
            self._last_rendered_series_idx = series_idx
            # ax.clear() detached our overlay artists — drop the stale refs.
            self._slice_2d_marker_artists = []
            self._slice_2d_measurement_artists = []
        else:
            self.slice_im.set_data(slice_2d)
            self.slice_im.set_clim(lo, hi)

        # Refresh 2D overlays (landmarks + active measurement) without
        # forcing their own draw — _maybe_render flushes the canvas once.
        self._refresh_2d_landmarks(draw=False)
        self._refresh_2d_measurement(draw=False)

        # 3D companion updates — both touch only the 3D plotter, so we
        # gate them together with the matplotlib redraw via _maybe_render.
        if self.slice_show_3d_indicator:
            self._update_3d_slice_indicator()
        if self.slice_hide_above_enabled:
            self._update_slice_clip_plane_origin()

        self._update_slice_status_label()
        self._maybe_render(force_render)

    def _update_clim_only(self, force_render=False):
        if self.slice_im is None:
            return
        lo, hi = self._wl_to_clim()
        self.slice_im.set_clim(lo, hi)
        self._update_slice_status_label()
        # Volume rendering shares the same Window/Level as the 2D view.
        # Push the new opacity TF in memory; _maybe_render below flushes
        # both canvases (2D + 3D) on the throttled cadence.
        has_volumes = bool(getattr(self, 'volume_actors', None))
        if has_volumes and hasattr(self, '_update_volume_opacity_transfer'):
            self._update_volume_opacity_transfer()
        # render_3d=True only when we actually have something to update
        # in the 3D plotter (volume opacity, clip indicator, clip plane).
        self._maybe_render(force_render, render_3d=has_volumes)

    def _maybe_render(self, force_render, render_3d=True):
        """Throttle 2D + 3D redraws together at ~60Hz.

        The cap rebuild (vtkCutter + vtkContourTriangulator on every
        bone actor) is the most expensive 3D update — gating it on the
        same throttle keeps drag-scroll smooth on large fused meshes.
        """
        now = time.time()
        if not force_render and now - self._slice_last_render < self._slice_min_render_interval:
            return
        self.slice_canvas.draw_idle()
        if render_3d:
            if self.slice_hide_above_enabled:
                self._update_clip_cap_mesh()
            need_3d_render = (
                self.slice_show_3d_indicator
                or self.slice_hide_above_enabled
                or bool(getattr(self, 'volume_actors', None))
            )
            if need_3d_render:
                self._render_plotter()
        self._slice_last_render = now

    def _render_plotter(self):
        try:
            self.plotter.render()
        except Exception:
            pass

    def _wl_to_clim(self):
        lo = self.window_level - self.window_width / 2.0
        hi = self.window_level + self.window_width / 2.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    # ──────────────────────────────────────────────────────────────────
    # 3D plotter — current slice indicator (red quad in base grid frame)
    # ──────────────────────────────────────────────────────────────────
    def _update_3d_slice_indicator(self):
        """Build (first call) or update (subsequent calls) the red quad
        marking the current 2D slice's location in the 3D plotter.

        Hot path: re-uses the cached vtkPolyData and only writes new
        points into it. No add_mesh / remove_actor on every drag frame,
        so the camera and zoom level are preserved automatically.
        """
        if not self.slice_show_3d_indicator:
            return
        if not self.slice_stack or self.plotter is None:
            return

        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        image_hu = series['image_hu']
        spacing = series['spacing']
        nz, ny, nx = image_hu.shape
        sz = float(spacing[0])
        sy = float(spacing[1])
        sx = float(spacing[2])
        z = local_idx * sz

        corners_series = np.array([
            [0.0,     0.0,     z],
            [nx * sx, 0.0,     z],
            [nx * sx, ny * sy, z],
            [0.0,     ny * sy, z],
        ], dtype=float)
        corners_base = self._series_grid_pts_to_base_grid(series_idx, corners_series)
        if corners_base is None:
            return

        if self._slice_3d_poly is None:
            # First time — add the actor once. reset_camera=False / render=False
            # keep the user's current zoom/angle intact.
            faces = np.array([4, 0, 1, 2, 3])
            self._slice_3d_poly = pv.PolyData(corners_base, faces)
            try:
                self.slice_3d_actor = self.plotter.add_mesh(
                    self._slice_3d_poly,
                    color='red',
                    opacity=0.25,
                    show_edges=True,
                    edge_color='red',
                    line_width=2,
                    lighting=False,
                    pickable=False,
                    reset_camera=False,
                    render=False,
                    name='slice_indicator_2d',
                )
            except TypeError:
                # Older pyvista signatures without 'render' kwarg.
                self.slice_3d_actor = self.plotter.add_mesh(
                    self._slice_3d_poly,
                    color='red',
                    opacity=0.25,
                    show_edges=True,
                    edge_color='red',
                    line_width=2,
                    lighting=False,
                    pickable=False,
                    reset_camera=False,
                    name='slice_indicator_2d',
                )
        else:
            # Hot path: just rewrite the points. No actor swap.
            self._slice_3d_poly.points = corners_base

    def _series_grid_pts_to_base_grid(self, series_idx, pts_series):
        """Transform Nx3 points from series-i grid into base series grid."""
        if not self.fusion_enabled or series_idx == self.base_series_index:
            return pts_series
        base_meta = self.all_series_data[self.base_series_index].get('meta', {}) or {}
        i_meta = self.all_series_data[series_idx].get('meta', {}) or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(i_meta)
        if T_base is None or T_i is None:
            return pts_series
        try:
            T_composite = np.linalg.inv(T_base) @ T_i
        except np.linalg.LinAlgError:
            return pts_series
        pts_h = np.hstack([pts_series, np.ones((len(pts_series), 1))])
        return (T_composite @ pts_h.T).T[:, :3]

    def _base_grid_pts_to_series_grid(self, series_idx, pts_base):
        """Inverse of _series_grid_pts_to_base_grid.

        Used to figure out *where* a landmark (stored in base-grid coords)
        falls on the currently displayed series-i slice — both for the
        z-tolerance check and to map (x, y) into the imshow extent.
        """
        if not self.fusion_enabled or series_idx == self.base_series_index:
            return pts_base
        base_meta = self.all_series_data[self.base_series_index].get('meta', {}) or {}
        i_meta = self.all_series_data[series_idx].get('meta', {}) or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(i_meta)
        if T_base is None or T_i is None:
            return pts_base
        try:
            T_composite = np.linalg.inv(T_base) @ T_i
            T_inv = np.linalg.inv(T_composite)
        except np.linalg.LinAlgError:
            return pts_base
        pts_h = np.hstack([pts_base, np.ones((len(pts_base), 1))])
        return (T_inv @ pts_h.T).T[:, :3]

    # ──────────────────────────────────────────────────────────────────
    # 2D landmark click + marker overlay
    # ──────────────────────────────────────────────────────────────────
    def _handle_2d_landmark_click(self, event):
        """Convert the click's image coords → base grid → existing
        landmark pipeline (sphere in 3D + row in table)."""
        if event.xdata is None or event.ydata is None:
            return
        if not self.slice_stack:
            return

        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        spacing = series['spacing']
        sz = float(spacing[0])

        # event.xdata / ydata are in mm (extent space) = series-i grid.
        x_mm = float(event.xdata)
        y_mm = float(event.ydata)
        z_mm = local_idx * sz

        pt_series = np.array([[x_mm, y_mm, z_mm]])
        pt_base = self._series_grid_pts_to_base_grid(series_idx, pt_series)[0]

        # Delegate to the existing landmark machinery — adds to the
        # central data dict, drops a red sphere in the 3D plotter,
        # refreshes the table, and recomputes LPS/RAS.
        if hasattr(self, 'on_landmark_picked'):
            try:
                self.on_landmark_picked(np.asarray(pt_base, dtype=float))
            except Exception as e:
                print(f"[2D viewer] Failed to place landmark: {e}")
                return

        # _refresh_landmark_table (called by on_landmark_picked) will
        # trigger our own 2D marker refresh via the hook in landmarks.py.

    def _refresh_2d_landmarks(self, draw=True):
        """Redraw landmark markers on the current 2D slice.

        Called whenever the landmark list or current slice changes. Only
        landmarks whose z falls within ±slice_thickness/2 of the visible
        slice are shown (so a pelvis landmark doesn't appear when scrolling
        through the knee, etc.).
        """
        # Tear down old marker artists. After ax.clear() the artists are
        # already detached, so remove() may raise — swallow it.
        for artist in self._slice_2d_marker_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._slice_2d_marker_artists = []

        if (not self.slice_stack
                or self.slice_ax is None
                or not getattr(self, 'landmark_data', None)):
            if draw and self.slice_canvas is not None:
                self.slice_canvas.draw_idle()
            return

        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        spacing = series['spacing']
        sz = float(spacing[0])
        sy = float(spacing[1])
        sx = float(spacing[2])
        z_local = local_idx * sz
        z_tol = sz * 0.5  # half slice thickness

        base_pts = np.array([np.asarray(entry['grid'], dtype=float)
                             for entry in self.landmark_data])
        series_pts = self._base_grid_pts_to_series_grid(series_idx, base_pts)

        offset = max(sx, sy) * 2.5
        for i, entry in enumerate(self.landmark_data):
            sp = series_pts[i]
            if abs(sp[2] - z_local) > z_tol:
                continue
            marker, = self.slice_ax.plot(
                sp[0], sp[1],
                marker='+',
                markersize=11,
                markeredgecolor='yellow',
                markeredgewidth=1.6,
                linestyle='None',
            )
            self._slice_2d_marker_artists.append(marker)
            text = self.slice_ax.text(
                sp[0] + offset, sp[1] + offset,
                str(entry.get('name', '?')),
                color='yellow',
                fontsize=8,
            )
            self._slice_2d_marker_artists.append(text)

        if draw and self.slice_canvas is not None:
            self.slice_canvas.draw_idle()

    # ──────────────────────────────────────────────────────────────────
    # 2D measurement overlay (distance / angle for selected landmarks)
    # ──────────────────────────────────────────────────────────────────
    def _refresh_2d_measurement(self, draw=True):
        """If 2-3 landmarks are selected in the table AND all sit on the
        current 2D slice, draw the measurement line + label."""
        for artist in self._slice_2d_measurement_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._slice_2d_measurement_artists = []

        if not self.slice_stack or self.slice_ax is None:
            if draw and self.slice_canvas is not None:
                self.slice_canvas.draw_idle()
            return

        rows = self._selected_landmark_rows()
        if len(rows) not in (2, 3):
            if draw and self.slice_canvas is not None:
                self.slice_canvas.draw_idle()
            return

        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        spacing = series['spacing']
        sz = float(spacing[0])
        z_local = local_idx * sz
        z_tol = sz * 0.5

        base_pts = np.array([np.asarray(self.landmark_data[r]['grid'], dtype=float)
                             for r in rows])
        series_pts = self._base_grid_pts_to_series_grid(series_idx, base_pts)

        if not np.all(np.abs(series_pts[:, 2] - z_local) <= z_tol):
            # Selection involves a landmark outside this slice — keep the
            # 3D measurement (handled by landmarks.py), just don't show
            # an incomplete one on the 2D canvas.
            if draw and self.slice_canvas is not None:
                self.slice_canvas.draw_idle()
            return

        if len(rows) == 2:
            p1, p2 = series_pts[0, :2], series_pts[1, :2]
            line, = self.slice_ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color='lime', linewidth=1.8,
            )
            self._slice_2d_measurement_artists.append(line)
            dist = float(np.linalg.norm(base_pts[0] - base_pts[1]))
            mid = (p1 + p2) * 0.5
            label = self.slice_ax.text(
                mid[0], mid[1], f"{dist:.1f} mm",
                color='lime', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6,
                          edgecolor='lime', boxstyle='round,pad=0.2'),
            )
            self._slice_2d_measurement_artists.append(label)
        else:  # 3 — angle at the middle (row index 1) point
            p1, vertex, p2 = series_pts[0, :2], series_pts[1, :2], series_pts[2, :2]
            line1, = self.slice_ax.plot(
                [vertex[0], p1[0]], [vertex[1], p1[1]],
                color='magenta', linewidth=1.8,
            )
            line2, = self.slice_ax.plot(
                [vertex[0], p2[0]], [vertex[1], p2[1]],
                color='magenta', linewidth=1.8,
            )
            self._slice_2d_measurement_artists.extend([line1, line2])
            v1 = base_pts[0] - base_pts[1]
            v2 = base_pts[2] - base_pts[1]
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom > 0:
                cos = float(np.dot(v1, v2)) / denom
                angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                label = self.slice_ax.text(
                    vertex[0], vertex[1], f"{angle:.1f}°",
                    color='magenta', fontsize=10,
                    bbox=dict(facecolor='black', alpha=0.6,
                              edgecolor='magenta', boxstyle='round,pad=0.2'),
                )
                self._slice_2d_measurement_artists.append(label)

        if draw and self.slice_canvas is not None:
            self.slice_canvas.draw_idle()

    def _selected_landmark_rows(self):
        """Indices of currently selected rows in the landmark table."""
        if not hasattr(self, 'landmark_table') or self.landmark_table is None:
            return []
        rows = sorted({item.row() for item in self.landmark_table.selectedItems()})
        return [r for r in rows if 0 <= r < len(self.landmark_data)]

    def _remove_3d_slice_indicator(self):
        if self._slice_3d_poly is None and self.slice_3d_actor is None:
            return
        if self.plotter is not None:
            try:
                self.plotter.remove_actor('slice_indicator_2d')
            except Exception:
                try:
                    self.plotter.remove_actor(self.slice_3d_actor)
                except Exception:
                    pass
        self.slice_3d_actor = None
        self._slice_3d_poly = None

    # ──────────────────────────────────────────────────────────────────
    # 3D plotter — hide-above clipping plane on every bone mesh
    # ──────────────────────────────────────────────────────────────────
    def _ensure_slice_clip_plane(self):
        """Lazy-create the shared vtkPlane used for the clip-above effect."""
        if self._slice_clip_plane is None:
            self._slice_clip_plane = vtk.vtkPlane()
            # Normal points to negative base-z → the kept side is "below"
            # (lower patient z = caudal in HFS). For non-axial fused series
            # this still cuts at the right z in base grid; the slight tilt
            # relative to the actual slice plane is invisible at typical
            # surgical-planning scale.
            self._slice_clip_plane.SetNormal(0.0, 0.0, -1.0)

    def _update_slice_clip_plane_origin(self):
        """Move the cached clip plane to the current slice's base-grid z.

        The plane reference is shared by every mapper that has it attached,
        so a single SetOrigin propagates to all bone actors automatically
        on the next render.
        """
        if not self.slice_stack:
            return
        self._ensure_slice_clip_plane()
        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        image_hu = series['image_hu']
        spacing = series['spacing']
        nz, ny, nx = image_hu.shape
        sz = float(spacing[0])
        sy = float(spacing[1])
        sx = float(spacing[2])
        # Centre of the current slice in series-i grid → base grid.
        center_series = np.array([[nx * sx / 2.0, ny * sy / 2.0, local_idx * sz]])
        center_base = self._series_grid_pts_to_base_grid(series_idx, center_series)
        if center_base is None:
            return
        self._slice_clip_plane.SetOrigin(0.0, 0.0, float(center_base[0, 2]))

    def _gather_bone_mesh_actors(self):
        """Every bone-representation actor that the hide-above clip plane
        should target (volumes in volume-render mode, plus any legacy
        mesh actors still hanging around).

        Volume actors are the visible representation in this build; mesh
        actors are kept hidden for STL export & bone-list features but we
        still attach the clip plane to them so re-enabling visibility
        anywhere doesn't reveal the unclipped mesh.
        """
        actors = []
        for actor in getattr(self, 'volume_actors', []) or []:
            if actor is not None:
                actors.append(actor)
        if getattr(self, 'current_mesh_actor', None) is not None:
            actors.append(self.current_mesh_actor)
        for actor in getattr(self, 'fusion_actors', []) or []:
            if actor is not None:
                actors.append(actor)
        for entry in getattr(self, 'separated_bones', []) or []:
            actor = entry.get('actor') if isinstance(entry, dict) else None
            if actor is not None:
                actors.append(actor)
        return actors

    def _apply_clip_to_all_actors(self):
        """Attach the shared clip plane to every current bone mesh actor."""
        if self._slice_clip_plane is None:
            return
        for actor in self._gather_bone_mesh_actors():
            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                mapper.RemoveAllClippingPlanes()
                mapper.AddClippingPlane(self._slice_clip_plane)
            except Exception:
                pass

    def _remove_clip_from_all_actors(self):
        for actor in self._gather_bone_mesh_actors():
            try:
                mapper = actor.GetMapper()
                if mapper is not None:
                    mapper.RemoveAllClippingPlanes()
            except Exception:
                pass

    def _reapply_slice_clipping_after_mesh_rebuild(self):
        """Called from update_base_mesh once a new mesh exists. Re-attaches
        the clip plane to fresh actors so the hide-above effect persists
        across threshold/smoothing changes."""
        if not self.slice_hide_above_enabled:
            return
        self._ensure_slice_clip_plane()
        self._update_slice_clip_plane_origin()
        self._apply_clip_to_all_actors()
        self._update_clip_cap_mesh()

    # ──────────────────────────────────────────────────────────────────
    # Clip-plane cap fill (so cut bones look solid, not hollow)
    #
    # Marching-cubes meshes are surface shells — the inside of every bone
    # is empty. With GPU mapper clipping alone, slicing through (e.g.) a
    # metal rod shows the inner surface of the shell, looking hollow.
    # To make it look solid we cut the mesh at the clipping plane to get
    # the boundary polylines, then triangulate those closed loops into a
    # filled cap and render it as a separate actor at the same z. Only
    # active while "hide above" is on; throttled with the same 60Hz cap
    # the slice canvas uses.
    # ──────────────────────────────────────────────────────────────────
    def _update_clip_cap_mesh(self):
        if not self.slice_hide_above_enabled or not self.slice_stack:
            self._remove_clip_cap_mesh()
            return
        if self._slice_clip_plane is None or self.plotter is None:
            self._remove_clip_cap_mesh()
            return

        # Volume rendering shows the interior natively — no shell, no
        # hollow look — so the contour-triangulated cap is unnecessary
        # (and would z-fight with the volume sample at the cut plane).
        if getattr(self, 'volume_actors', None):
            self._remove_clip_cap_mesh()
            return

        actors = self._gather_bone_mesh_actors()
        if not actors:
            self._remove_clip_cap_mesh()
            return

        appender = vtk.vtkAppendPolyData()
        any_cap = False
        for actor in actors:
            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                polydata = mapper.GetInput()
                if polydata is None:
                    continue
            except Exception:
                continue

            # 1) intersect mesh with the clip plane → line segments
            cutter = vtk.vtkCutter()
            cutter.SetCutFunction(self._slice_clip_plane)
            cutter.SetInputData(polydata)
            cutter.Update()
            cut = cutter.GetOutput()
            if cut.GetNumberOfPoints() == 0:
                continue

            # 2) stitch line segments into closed polylines
            stripper = vtk.vtkStripper()
            stripper.SetInputData(cut)
            stripper.JoinContiguousSegmentsOn()
            stripper.Update()
            loops = stripper.GetOutput()
            if loops.GetNumberOfCells() == 0:
                continue

            # 3) triangulate the closed loops → filled cap
            tri = vtk.vtkContourTriangulator()
            tri.SetInputData(loops)
            tri.Update()
            cap = tri.GetOutput()
            if cap.GetNumberOfCells() > 0:
                appender.AddInputData(cap)
                any_cap = True

        if not any_cap:
            self._remove_clip_cap_mesh()
            return

        appender.Update()
        cap_poly = pv.wrap(appender.GetOutput())

        # name= replaces the previous cap actor in a single call without
        # triggering a camera reset (we also pass reset_camera=False).
        try:
            self._slice_cap_actor = self.plotter.add_mesh(
                cap_poly,
                color='ivory',
                lighting=True,
                specular=0.0,
                reset_camera=False,
                render=False,
                pickable=False,
                name='slice_clip_cap',
            )
        except TypeError:
            # Older pyvista without 'render' kwarg.
            self._slice_cap_actor = self.plotter.add_mesh(
                cap_poly,
                color='ivory',
                lighting=True,
                specular=0.0,
                reset_camera=False,
                pickable=False,
                name='slice_clip_cap',
            )
        except Exception as e:
            print(f"[2D viewer] Cap rendering failed: {e}")

    def _remove_clip_cap_mesh(self):
        if self.plotter is None:
            self._slice_cap_actor = None
            return
        try:
            self.plotter.remove_actor('slice_clip_cap')
        except Exception:
            try:
                if self._slice_cap_actor is not None:
                    self.plotter.remove_actor(self._slice_cap_actor)
            except Exception:
                pass
        self._slice_cap_actor = None

    # ──────────────────────────────────────────────────────────────────
    # Status label
    # ──────────────────────────────────────────────────────────────────
    def _update_slice_status_label(self):
        if not self.slice_stack:
            self.slice_status_label.setText("Load a patient to view slices.")
            return
        series_idx, local_idx = self.slice_stack[self.slice_unified_idx]
        series = self.all_series_data[series_idx]
        nz = series['image_hu'].shape[0]
        desc = str(series.get('meta', {}).get('series_description',
                                              f'Series {series_idx}'))
        if len(desc) > 22:
            desc = desc[:19] + '…'
        total = len(self.slice_stack)
        self.slice_status_label.setText(
            f"Pos {self.slice_unified_idx + 1:>4}/{total}     "
            f"[{series_idx}] {desc}  slice {local_idx + 1:>3}/{nz}     "
            f"W: {self.window_width}   L: {self.window_level}"
        )
