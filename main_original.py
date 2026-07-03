import sys
import os
import time
from datetime import datetime
import pyvista as pv
import vtk
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QHBoxLayout, QWidget, QComboBox, 
                             QLabel, QSlider, QPushButton, QProgressDialog, QSpinBox, QCheckBox,
                             QTextEdit, QFileDialog, QMessageBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QScrollArea, QFrame)
from PyQt5.QtCore import Qt
import numpy as np
from scipy.ndimage import (
    binary_opening, binary_closing, binary_fill_holes,
    binary_erosion, generate_binary_structure,
    label as ndi_label, distance_transform_edt,
)
from skimage.segmentation import watershed
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Import our custom modules
from dicom_utils import load_scan, get_series_info, load_series, get_pixels_hu

# The folder where the patient directories are located
BASE_DATA_DIR = "11423945"


class CollapsibleSection(QWidget):
    """A section widget with a clickable header that toggles its content's visibility.

    Usage:
        section = CollapsibleSection("Particle Removal")
        section.addWidget(my_checkbox)
        section.addLayout(my_hbox)
        parent_layout.addWidget(section)
    """

    HEADER_STYLE = (
        "QPushButton { text-align: left; padding: 6px 8px; font-weight: bold; "
        "color: #1a3a5c; background: #d8e4f0; border: none; border-radius: 3px; }"
        "QPushButton:hover { background: #c4d4ec; }"
    )

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        self.header_btn = QPushButton()
        self.header_btn.setCheckable(True)
        self.header_btn.setChecked(expanded)
        self.header_btn.setCursor(Qt.PointingHandCursor)
        self.header_btn.setStyleSheet(self.HEADER_STYLE)
        self.header_btn.toggled.connect(self._on_toggled)
        outer.addWidget(self.header_btn)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 4, 4, 6)
        self.content_layout.setSpacing(4)
        outer.addWidget(self.content_widget)

        self.content_widget.setVisible(expanded)
        self._update_header()

    def _on_toggled(self, checked: bool):
        self.content_widget.setVisible(checked)
        self._update_header()

    def _update_header(self):
        arrow = "▼" if self.header_btn.isChecked() else "▶"
        self.header_btn.setText(f"{arrow}  {self._title}")

    # Forwarding helpers so the section looks like a layout to callers
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def addSpacing(self, n: int):
        self.content_layout.addSpacing(n)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("3D CT Bone Visualizer")
        self.resize(1024, 768)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Left Panel (Controls) — wrapped in a QScrollArea so it stays usable
        # even when the window is small or the panel grows over time.
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setFrameShape(QFrame.NoFrame)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setMinimumWidth(290)

        control_widget = QWidget()
        self.control_layout = QVBoxLayout(control_widget)
        self.control_layout.setContentsMargins(8, 8, 8, 8)
        self.control_layout.setSpacing(6)
        self.control_scroll.setWidget(control_widget)
        self.main_layout.addWidget(self.control_scroll, stretch=1)

        # Right Panel (3D Viewer)
        self.plotter = QtInteractor(self.central_widget)
        self.main_layout.addWidget(self.plotter.interactor, stretch=4)
        
        # Add axes to the plotter
        self.plotter.add_axes()

        # Data states (initialize BEFORE setup_controls so UI can read them)
        self.current_image_hu = None
        self.current_spacing = (1.0, 1.0, 1.0)
        self.volume_grid = None
        self.current_mesh_actor = None
        
        self.base_mesh = None
        self.cropping_bounds = None
        self.last_box_bounds = None
        self.scout_arrays = []
        self.current_scout_index = 0
        
        self.current_min_threshold = 300
        self.current_max_threshold = 3000
        
        self.current_meta_info = None
        self.all_series_data = []  # List of dicts with image_hu, spacing, meta for each series
        self._switching_series = False  # Guard to prevent re-entrant render

        # ===== Multi-Series Fusion State =====
        # When enabled, every CT series of the loaded patient is rendered
        # simultaneously in a common coordinate frame (the "base" series'
        # grid space). Each series is transformed via:
        #     base_grid <- inv(T_base) @ T_i @ series_i_grid
        # where T_x = grid->LPS transform built from each series' DICOM IPP/IOP.
        # That way pelvis/knee/ankle scans of the same patient appear at the
        # correct physical offsets while keeping their native spacing.
        self.fusion_enabled = True
        self.base_series_index = 0   # which series defines the common grid frame
        self.series_volume_grids = []  # list[pv.ImageData] aligned to self.all_series_data
        self.fusion_meshes = []        # list[(series_idx, transformed_mesh)] currently rendered
        self.fusion_actors = []        # parallel actor list (for cleanup)
        # Per-series visibility in fusion. Aligned to self.all_series_data.
        # Rebuilt on patient load (see _refresh_series_include_list).
        self.fusion_include_flags = []
        # When True, on every patient load we auto-detect "Mako protocol"
        # series (by keyword match in series_description) and uncheck the rest.
        # Catches calibration phantoms / unrelated reconstructions automatically.
        self.mako_only_mode = True
        # Case-insensitive substrings that identify a Mako protocol series.
        self.MAKO_KEYWORDS = ("mako", "stryker")
        # The dynamic UI list of per-series include checkboxes (rebuilt per patient).
        self.series_include_checkboxes = []

        # ===== Particle Removal State =====
        # Stage C: voxel-level (pre-contour) morphological opening.
        # binary_opening = erode N times → dilate N times. Removes thin
        # structures (≤ 2N voxels wide) while preserving thicker bone bodies.
        # Connectivity selects the structuring element:
        #   1 → 6-conn (faces only, conservative, smoother result)
        #   2 → 18-conn (faces + edges)
        #   3 → 26-conn (faces + edges + corners, most aggressive)
        # iterations == 0 ⇒ Stage C is skipped entirely (no-op).
        self.particle_removal_enabled = True
        self.opening_iterations = 1
        self.opening_connectivity = 1
        # Stage A: mesh-level post-cleanup
        self.mesh_cleanup_enabled = False
        self.keep_largest_only = False
        self.min_fragment_faces = 100
        # Stage B: interactive click-to-remove
        self.picking_enabled = False
        self._manual_undo_stack = []  # snapshots of base_mesh before each manual removal
        self._max_undo = 20

        # ===== Bone Separation State (Phase 1) =====
        # When ON, the single base mesh is replaced by N per-bone meshes
        # (one PolyData + actor per connected anatomical bone) so the user
        # can toggle individual bones on/off and inspect them in isolation.
        #
        # Pipeline (see _compute_separation_labels):
        #   bone_mask  ──► binary_closing(K)  ──► binary_fill_holes
        #              ──► binary_erosion(N) (break narrow bridges)
        #              ──► ndi_label (find separated cores)
        #              ──► filter cores by min voxel count
        #              ──► watershed(-distance, mask=closed_mask) to grow
        #                  each core back to the original closed bone shape
        #                  without letting neighboring labels overlap.
        # Each label → one binary contour() → one mesh.
        self.bone_separation_enabled = False     # True while separated meshes are shown
        self.hole_fill_iterations = 2            # binary_closing iterations (0..5)
        self.separation_erosion = 3              # binary_erosion iterations (0..10)
        self.min_bone_voxels = 1000              # noise cutoff in voxel count
        # Each entry: {'id', 'mesh', 'actor', 'visible', 'color', 'voxel_count', 'name'}
        self.separated_bones = []
        # Parallel list of QCheckBox widgets in the Bone Separation panel
        self.bone_separation_checkboxes = []
        # Snapshot of the original single-mesh actor's visibility so we can
        # restore it when the user clicks "Clear Separation".
        self._presep_base_actor_was_visible = True
        # Fusion: parallel visibility flags for fusion_actors before separation.
        self._presep_fusion_visibility = []

        # ===== Landmark Picking State =====
        self.landmark_picking_enabled = False
        # Each landmark is a dict:
        #   {'name': str,
        #    'grid': np.ndarray(3,)  PyVista grid coords (mm, origin (0,0,0)),
        #    'lps':  np.ndarray(3,) or None  DICOM patient LPS (mm),
        #    'ras':  np.ndarray(3,) or None  DICOM patient RAS (mm)}
        self.landmark_data = []
        self.landmark_actors = []   # parallel list of sphere actors for visualization
        self.landmark_radius_factor = 0.005  # sphere radius = factor * diag(bounds)
        self.landmark_counter = 0   # for default names L1, L2, ...
        # 'grid' | 'lps' | 'ras' — which coord system the table currently shows
        self.landmark_coord_system = 'grid'

        # ===== Measurement State =====
        # Actors currently visualizing the active measurement (lines + labels).
        # Rebuilt whenever the table selection changes.
        self.measurement_actors = []

        # ===== Session Restore State =====
        # When True, slider/checkbox/combobox handlers should avoid triggering
        # update_base_mesh — the session loader rebuilds everything once at the
        # end after all controls have been set.
        self._loading_session = False
        self.SESSION_FORMAT = "stanford_medicine_session"
        # v2: Stage C switched from connected-component size filter to
        #     scipy.ndimage.binary_opening. Old v1 sessions still load,
        #     with default opening parameters substituted in.
        # v3: Multi-series fusion. Adds 'fusion' block (enabled flag +
        #     base_series_index). Older sessions default to fusion ON
        #     with base = series 0.
        # v4: 'fusion' block extended with `mako_only` + `include_flags`
        #     so per-series visibility round-trips. Older sessions get
        #     Mako auto-detect at load time.
        self.SESSION_VERSION = 4

        # ===== Hover Preview State =====
        # Single yellow sphere that follows the mouse over the bone surface
        # while landmark picking mode is ON, giving visual feedback before clicking.
        self._hover_sphere_actor = None
        self._hover_picker = None
        self._hover_observer_id = None
        self._hover_last_render = 0.0
        self._hover_min_interval = 1.0 / 60.0  # cap re-render rate to ~60Hz

        self.setup_controls()
        
    def setup_controls(self):
        # ============================================================
        # 1) Patient Selection (always visible at top)
        # ============================================================
        self.control_layout.addWidget(QLabel("<b>Patient</b>"))
        self.patient_combo = QComboBox()
        self.control_layout.addWidget(self.patient_combo)
        self.load_patient_list()

        self.load_btn = QPushButton("Load CT && Render")
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.control_layout.addWidget(self.load_btn)

        self.export_btn = QPushButton("Export to STL")
        self.export_btn.clicked.connect(self.on_export_stl_clicked)
        self.control_layout.addWidget(self.export_btn)

        session_row = QHBoxLayout()
        self.save_session_btn = QPushButton("Save Session…")
        self.save_session_btn.setToolTip(
            "Save current patient, series, thresholds, cropping, particle\n"
            "removal settings, landmarks and camera into a single JSON file."
        )
        self.save_session_btn.clicked.connect(self.on_save_session_clicked)
        session_row.addWidget(self.save_session_btn)

        self.load_session_btn = QPushButton("Load Session…")
        self.load_session_btn.setToolTip(
            "Restore a previously saved session.\n"
            "The patient will be auto-loaded if necessary."
        )
        self.load_session_btn.clicked.connect(self.on_load_session_clicked)
        session_row.addWidget(self.load_session_btn)
        self.control_layout.addLayout(session_row)

        # ============================================================
        # 2) Series & Display (always visible)
        # ============================================================
        self.control_layout.addSpacing(8)
        self.control_layout.addWidget(QLabel("<b>Series &amp; Display</b>"))

        self.fusion_checkbox = QCheckBox("Fuse all series (same patient)")
        self.fusion_checkbox.setChecked(self.fusion_enabled)
        self.fusion_checkbox.setToolTip(
            "Render every CT series of this patient at once,\n"
            "aligned by DICOM IPP/IOP so each ST is placed at its true\n"
            "physical position. Click-to-remove and 3D Cropping are\n"
            "disabled while fusion is ON."
        )
        self.fusion_checkbox.stateChanged.connect(self.on_fusion_toggled)
        self.control_layout.addWidget(self.fusion_checkbox)

        self.control_layout.addWidget(QLabel("Select Scan / ST (base series in fusion):"))
        self.series_combo = QComboBox()
        self.series_combo.currentIndexChanged.connect(self.on_series_switched)
        self.control_layout.addWidget(self.series_combo)

        # ----- Per-series include list (which series go into fusion) -----
        self.series_include_section = CollapsibleSection("Series in fusion", expanded=False)

        self.mako_only_checkbox = QCheckBox("Mako protocol only (auto-detect)")
        self.mako_only_checkbox.setChecked(self.mako_only_mode)
        self.mako_only_checkbox.setToolTip(
            "If ON: only series whose description contains 'Mako' / 'Stryker'\n"
            "(case-insensitive) are included in the fusion. Phantoms and\n"
            "unrelated reconstructions are skipped automatically.\n"
            "If no matching series exist, every series is included as a fallback."
        )
        self.mako_only_checkbox.stateChanged.connect(self.on_mako_only_toggled)
        self.series_include_section.addWidget(self.mako_only_checkbox)

        # Container that the dynamic per-series checkboxes get added to.
        # Filled in _refresh_series_include_list().
        self.series_include_list_widget = QWidget()
        self.series_include_list_layout = QVBoxLayout(self.series_include_list_widget)
        self.series_include_list_layout.setContentsMargins(0, 4, 0, 0)
        self.series_include_list_layout.setSpacing(2)
        self.series_include_section.addWidget(self.series_include_list_widget)

        self.control_layout.addWidget(self.series_include_section)

        self.axes_checkbox = QCheckBox("Show 3D Axes")
        self.axes_checkbox.setChecked(True)
        self.axes_checkbox.stateChanged.connect(self.on_axes_toggled)
        self.control_layout.addWidget(self.axes_checkbox)

        self.control_layout.addWidget(QLabel("Smoothing Method:"))
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems([
            "None (Raw Pixels)", "Laplacian (Standard)", "Windowed Sinc (Advanced)"
        ])
        # Default to Windowed Sinc / Taubin — the medical-imaging standard:
        # low-pass smoothing that suppresses marching-cubes grid artifacts
        # while preserving volume and anatomical features.
        self.smooth_combo.setCurrentIndex(2)
        self.smooth_combo.currentIndexChanged.connect(self.on_smooth_changed)
        self.control_layout.addWidget(self.smooth_combo)

        # ============================================================
        # 3) Thresholds (always visible — most frequently used)
        # ============================================================
        self.control_layout.addSpacing(8)
        self.control_layout.addWidget(QLabel("<b>Thresholds (HU)</b>"))

        self.control_layout.addWidget(QLabel("Min Threshold:"))
        min_layout = QHBoxLayout()
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(-1000, 3000)
        self.min_slider.setValue(300)
        self.min_spinbox = QSpinBox()
        self.min_spinbox.setRange(-1000, 3000)
        self.min_spinbox.setValue(300)
        self.min_spinbox.setKeyboardTracking(False)
        min_layout.addWidget(self.min_slider)
        min_layout.addWidget(self.min_spinbox)
        self.control_layout.addLayout(min_layout)
        self.min_slider.valueChanged.connect(self.on_min_slider_changed)
        self.min_spinbox.valueChanged.connect(self.on_min_spinbox_changed)

        self.control_layout.addWidget(QLabel("Max Threshold:"))
        max_layout = QHBoxLayout()
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(-1000, 3000)
        self.max_slider.setValue(3000)
        self.max_spinbox = QSpinBox()
        self.max_spinbox.setRange(-1000, 3000)
        self.max_spinbox.setValue(3000)
        self.max_spinbox.setKeyboardTracking(False)
        max_layout.addWidget(self.max_slider)
        max_layout.addWidget(self.max_spinbox)
        self.control_layout.addLayout(max_layout)
        self.max_slider.valueChanged.connect(self.on_max_slider_changed)
        self.max_spinbox.valueChanged.connect(self.on_max_spinbox_changed)

        # ============================================================
        # 4) 3D Cropping (collapsible)
        # ============================================================
        self.control_layout.addSpacing(8)
        self.cropping_section = CollapsibleSection("3D Cropping", expanded=False)

        crop_row = QHBoxLayout()
        self.crop_checkbox = QCheckBox("Enable 3D Cropping Box")
        self.crop_checkbox.setChecked(False)
        self.crop_checkbox.stateChanged.connect(self.on_crop_toggled)
        crop_row.addWidget(self.crop_checkbox)

        self.crop_reset_btn = QPushButton("Reset")
        self.crop_reset_btn.setToolTip("Reset the crop box back to full bounds")
        self.crop_reset_btn.clicked.connect(self.on_crop_reset_clicked)
        crop_row.addWidget(self.crop_reset_btn)
        self.cropping_section.addLayout(crop_row)

        self.hide_handles_checkbox = QCheckBox("Hide Crop Handles")
        self.hide_handles_checkbox.setChecked(False)
        self.hide_handles_checkbox.setEnabled(False)
        self.hide_handles_checkbox.stateChanged.connect(self.on_hide_handles_toggled)
        self.cropping_section.addWidget(self.hide_handles_checkbox)

        self.control_layout.addWidget(self.cropping_section)

        # ============================================================
        # 5a) Bone Separation (collapsible)
        # ============================================================
        # Splits the cropped region into individually selectable bones
        # using morphological closing + watershed labeling. See the state
        # comments in __init__ for the algorithm overview.
        self.separation_section = CollapsibleSection("Bone Separation", expanded=False)

        hole_layout = QHBoxLayout()
        hole_layout.addWidget(QLabel("Hole fill iter:"))
        self.hole_fill_spinbox = QSpinBox()
        self.hole_fill_spinbox.setRange(0, 5)
        self.hole_fill_spinbox.setValue(int(self.hole_fill_iterations))
        self.hole_fill_spinbox.setKeyboardTracking(False)
        self.hole_fill_spinbox.setToolTip(
            "binary_closing iterations BEFORE fill_holes.\n"
            "0 = skip closing\n"
            "1–2 = fill small surface cracks (recommended)\n"
            "3+ = fill larger gaps (may bridge truly separate bones)"
        )
        self.hole_fill_spinbox.valueChanged.connect(self.on_hole_fill_changed)
        hole_layout.addWidget(self.hole_fill_spinbox)
        self.separation_section.addLayout(hole_layout)

        sep_iter_layout = QHBoxLayout()
        sep_iter_layout.addWidget(QLabel("Separation erosion:"))
        self.sep_erosion_spinbox = QSpinBox()
        self.sep_erosion_spinbox.setRange(0, 15)
        self.sep_erosion_spinbox.setValue(int(self.separation_erosion))
        self.sep_erosion_spinbox.setKeyboardTracking(False)
        self.sep_erosion_spinbox.setToolTip(
            "binary_erosion iterations used to BREAK narrow bridges between bones.\n"
            "Higher = separates bones with thicker junctions (e.g. fused vertebrae).\n"
            "Too high will fragment a single bone into several pieces."
        )
        self.sep_erosion_spinbox.valueChanged.connect(self.on_sep_erosion_changed)
        sep_iter_layout.addWidget(self.sep_erosion_spinbox)
        self.separation_section.addLayout(sep_iter_layout)

        min_vox_layout = QHBoxLayout()
        min_vox_layout.addWidget(QLabel("Min bone voxels:"))
        self.min_bone_vox_spinbox = QSpinBox()
        self.min_bone_vox_spinbox.setRange(0, 1000000)
        self.min_bone_vox_spinbox.setValue(int(self.min_bone_voxels))
        self.min_bone_vox_spinbox.setSingleStep(100)
        self.min_bone_vox_spinbox.setKeyboardTracking(False)
        self.min_bone_vox_spinbox.setToolTip(
            "Connected components smaller than this voxel count are discarded "
            "as noise after labeling."
        )
        self.min_bone_vox_spinbox.valueChanged.connect(self.on_min_bone_vox_changed)
        min_vox_layout.addWidget(self.min_bone_vox_spinbox)
        self.separation_section.addLayout(min_vox_layout)

        sep_btn_layout = QHBoxLayout()
        self.separate_btn = QPushButton("Separate Bones")
        self.separate_btn.setToolTip(
            "Run the separation pipeline on the current bone mask.\n"
            "Single mode: one series. Fusion mode: every included series, "
            "aligned to the base frame.\n"
            "Uses the active Cropping box when enabled."
        )
        self.separate_btn.clicked.connect(self.on_separate_bones_clicked)
        sep_btn_layout.addWidget(self.separate_btn)

        self.clear_separation_btn = QPushButton("Clear")
        self.clear_separation_btn.setToolTip("Restore the single combined bone mesh.")
        self.clear_separation_btn.setEnabled(False)
        self.clear_separation_btn.clicked.connect(self.on_clear_separation_clicked)
        sep_btn_layout.addWidget(self.clear_separation_btn)
        self.separation_section.addLayout(sep_btn_layout)

        # Status label below the buttons (e.g. "3 bones, took 1.4s")
        self.separation_status_label = QLabel("Not separated.")
        self.separation_status_label.setStyleSheet(
            "QLabel { color: #666; padding: 2px; }"
        )
        self.separation_section.addWidget(self.separation_status_label)

        # Container for the dynamic per-bone visibility checkboxes
        self.separation_list_widget = QWidget()
        self.separation_list_layout = QVBoxLayout(self.separation_list_widget)
        self.separation_list_layout.setContentsMargins(0, 4, 0, 0)
        self.separation_list_layout.setSpacing(2)
        self.separation_section.addWidget(self.separation_list_widget)

        self.control_layout.addWidget(self.separation_section)

        # ============================================================
        # 5) Particle Removal (collapsible — per user request)
        # ============================================================
        self.particle_section = CollapsibleSection("Particle Removal", expanded=False)

        # Stage C — morphological opening (erode N × dilate N) on the bone mask
        self.voxel_cleanup_checkbox = QCheckBox("Morphological denoise (opening)")
        self.voxel_cleanup_checkbox.setChecked(self.particle_removal_enabled)
        self.voxel_cleanup_checkbox.setToolTip(
            "Erode then dilate the bone mask BEFORE meshing.\n"
            "Removes thin/noisy structures while keeping bulk bone shape.\n"
            "Larger iterations = removes thicker structures (but also shrinks bone slightly)."
        )
        self.voxel_cleanup_checkbox.stateChanged.connect(self.on_voxel_cleanup_toggled)
        self.particle_section.addWidget(self.voxel_cleanup_checkbox)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("  Iterations:"))
        self.opening_iter_spinbox = QSpinBox()
        self.opening_iter_spinbox.setRange(0, 5)
        self.opening_iter_spinbox.setValue(int(self.opening_iterations))
        self.opening_iter_spinbox.setSingleStep(1)
        self.opening_iter_spinbox.setKeyboardTracking(False)
        self.opening_iter_spinbox.setToolTip(
            "0 = skip opening (no-op)\n"
            "1 = light denoise (removes ≤1-voxel-thin noise)\n"
            "2+ = more aggressive (also nibbles thin bone)"
        )
        self.opening_iter_spinbox.valueChanged.connect(self.on_opening_iter_changed)
        iter_layout.addWidget(self.opening_iter_spinbox)
        self.particle_section.addLayout(iter_layout)

        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("  Connectivity:"))
        self.opening_conn_combo = QComboBox()
        self.opening_conn_combo.addItem("6 (faces)", 1)
        self.opening_conn_combo.addItem("18 (faces+edges)", 2)
        self.opening_conn_combo.addItem("26 (full 3×3×3)", 3)
        # Map current connectivity to combo index
        for i in range(self.opening_conn_combo.count()):
            if self.opening_conn_combo.itemData(i) == self.opening_connectivity:
                self.opening_conn_combo.setCurrentIndex(i)
                break
        self.opening_conn_combo.setToolTip(
            "Structuring element for erode/dilate:\n"
            "6  – face neighbors only   (smoothest, most conservative)\n"
            "18 – + edge neighbors\n"
            "26 – + corner neighbors    (most aggressive)"
        )
        self.opening_conn_combo.currentIndexChanged.connect(self.on_opening_conn_changed)
        conn_layout.addWidget(self.opening_conn_combo, stretch=1)
        self.particle_section.addLayout(conn_layout)

        # Sync initial enabled state of sub-widgets with the toggle
        self.opening_iter_spinbox.setEnabled(self.particle_removal_enabled)
        self.opening_conn_combo.setEnabled(self.particle_removal_enabled)

        # Stage A
        self.mesh_cleanup_checkbox = QCheckBox("Clean mesh fragments")
        self.mesh_cleanup_checkbox.setChecked(self.mesh_cleanup_enabled)
        self.mesh_cleanup_checkbox.setToolTip(
            "Removes disconnected mesh components AFTER mesh generation.\n"
            "Useful as a second-pass safety net."
        )
        self.mesh_cleanup_checkbox.stateChanged.connect(self.on_mesh_cleanup_toggled)
        self.particle_section.addWidget(self.mesh_cleanup_checkbox)

        self.keep_largest_checkbox = QCheckBox("  Keep largest component only")
        self.keep_largest_checkbox.setChecked(self.keep_largest_only)
        self.keep_largest_checkbox.setEnabled(self.mesh_cleanup_enabled)
        self.keep_largest_checkbox.stateChanged.connect(self.on_keep_largest_toggled)
        self.particle_section.addWidget(self.keep_largest_checkbox)

        frag_layout = QHBoxLayout()
        frag_layout.addWidget(QLabel("  Min faces:"))
        self.fragment_faces_spinbox = QSpinBox()
        self.fragment_faces_spinbox.setRange(0, 1000000)
        self.fragment_faces_spinbox.setValue(self.min_fragment_faces)
        self.fragment_faces_spinbox.setSingleStep(50)
        self.fragment_faces_spinbox.setKeyboardTracking(False)
        self.fragment_faces_spinbox.setEnabled(
            self.mesh_cleanup_enabled and not self.keep_largest_only
        )
        self.fragment_faces_spinbox.valueChanged.connect(self.on_fragment_faces_changed)
        frag_layout.addWidget(self.fragment_faces_spinbox)
        self.particle_section.addLayout(frag_layout)

        # Stage B
        pick_layout = QHBoxLayout()
        self.pick_btn = QPushButton("Click-to-Remove: OFF")
        self.pick_btn.setCheckable(True)
        self.pick_btn.setToolTip(
            "Toggle on, then click on a particle in the 3D view.\n"
            "The connected component you click will be removed."
        )
        self.pick_btn.toggled.connect(self.on_picking_toggled)
        pick_layout.addWidget(self.pick_btn)

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.on_undo_clicked)
        pick_layout.addWidget(self.undo_btn)
        self.particle_section.addLayout(pick_layout)

        self.control_layout.addWidget(self.particle_section)

        # ============================================================
        # 6) Landmark Points (collapsible, expanded by default — main feature)
        # ============================================================
        self.landmarks_section = CollapsibleSection("Landmark Points", expanded=True)

        lm_btn_row = QHBoxLayout()
        self.landmark_pick_btn = QPushButton("Pick Landmark: OFF")
        self.landmark_pick_btn.setCheckable(True)
        self.landmark_pick_btn.setToolTip(
            "Toggle on, then click on the bone surface to mark a point.\n"
            "Each click adds a row with its (X, Y, Z) coordinates in mm."
        )
        self.landmark_pick_btn.toggled.connect(self.on_landmark_picking_toggled)
        lm_btn_row.addWidget(self.landmark_pick_btn)

        self.landmark_clear_btn = QPushButton("Clear All")
        self.landmark_clear_btn.clicked.connect(self.on_clear_landmarks_clicked)
        lm_btn_row.addWidget(self.landmark_clear_btn)
        self.landmarks_section.addLayout(lm_btn_row)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Coord:"))
        self.landmark_coord_combo = QComboBox()
        self.landmark_coord_combo.addItem("Grid (mm)", "grid")
        self.landmark_coord_combo.addItem("LPS (DICOM)", "lps")
        self.landmark_coord_combo.addItem("RAS", "ras")
        self.landmark_coord_combo.setToolTip(
            "Grid: PyVista volume-grid coords (origin at (0,0,0))\n"
            "LPS:  DICOM patient coords (Left/Posterior/Superior)\n"
            "RAS:  Right/Anterior/Superior (Slicer / common medical imaging)"
        )
        self.landmark_coord_combo.currentIndexChanged.connect(self.on_landmark_coord_changed)
        coord_row.addWidget(self.landmark_coord_combo, stretch=1)

        self.landmark_export_btn = QPushButton("Export…")
        self.landmark_export_btn.setToolTip("Export all landmark points to CSV or JSON.")
        self.landmark_export_btn.clicked.connect(self.on_landmark_export_clicked)
        coord_row.addWidget(self.landmark_export_btn)
        self.landmarks_section.addLayout(coord_row)

        self.landmark_table = QTableWidget(0, 6)
        self._update_landmark_table_header()
        header = self.landmark_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.landmark_table.verticalHeader().setVisible(False)
        self.landmark_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.landmark_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.landmark_table.setMaximumHeight(280)
        self.landmark_table.itemChanged.connect(self._on_landmark_table_item_changed)
        self.landmark_table.itemSelectionChanged.connect(self._on_landmark_selection_changed)
        self._suppress_table_signal = False
        self.landmarks_section.addWidget(self.landmark_table)

        self.measurement_label = QLabel(
            "<i>Select 2 rows for distance, 3 rows for angle (Ctrl+Click).</i>"
        )
        self.measurement_label.setWordWrap(True)
        self.measurement_label.setStyleSheet(
            "QLabel { padding: 6px; background: #f0f0f0; border-radius: 4px; }"
        )
        self.landmarks_section.addWidget(self.measurement_label)

        self.control_layout.addWidget(self.landmarks_section)

        # ============================================================
        # 7) Stretch then bottom bar (Scout / Patient Info)
        # ============================================================
        # Apply initial mutual-exclusion state based on default fusion setting.
        # Click-to-remove and Bone Separation are single-series-only because
        # they operate on a single mesh / single label volume. 3D cropping
        # and landmark picking both work in fusion mode.
        if self.fusion_enabled:
            self.pick_btn.setEnabled(False)

        # Initial "no patient loaded" placeholder for the series include list
        self._refresh_series_include_list()

        self.control_layout.addStretch()

        bottom_row = QHBoxLayout()
        self.scout_btn = QPushButton("Open Scout Viewer")
        self.scout_btn.clicked.connect(self.open_scout_window)
        bottom_row.addWidget(self.scout_btn)

        self.info_btn = QPushButton("Open Patient Info")
        self.info_btn.clicked.connect(self.open_info_window)
        bottom_row.addWidget(self.info_btn)
        self.control_layout.addLayout(bottom_row)

        # ============================================================
        # External windows (hidden until opened from the bottom buttons)
        # ============================================================
        # Patient Info window
        self.info_window = QMainWindow(self)
        self.info_window.setWindowTitle("Patient & Scan Information")
        self.info_window.resize(400, 300)
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setText("Load a patient to see information here.")
        info_layout.addWidget(self.info_text)
        self.info_window.setCentralWidget(info_widget)

        # Scout window
        self.scout_window = QMainWindow(self)
        self.scout_window.setWindowTitle("Scout View (Navigation Map)")
        self.scout_window.resize(600, 800)
        scout_widget = QWidget()
        scout_layout = QVBoxLayout(scout_widget)
        self.scout_fig = Figure(figsize=(6, 8))
        self.scout_canvas = FigureCanvas(self.scout_fig)
        self.scout_ax = self.scout_fig.add_subplot(111)
        self.scout_ax.axis('off')
        self.scout_toolbar = NavigationToolbar(self.scout_canvas, self.scout_window)
        scout_layout.addWidget(self.scout_toolbar)
        scout_layout.addWidget(self.scout_canvas)

        scout_nav_layout = QHBoxLayout()
        self.scout_prev_btn = QPushButton("< Prev")
        self.scout_prev_btn.clicked.connect(self.on_scout_prev)
        self.scout_next_btn = QPushButton("Next >")
        self.scout_next_btn.clicked.connect(self.on_scout_next)
        scout_nav_layout.addWidget(self.scout_prev_btn)
        scout_nav_layout.addWidget(self.scout_next_btn)
        scout_layout.addLayout(scout_nav_layout)

        self.scout_window.setCentralWidget(scout_widget)

    def open_scout_window(self):
        self.scout_window.show()
        self.scout_window.raise_()
        self.scout_window.activateWindow()

    def open_info_window(self):
        self.info_window.show()
        self.info_window.raise_()
        self.info_window.activateWindow()

    # ----- Per-series fusion include list -----
    def _autodetect_mako_flags(self):
        """Return a list[bool] matching all_series_data, with True for series
        whose description matches any of MAKO_KEYWORDS. Falls back to
        all-True when no series matches (don't strand the user with nothing)."""
        flags = []
        any_hit = False
        for sd in self.all_series_data:
            desc = (sd.get('meta', {}) or {}).get('series_description', '') or ''
            d_low = str(desc).lower()
            hit = any(k in d_low for k in self.MAKO_KEYWORDS)
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

    def show_scout_image(self):
        """Displays the current scout image in the scout viewer window."""
        self.scout_ax.clear()
        if self.scout_arrays:
            idx = self.current_scout_index
            self.scout_ax.imshow(self.scout_arrays[idx], cmap='gray')
            self.scout_ax.set_title(f"Scout {idx + 1} / {len(self.scout_arrays)}")
        else:
            self.scout_ax.text(0.5, 0.5, "No Scout Available", 
                               horizontalalignment='center', verticalalignment='center')
        self.scout_ax.axis('off')
        self.scout_fig.tight_layout()
        self.scout_canvas.draw()

    def on_scout_prev(self):
        if self.scout_arrays:
            self.current_scout_index = (self.current_scout_index - 1) % len(self.scout_arrays)
            self.show_scout_image()

    def on_scout_next(self):
        if self.scout_arrays:
            self.current_scout_index = (self.current_scout_index + 1) % len(self.scout_arrays)
            self.show_scout_image()

    def load_patient_list(self):
        if os.path.exists(BASE_DATA_DIR):
            patients = [d for d in os.listdir(BASE_DATA_DIR) 
                        if os.path.isdir(os.path.join(BASE_DATA_DIR, d))]
            self.patient_combo.addItems(patients)
        else:
            self.patient_combo.addItem("Data folder not found")

    def on_axes_toggled(self, state):
        if state == Qt.Checked:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self.plotter.update()
        
    def on_export_stl_clicked(self):
        if self.base_mesh is None:
            QMessageBox.warning(self, "No Data", "Please load a patient and render the 3D model first.")
            return
            
        base_dir = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not base_dir:
            return
            
        try:
            # Generate folder name: PatientName_StudyDate_HHMMSS_3D_Export
            name = "UnknownPatient"
            date_str = "UnknownDate"
            if self.current_meta_info:
                name = self.current_meta_info.get('patient_name', 'UnknownPatient')
                date_str = self.current_meta_info.get('study_date', 'UnknownDate')
                
            # Clean up strings for folder names (remove invalid chars)
            name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            
            timestamp = datetime.now().strftime("%H%M%S")
            folder_name = f"{name}_{date_str}_{timestamp}_3D_Export"
            export_dir = os.path.join(base_dir, folder_name)
            
            os.makedirs(export_dir, exist_ok=True)
            stl_path = os.path.join(export_dir, "mesh.stl")
            report_path = os.path.join(export_dir, "export_report.txt")
            
            # Decide what to export. In fusion mode we merge every visible
            # series into one STL (cropping is disabled while fusing).
            cropping_info = "No Cropping"
            if self.fusion_enabled and self.fusion_meshes:
                meshes = [m for _, m in self.fusion_meshes
                          if m is not None and m.n_points > 0]
                if not meshes:
                    QMessageBox.warning(self, "Nothing to export",
                                        "No fused meshes are currently rendered.")
                    return
                mesh_to_export = meshes[0]
                for extra in meshes[1:]:
                    try:
                        mesh_to_export = mesh_to_export.merge(extra)
                    except Exception:
                        # Fall back to PolyData concatenation
                        mesh_to_export = mesh_to_export + extra
                cropping_info = f"Fusion of {len(meshes)} series"
            else:
                mesh_to_export = self.base_mesh
                if self.cropping_bounds is not None and self.crop_checkbox.isChecked():
                    mesh_to_export = mesh_to_export.clip_box(
                        self.cropping_bounds, invert=False
                    )
                    cropping_info = f"Cropped to Box Bounds: {self.cropping_bounds.bounds}"

            mesh_to_export.save(stl_path)
            
            # Write Report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"--- 3D CT Reconstruction Export Report ---\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("[Patient & Scan Info]\n")
                if self.current_meta_info:
                    f.write(f"Patient Name: {self.current_meta_info.get('patient_name', 'N/A')}\n")
                    f.write(f"Study Date: {self.current_meta_info.get('study_date', 'N/A')}\n")
                    f.write(f"Modality: {self.current_meta_info.get('modality', 'N/A')}\n")
                    f.write(f"Series Description: {self.current_meta_info.get('series_description', 'N/A')}\n")
                    f.write(f"Series UID: {self.current_meta_info.get('series_uid', 'N/A')}\n")
                f.write("\n")
                
                f.write("[Reconstruction Parameters]\n")
                st_val = self.current_meta_info.get('slice_thickness', 'N/A') if self.current_meta_info else 'N/A'
                f.write(f"Slice Thickness: {st_val} mm\n")
                f.write(f"Min HU Threshold: {self.current_min_threshold}\n")
                f.write(f"Max HU Threshold: {self.current_max_threshold}\n")
                f.write(f"Smoothing Method: {self.smooth_combo.currentText()}\n")
                f.write(f"Cropping / Fusion: {cropping_info}\n")
                f.write(f"Fusion Mode: {'ON' if self.fusion_enabled else 'OFF'}\n")
                if self.fusion_enabled and self.fusion_meshes:
                    f.write("Series included in fusion:\n")
                    for idx, _ in self.fusion_meshes:
                        m = self.all_series_data[idx].get('meta', {})
                        f.write(
                            f"  - [{idx}] {m.get('series_description', 'N/A')}"
                            f" (ST={m.get('slice_thickness', '?')}mm)\n"
                        )

                f.write("\n[Particle Removal]\n")
                if self.particle_removal_enabled and self.opening_iterations > 0:
                    conn_label = {1: "6", 2: "18", 3: "26"}.get(
                        self.opening_connectivity, str(self.opening_connectivity)
                    )
                    f.write(
                        f"Voxel-level (Stage C): ON, morphological opening "
                        f"(iterations={self.opening_iterations}, connectivity={conn_label})\n"
                    )
                else:
                    f.write("Voxel-level (Stage C): OFF\n")
                if self.mesh_cleanup_enabled:
                    if self.keep_largest_only:
                        f.write("Mesh-level (Stage A): ON, keep largest component only\n")
                    else:
                        f.write(f"Mesh-level (Stage A): ON, min faces per fragment = {self.min_fragment_faces}\n")
                else:
                    f.write("Mesh-level (Stage A): OFF\n")
                
            QMessageBox.information(self, "Success", f"Successfully exported to folder:\n{export_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to save STL and Report:\n{str(e)}")
        
    # ----- Cropping helpers (single + fusion aware) -----
    def _full_render_bounds(self):
        """Aggregate bounds for whatever is currently on screen.

        - Fusion mode: union of every visible fusion-mesh's bounds, expressed
          in the base series' grid frame (which is the frame the crop widget
          lives in, since all clipping happens there).
        - Single mode: the active volume grid's bounds.
        Returns None when nothing is rendered yet.
        """
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

    def _rerender_for_crop(self):
        """Re-run whichever renderer owns the scene so crop changes show up."""
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

    def on_picking_toggled(self, checked):
        self.picking_enabled = checked
        if checked:
            # Mutual exclusion: turn off landmark picking if it was on
            if hasattr(self, 'landmark_pick_btn') and self.landmark_pick_btn.isChecked():
                self.landmark_pick_btn.setChecked(False)
            self.pick_btn.setText("Click-to-Remove: ON")
            # left_clicking=True is REQUIRED for picking to trigger on left mouse click
            # (PyVista default binds picking to the 'P' key only).
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

            # Remove cells belonging to the picked region (keeps geometry of others intact)
            region_ids_cell = labeled.cell_data['RegionId']
            cells_to_remove = np.where(region_ids_cell == target_region)[0]
            if len(cells_to_remove) == 0:
                self._manual_undo_stack.pop()
                if not self._manual_undo_stack:
                    self.undo_btn.setEnabled(False)
                return

            cleaned = labeled.remove_cells(cells_to_remove)
            # Ensure PolyData so STL export & downstream PolyData ops keep working
            if not isinstance(cleaned, pv.PolyData):
                cleaned = cleaned.extract_surface()
            # Drop helper arrays so they don't leak into export
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
    def on_landmark_picking_toggled(self, checked):
        self.landmark_picking_enabled = checked
        if checked:
            # Mutual exclusion: turn off click-to-remove if it was on
            if hasattr(self, 'pick_btn') and self.pick_btn.isChecked():
                self.pick_btn.setChecked(False)
            self.landmark_pick_btn.setText("Pick Landmark: ON")
            # Prefer surface-point picking (samples a point on the mesh surface).
            # Fall back to point picking if not available in this PyVista version.
            # left_clicking=True is REQUIRED so left mouse click triggers the pick.
            try:
                self.plotter.enable_surface_point_picking(
                    callback=self.on_landmark_picked,
                    show_message="Click on the bone surface to mark a landmark",
                    show_point=False,
                    left_clicking=True,
                    pickable_window=False,
                )
            except (TypeError, AttributeError):
                try:
                    self.plotter.enable_point_picking(
                        callback=self.on_landmark_picked,
                        show_message="Click on the bone surface to mark a landmark",
                        use_picker='cell',
                        show_point=False,
                        pickable_window=False,
                        left_clicking=True,
                    )
                except Exception as e:
                    print(f"[Landmark] Failed to enable picking: {e}")
            # Turn on the hover preview sphere
            self._enable_hover_preview()
        else:
            self.landmark_pick_btn.setText("Pick Landmark: OFF")
            try:
                self.plotter.disable_picking()
            except Exception:
                pass
            self._disable_hover_preview()

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
            }
            self.landmark_data.append(entry)

            # Visualize as a small red sphere. Radius scales with scene size.
            radius = self._estimate_landmark_radius()
            sphere = pv.Sphere(radius=radius, center=grid_pt)
            actor = self.plotter.add_mesh(
                sphere,
                color="red",
                specular=0.3,
                smooth_shading=True,
                pickable=False,  # so future clicks pass through to the bone mesh
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

    def _on_landmark_selection_changed(self):
        if self._suppress_table_signal:
            return
        sel_model = self.landmark_table.selectionModel()
        if sel_model is None:
            return
        rows = sorted({idx.row() for idx in sel_model.selectedRows()})
        # Filter out invalid rows defensively
        rows = [r for r in rows if 0 <= r < len(self.landmark_data)]

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
    def on_save_session_clicked(self):
        if not self.all_series_data:
            QMessageBox.warning(self, "No Session", "Load a patient before saving a session.")
            return

        meta = self.current_meta_info or {}
        name = "".join(c for c in str(meta.get('patient_name', 'patient'))
                       if c.isalnum() or c in (' ', '-', '_')).strip() or 'patient'
        date_str = str(meta.get('study_date', ''))
        timestamp = datetime.now().strftime('%H%M%S')
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
            state = self._collect_session_state()
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Session Saved",
                                    f"Session saved to:\n{path}")
        except Exception as e:
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

        if state.get('format') != self.SESSION_FORMAT:
            QMessageBox.warning(self, "Wrong Format",
                                "This file does not look like a Stanford Medicine session.")
            return
        if int(state.get('version', 0)) > self.SESSION_VERSION:
            QMessageBox.warning(
                self, "Newer Format",
                f"This session was saved by a newer app version "
                f"(v{state.get('version')}). Some fields may be ignored."
            )

        try:
            self._apply_session_state(state)
            QMessageBox.information(self, "Session Loaded",
                                    f"Restored session from:\n{path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Apply Failed",
                                 f"Session loaded but could not be fully applied:\n{e}")

    def _collect_session_state(self):
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
            }
            if e.get('lps') is not None:
                item['lps'] = [float(v) for v in e['lps']]
            if e.get('ras') is not None:
                item['ras'] = [float(v) for v in e['ras']]
            landmarks.append(item)

        state = {
            'format': self.SESSION_FORMAT,
            'version': self.SESSION_VERSION,
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
                'max_threshold': int(self.current_max_threshold),
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
        return state

    def _apply_session_state(self, state):
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
            max_th = int(render.get('max_threshold', self.current_max_threshold))
            self._set_widget_value(self.min_slider, min_th)
            self._set_widget_value(self.max_slider, max_th)
            self._set_widget_value(self.min_spinbox, min_th)
            self._set_widget_value(self.max_spinbox, max_th)
            self.current_min_threshold = min_th
            self.current_max_threshold = max_th

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

    # ===== Hover Preview (yellow sphere follows mouse) =====
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

    def on_min_slider_changed(self, value):
        self.min_spinbox.blockSignals(True)
        self.min_spinbox.setValue(value)
        self.min_spinbox.blockSignals(False)
        self.update_thresholds()
        
    def on_min_spinbox_changed(self, value):
        self.min_slider.blockSignals(True)
        self.min_slider.setValue(value)
        self.min_slider.blockSignals(False)
        self.update_thresholds()
        
    def on_max_slider_changed(self, value):
        self.max_spinbox.blockSignals(True)
        self.max_spinbox.setValue(value)
        self.max_spinbox.blockSignals(False)
        self.update_thresholds()
        
    def on_max_spinbox_changed(self, value):
        self.max_slider.blockSignals(True)
        self.max_slider.setValue(value)
        self.max_slider.blockSignals(False)
        self.update_thresholds()
        
    def update_thresholds(self):
        self.current_min_threshold = self.min_slider.value()
        self.current_max_threshold = self.max_slider.value()
        
        if self.current_min_threshold >= self.current_max_threshold:
            return # Invalid range
            
        if self.volume_grid is not None:
            self.update_base_mesh()

    def on_load_clicked(self):
        patient_id = self.patient_combo.currentText()
        if not patient_id or patient_id == "Data folder not found":
            return
            
        patient_dir = os.path.join(BASE_DATA_DIR, patient_id)
        
        if not os.path.exists(patient_dir):
            return

        # Disable click-to-remove during a fresh load to avoid stale picking on old mesh
        if self.picking_enabled:
            self.pick_btn.setChecked(False)
        # Disable landmark picking and drop any existing landmarks
        # (their coordinates only make sense in the previous patient's frame)
        if self.landmark_picking_enabled:
            self.landmark_pick_btn.setChecked(False)
        self._clear_measurement_visualization()
        if self.landmark_data:
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
        # Destroy hover preview so it will be rebuilt with the new mesh's scale
        self._destroy_hover_preview()

        # Show a simple progress dialog
        progress = QProgressDialog("Loading DICOM files...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        QApplication.processEvents()
        
        try:
            # cache_v5 — new schema after z-gap sub-stack split in dicom_utils.
            # Old v4 caches are intentionally not used (they'd merge multi-region
            # volumes back into one stack and re-introduce the welding bug).
            cache_file = os.path.join(patient_dir, f"{patient_id}_cache_v5.npz")
            self.scout_arrays = []
            self.current_scout_index = 0
            self.all_series_data = []
            # Drop per-series ImageData cache from the previous patient
            self.series_volume_grids = []
            self.base_series_index = 0
            # Force the fusion diagnostics block to print once on next render
            self._fusion_diag_printed = False
            # Any separated-bone state from a previous patient is invalid now
            self._auto_clear_separation_on_remesh("new patient loaded")
            
            if os.path.exists(cache_file):
                # ---- FAST PATH: Load from v3 cache ----
                progress.setLabelText("Loading cached data (Fast Load)...")
                progress.setValue(20)
                QApplication.processEvents()
                
                data = np.load(cache_file, allow_pickle=True)
                
                # Load scout images
                i = 0
                while f'scout_{i}' in data:
                    self.scout_arrays.append(data[f'scout_{i}'])
                    i += 1
                
                # Load all series
                series_count = int(data['series_count'])
                for idx in range(series_count):
                    self.all_series_data.append({
                        'image_hu': data[f'series_{idx}_image_hu'],
                        'spacing': tuple(data[f'series_{idx}_spacing']),
                        'meta': data[f'series_{idx}_meta'].item(),
                    })
                    progress.setValue(20 + int(60 * (idx + 1) / series_count))
                    QApplication.processEvents()
                    
            else:
                # ---- SLOW PATH: Load raw DICOMs, process ALL series ----
                progress.setLabelText("Loading DICOM files...")
                progress.setValue(5)
                QApplication.processEvents()
                series_dict, scout_slices = load_scan(patient_dir)
                
                series_info = get_series_info(series_dict)
                if not series_info:
                    QMessageBox.warning(self, "No Data", "No valid CT series found in this folder.")
                    return
                
                # Collect ALL scout images
                for sc in scout_slices:
                    self.scout_arrays.append(sc.pixel_array)
                
                # Process each valid series
                total = len(series_info)
                for idx, info in enumerate(series_info):
                    series_key = info['key']
                    progress.setLabelText(f"Processing series {idx+1}/{total}: {info['description']} (ST:{info['slice_thickness']}mm)...")
                    progress.setValue(10 + int(50 * idx / total))
                    QApplication.processEvents()
                    
                    slices, spacing = load_series(series_dict, series_key)
                    image_hu = get_pixels_hu(slices)
                    
                    sample = slices[0]
                    meta = {
                        'patient_name': str(getattr(sample, 'PatientName', 'N/A')),
                        'study_date': getattr(sample, 'StudyDate', 'N/A'),
                        'series_description': getattr(sample, 'SeriesDescription', 'N/A'),
                        'series_uid': info['uid'],
                        'sub_idx': info.get('sub_idx', 0),
                        'modality': getattr(sample, 'Modality', 'N/A'),
                        'slice_thickness': info['slice_thickness'],
                        # z-range in the slice-stacking direction (LPS mm)
                        # — useful for labels and for diagnosing sub-stack splits.
                        'z_min': float(info.get('z_min', 0.0)),
                        'z_max': float(info.get('z_max', 0.0)),
                    }

                    # DICOM patient-coordinate (LPS) geometry from first sorted slice.
                    # IPP   : Image Position Patient  (origin of first slice, LPS, mm)
                    # IOP   : Image Orientation Patient (row + column direction cosines, LPS)
                    # Normal: cross(row_dir, col_dir) — the slice-stacking direction.
                    try:
                        iop = np.array(sample.ImageOrientationPatient, dtype=float)
                        ipp = np.array(sample.ImagePositionPatient, dtype=float)
                        row_dir = iop[:3]
                        col_dir = iop[3:]
                        normal_dir = np.cross(row_dir, col_dir)
                        meta['ipp_first'] = ipp.tolist()
                        meta['row_dir'] = row_dir.tolist()
                        meta['col_dir'] = col_dir.tolist()
                        meta['normal_dir'] = normal_dir.tolist()
                        meta['patient_position'] = str(getattr(sample, 'PatientPosition', 'N/A'))
                    except Exception:
                        # Geometry unavailable: LPS/RAS conversion will be disabled
                        pass

                    self.all_series_data.append({
                        'image_hu': image_hu,
                        'spacing': spacing,
                        'meta': meta,
                    })
                
                # Save cache (v3 format)
                progress.setLabelText("Saving all series to cache...")
                progress.setValue(65)
                QApplication.processEvents()
                
                save_dict = {'series_count': np.array(len(self.all_series_data))}
                for i, sa in enumerate(self.scout_arrays):
                    save_dict[f'scout_{i}'] = sa
                for idx, sd in enumerate(self.all_series_data):
                    save_dict[f'series_{idx}_image_hu'] = sd['image_hu']
                    save_dict[f'series_{idx}_spacing'] = sd['spacing']
                    save_dict[f'series_{idx}_meta'] = np.array(sd['meta'])
                
                np.savez_compressed(cache_file, **save_dict)
            
            # Populate the series dropdown
            progress.setLabelText("Preparing UI...")
            progress.setValue(80)
            QApplication.processEvents()
            
            self._switching_series = True  # Block on_series_switched during population
            self.series_combo.clear()
            for sd in self.all_series_data:
                m = sd['meta']
                z_min = m.get('z_min')
                z_max = m.get('z_max')
                if z_min is not None and z_max is not None and z_min != z_max:
                    z_part = f", z[{z_min:.0f}…{z_max:.0f}]"
                else:
                    z_part = ""
                sub_idx = m.get('sub_idx', 0)
                sub_part = f" #{sub_idx}" if sub_idx else ""
                label = (
                    f"{m.get('series_description', 'N/A')}{sub_part} "
                    f"(ST: {m.get('slice_thickness', '?')}mm, "
                    f"{sd['image_hu'].shape[0]} slices{z_part})"
                )
                self.series_combo.addItem(label)
            self._switching_series = False

            # Build fusion include flags (auto-detect Mako protocol if requested)
            if self.mako_only_mode:
                self.fusion_include_flags = self._autodetect_mako_flags()
            else:
                self.fusion_include_flags = [True] * len(self.all_series_data)
            # Pick a default base series that's actually included
            if any(self.fusion_include_flags):
                self.base_series_index = self.fusion_include_flags.index(True)
            else:
                self.base_series_index = 0
            self._refresh_series_include_list()

            # Show scout image
            self.show_scout_image()

            # Activate the base series (this triggers on_series_switched)
            progress.setLabelText("Generating 3D Surface...")
            progress.setValue(90)
            QApplication.processEvents()

            if self.all_series_data:
                target = self.base_series_index
                self.series_combo.setCurrentIndex(target)
                self.on_series_switched(target)
            
            progress.setValue(100)
            
        except Exception as e:
            print(f"Error loading patient data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()

    # ----- Shared mesh-pipeline helpers (used by both single & fusion paths) -----
    def _build_image_data(self, image_hu, spacing):
        """Wrap a HU array into a pv.ImageData with axis-aligned spacing.

        Coordinate convention (kept consistent with _grid_to_lps):
          image_hu.shape == (nz, ny, nx) [DICOM standard: slices, rows, cols]
          spacing        == (z_sp, y_sp, x_sp)
          PyVista grid axis (i,j,k) ↔ image_hu[k, j, i]
            i (x) → col direction → row_dir in LPS
            j (y) → row direction → col_dir in LPS
            k (z) → slice direction → normal_dir in LPS
        """
        nz, ny, nx = image_hu.shape
        sz, sy, sx = spacing
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(sx, sy, sz),
        )
        grid.point_data["values"] = image_hu.flatten(order="C")
        return grid

    def _compute_masked_values(self, image_hu):
        """Run Stage C (morphological opening) on the bone mask and return
        a flat int16 array with non-bone voxels pushed below min_threshold.
        See update_base_mesh comments for the boundary-preservation rationale.
        """
        values = image_hu.astype(np.int16, copy=True).ravel(order="C")
        mask_out = image_hu > self.current_max_threshold

        if self.particle_removal_enabled and self.opening_iterations > 0:
            try:
                bone_mask = (image_hu >= self.current_min_threshold) & ~mask_out
                struct = generate_binary_structure(3, self.opening_connectivity)
                opened_mask = binary_opening(
                    bone_mask, structure=struct, iterations=self.opening_iterations
                )
                removed = bone_mask & ~opened_mask
                mask_out = mask_out | removed
                del bone_mask, opened_mask, removed, struct
            except Exception as e:
                print(f"[Stage C] Morphological opening failed: {e}")

        values[mask_out.ravel(order="C")] = self.current_min_threshold - 1
        return values

    def _apply_smoothing(self, mesh):
        """Apply smoothing per current smooth_combo selection."""
        if mesh is None or mesh.n_points == 0:
            return mesh
        if not hasattr(self, 'smooth_combo'):
            return mesh
        method = self.smooth_combo.currentText()
        if "Laplacian" in method:
            return mesh.smooth(n_iter=100)
        if "Windowed Sinc" in method:
            return mesh.smooth_taubin(n_iter=50, pass_band=0.05)
        return mesh

    def _apply_stage_a(self, mesh):
        """Stage A: remove disconnected mesh fragments by size.
        Returns a PolyData (extract_surface fallback if remove_cells returns UG)."""
        if mesh is None or mesh.n_points == 0 or not self.mesh_cleanup_enabled:
            return mesh
        try:
            if self.keep_largest_only:
                mesh = mesh.extract_largest()
            elif self.min_fragment_faces > 0:
                labeled = mesh.connectivity(extraction_mode='all')
                region_ids_cell = labeled.cell_data.get('RegionId')
                if region_ids_cell is not None:
                    unique, counts = np.unique(region_ids_cell, return_counts=True)
                    keep_regions = unique[counts >= self.min_fragment_faces]
                    if len(keep_regions) == 0:
                        mesh = mesh.extract_largest()
                    elif len(keep_regions) < len(unique):
                        remove_ids = unique[counts < self.min_fragment_faces]
                        cells_to_remove = np.where(
                            np.isin(region_ids_cell, remove_ids)
                        )[0]
                        new_mesh = labeled.remove_cells(cells_to_remove)
                        if not isinstance(new_mesh, pv.PolyData):
                            new_mesh = new_mesh.extract_surface()
                        mesh = new_mesh
            for arr_name in ('RegionId',):
                if arr_name in mesh.point_data:
                    del mesh.point_data[arr_name]
                if arr_name in mesh.cell_data:
                    del mesh.cell_data[arr_name]
        except Exception as e:
            print(f"[Stage A] Mesh fragment cleanup failed: {e}")
        return mesh

    # ===== Bone Separation (Phase 1) =====
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

    def _bounds_to_voxel_slice(self, bounds, image_shape, spacing):
        """Convert a 6-tuple (x0,x1,y0,y1,z0,z1) in grid (mm) to a (k,j,i)
        numpy slice index for an (nz,ny,nx) HU array. Clipped to the volume."""
        nz, ny, nx = image_shape
        sz, sy, sx = spacing
        x0, x1, y0, y1, z0, z1 = bounds
        i0 = int(np.clip(np.floor(x0 / sx), 0, nx))
        i1 = int(np.clip(np.ceil(x1 / sx),  0, nx))
        j0 = int(np.clip(np.floor(y0 / sy), 0, ny))
        j1 = int(np.clip(np.ceil(y1 / sy),  0, ny))
        k0 = int(np.clip(np.floor(z0 / sz), 0, nz))
        k1 = int(np.clip(np.ceil(z1 / sz),  0, nz))
        if i1 <= i0: i1 = min(nx, i0 + 1)
        if j1 <= j0: j1 = min(ny, j0 + 1)
        if k1 <= k0: k1 = min(nz, k0 + 1)
        return (slice(k0, k1), slice(j0, j1), slice(i0, i1))

    def _crop_bounds_in_series_grid(self, series_meta, base_meta):
        """Map the crop box (defined in the base series' grid mm) into an
        axis-aligned bounding box in another series' native grid mm.

        Transforms all 8 corners base_grid→LPS→series_grid and takes the AABB.
        Returns None when geometry is missing or cropping is off.
        """
        if (self.cropping_bounds is None or
                not hasattr(self, 'crop_checkbox') or
                not self.crop_checkbox.isChecked()):
            return None
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_series = self._series_grid_to_lps_matrix(series_meta)
        if T_base is None or T_series is None:
            return None
        try:
            x0, x1, y0, y1, z0, z1 = self.cropping_bounds.bounds
            corners_base = [
                (x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0),
                (x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1),
            ]
            T_series_inv = np.linalg.inv(T_series)
            pts = []
            for cx, cy, cz in corners_base:
                p_base = np.array([cx, cy, cz, 1.0], dtype=float)
                lps = T_base @ p_base
                p_series = T_series_inv @ lps
                pts.append(p_series[:3])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            zs = [p[2] for p in pts]
            return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        except Exception as e:
            print(f"[Separation] crop bounds transform failed: {e}")
            return None

    def _apply_crop_to_bone_mask(self, bone_mask, image_shape, spacing, crop_bounds_mm=None):
        """Zero out voxels outside crop_bounds_mm (series-native grid mm).

        If crop_bounds_mm is None and the crop checkbox is on, uses
        self.cropping_bounds.bounds (valid when the volume IS the base series).
        """
        bounds = crop_bounds_mm
        if bounds is None:
            if (self.cropping_bounds is None or
                    not hasattr(self, 'crop_checkbox') or
                    not self.crop_checkbox.isChecked()):
                return bone_mask
            bounds = self.cropping_bounds.bounds
        try:
            sl = self._bounds_to_voxel_slice(bounds, image_shape, spacing)
            masked = np.zeros_like(bone_mask)
            masked[sl] = bone_mask[sl]
            return masked
        except Exception as e:
            print(f"[Separation] crop ROI failed, using full volume: {e}")
            return bone_mask

    def _compute_separation_labels(self, image_hu, spacing, crop_bounds_mm=None):
        """Run the full separation pipeline.

        Steps:
          1. Bone mask = (min ≤ HU ≤ max), then current Stage C opening.
          2. Restrict to crop_bounds_mm (or active crop box when None),
             otherwise the full volume.
          3. binary_closing(K) + binary_fill_holes — turn each anatomical bone
             into a single solid connected component (fills cancellous holes,
             medullary cavities, surface cracks).
          4. binary_erosion(N) — break thin bridges between adjacent bones
             (vertebral joints, fibula↔tibia, finger joints, etc).
          5. ndi_label — assign integer IDs to each eroded core.
          6. Drop cores below min_bone_voxels.
          7. watershed(-distance, markers=cores, mask=closed_mask) — re-expand
             each core back to the original bone shape, with neighboring labels
             stopping at the ridge between them.

        Returns (label_volume, n_bones, elapsed_seconds).
        label_volume.shape == image_hu.shape, dtype int32 with 0 = background.
        """
        t0 = time.time()
        try:
            bone_mask = (
                (image_hu >= self.current_min_threshold) &
                (image_hu <= self.current_max_threshold)
            )
            if self.particle_removal_enabled and self.opening_iterations > 0:
                struct = generate_binary_structure(3, self.opening_connectivity)
                bone_mask = binary_opening(
                    bone_mask, structure=struct,
                    iterations=self.opening_iterations,
                )

            bone_mask = self._apply_crop_to_bone_mask(
                bone_mask, image_hu.shape, spacing, crop_bounds_mm=crop_bounds_mm
            )

            if not bool(bone_mask.any()):
                return None, 0, time.time() - t0

            struct3 = generate_binary_structure(3, 1)
            if self.hole_fill_iterations > 0:
                bone_mask = binary_closing(
                    bone_mask, structure=struct3,
                    iterations=self.hole_fill_iterations,
                )
            bone_mask = binary_fill_holes(bone_mask)

            if self.separation_erosion > 0:
                cores_mask = binary_erosion(
                    bone_mask, iterations=self.separation_erosion
                )
            else:
                cores_mask = bone_mask.copy()

            cores, _ = ndi_label(cores_mask)
            if cores.max() == 0:
                return None, 0, time.time() - t0

            # Filter cores by voxel count
            if self.min_bone_voxels > 0:
                sizes = np.bincount(cores.ravel())
                keep_ids = np.where(sizes >= self.min_bone_voxels)[0]
                keep_ids = keep_ids[keep_ids != 0]
                if len(keep_ids) == 0:
                    return None, 0, time.time() - t0
                # Re-number 1..k to keep label ids dense
                renum = np.zeros_like(cores)
                for new_id, old_id in enumerate(keep_ids, start=1):
                    renum[cores == old_id] = new_id
                cores = renum
                n_bones = int(len(keep_ids))
            else:
                n_bones = int(cores.max())

            # Watershed expansion
            try:
                distance = distance_transform_edt(bone_mask)
                labels_full = watershed(
                    -distance, markers=cores, mask=bone_mask
                )
            except Exception as e:
                print(f"[Separation] watershed failed, using raw cores: {e}")
                labels_full = cores

            return labels_full.astype(np.int32, copy=False), n_bones, time.time() - t0
        except Exception as e:
            print(f"[Separation] pipeline failed: {e}")
            return None, 0, time.time() - t0

    def _build_separated_bone_meshes(self, label_volume, spacing):
        """For each non-zero label id, run a binary marching cubes and
        return a list of (label_id, mesh, voxel_count). Smoothing is applied
        per the global smooth_combo selection; Stage A is intentionally NOT
        applied here so users still see all fragments per bone."""
        if label_volume is None:
            return []
        nz, ny, nx = label_volume.shape
        sz, sy, sx = spacing
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(sx, sy, sz),
        )
        out = []
        unique_ids = np.unique(label_volume)
        unique_ids = unique_ids[unique_ids > 0]
        for lid in unique_ids:
            sub = (label_volume == lid).astype(np.uint8)
            voxel_count = int(sub.sum())
            grid.point_data["sub"] = sub.flatten(order="C")
            try:
                mesh = grid.contour([0.5], scalars="sub")
            except Exception as e:
                print(f"[Separation] contour failed for label {lid}: {e}")
                continue
            if mesh is None or mesh.n_points == 0:
                continue
            mesh = self._apply_smoothing(mesh)
            if mesh is None or mesh.n_points == 0:
                continue
            out.append((int(lid), mesh, voxel_count))
        if "sub" in grid.point_data:
            del grid.point_data["sub"]
        return out

    def _clear_separated_actors(self):
        """Remove every separated-bone actor from the plotter and reset state.
        Leaves the base single-mesh actor untouched."""
        for bone in self.separated_bones:
            actor = bone.get('actor')
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.separated_bones.clear()

    def _refresh_separation_list(self):
        """Rebuild the dynamic per-bone checkbox list inside the Bone
        Separation panel from self.separated_bones."""
        for cb in self.bone_separation_checkboxes:
            try:
                cb.setParent(None)
                cb.deleteLater()
            except Exception:
                pass
        self.bone_separation_checkboxes.clear()

        if not self.separated_bones:
            return

        for idx, bone in enumerate(self.separated_bones):
            r, g, b = bone['color']
            cb = QCheckBox(f"{bone['name']}  ({bone['voxel_count']:,} vox)")
            cb.setChecked(bool(bone.get('visible', True)))
            # Color the indicator so the user can match it to the 3D view
            cb.setStyleSheet(
                "QCheckBox::indicator:checked { "
                f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)}); "
                "border: 1px solid #444; }"
            )
            cb.stateChanged.connect(
                lambda state, i=idx: self._on_bone_visibility_toggled(i, state)
            )
            self.separation_list_layout.addWidget(cb)
            self.bone_separation_checkboxes.append(cb)

    def _on_bone_visibility_toggled(self, idx, state):
        if idx < 0 or idx >= len(self.separated_bones):
            return
        bone = self.separated_bones[idx]
        visible = (state == Qt.Checked)
        bone['visible'] = visible
        actor = bone.get('actor')
        if actor is not None:
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass
        try:
            self.plotter.render()
        except Exception:
            pass

    def on_hole_fill_changed(self, value):
        self.hole_fill_iterations = int(value)

    def on_sep_erosion_changed(self, value):
        self.separation_erosion = int(value)

    def on_min_bone_vox_changed(self, value):
        self.min_bone_voxels = int(value)

    def _compute_single_separated_bone_entries(self):
        """Run separation on the active single series.

        Returns list of dicts ready for _show_separated_bones, or [] on failure.
        Each dict: name, mesh, voxel_count, series_index (optional).
        """
        if self.volume_grid is None or self.current_image_hu is None:
            return [], 0.0

        crop_bounds = None
        if (hasattr(self, 'crop_checkbox') and self.crop_checkbox.isChecked() and
                self.cropping_bounds is not None):
            crop_bounds = self.cropping_bounds.bounds

        label_vol, n_bones, took = self._compute_separation_labels(
            self.current_image_hu, self.current_spacing, crop_bounds_mm=crop_bounds
        )
        if label_vol is None or n_bones == 0:
            return [], took

        triplets = self._build_separated_bone_meshes(
            label_vol, self.current_spacing
        )
        entries = []
        for lid, mesh, voxel_count in triplets:
            entries.append({
                'name': f"Bone {lid}",
                'mesh': mesh,
                'voxel_count': voxel_count,
                'series_index': None,
                'label_id': lid,
            })
        return entries, took

    def _compute_fusion_separated_bone_entries(self):
        """Run separation on every included fusion series; transform each bone
        mesh into the base series' grid frame (same as _update_fused_meshes).

        Returns (entries, elapsed_seconds).
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

            crop_bounds = None
            if crop_active:
                if i == base_idx:
                    crop_bounds = self.cropping_bounds.bounds
                else:
                    crop_bounds = self._crop_bounds_in_series_grid(meta, base_meta)

            label_vol, n_bones, _ = self._compute_separation_labels(
                image_hu, spacing, crop_bounds_mm=crop_bounds
            )
            if label_vol is None or n_bones == 0:
                continue

            triplets = self._build_separated_bone_meshes(label_vol, spacing)
            desc = str(meta.get('series_description', f'Series {i}'))[:24]
            sub_idx = meta.get('sub_idx', '')
            series_tag = f"[{i}]"
            if sub_idx != '' and sub_idx is not None:
                series_tag += f".{sub_idx}"

            for lid, mesh, voxel_count in triplets:
                if i != base_idx and T_base_inv is not None and T_i is not None:
                    T_composite = T_base_inv @ T_i
                    mesh.transform(T_composite, inplace=True)

                if crop_active and self.cropping_bounds is not None:
                    try:
                        mesh = mesh.clip_box(self.cropping_bounds, invert=False)
                    except Exception:
                        pass
                    if mesh is None or mesh.n_points == 0:
                        continue

                entries.append({
                    'name': f"{series_tag} {desc} — B{lid}",
                    'mesh': mesh,
                    'voxel_count': voxel_count,
                    'series_index': i,
                    'label_id': lid,
                })

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
                'id': entry.get('label_id', 0),
                'mesh': mesh,
                'actor': actor,
                'visible': True,
                'color': color,
                'voxel_count': entry['voxel_count'],
                'name': entry['name'],
                'series_index': entry.get('series_index'),
            })

        self.bone_separation_enabled = True
        self.clear_separation_btn.setEnabled(True)
        mode = "fusion" if self.fusion_enabled else "single"
        self.separation_status_label.setText(
            f"Separated into {len(entries)} bone(s) ({mode}, {took:.1f}s, "
            f"fill={self.hole_fill_iterations}, erode={self.separation_erosion}, "
            f"min={self.min_bone_voxels:,})"
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

    def on_clear_separation_clicked(self):
        """Remove per-bone actors and restore the combined mesh view."""
        self._clear_separated_actors()
        self.bone_separation_enabled = False
        self.clear_separation_btn.setEnabled(False)
        self.separation_status_label.setText("Not separated.")
        self._refresh_separation_list()
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
        if hasattr(self, 'separation_list_layout'):
            self._refresh_separation_list()

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

    def _series_grid_to_lps_matrix(self, meta):
        """Build the 4x4 affine that takes a PyVista grid coord (mm, origin (0,0,0))
        to DICOM patient LPS (mm) for this series.

            LPS = ipp + x·row_dir + y·col_dir + z·normal_dir

        Returns None when DICOM geometry is missing or malformed.
        """
        if not meta:
            return None
        try:
            ipp = np.array(meta['ipp_first'], dtype=float)
            row_dir = np.array(meta['row_dir'], dtype=float)
            col_dir = np.array(meta['col_dir'], dtype=float)
            normal_dir = np.array(meta['normal_dir'], dtype=float)
        except (KeyError, TypeError):
            return None
        if ipp.size != 3 or row_dir.size != 3 or col_dir.size != 3 or normal_dir.size != 3:
            return None
        T = np.eye(4)
        T[:3, 0] = row_dir
        T[:3, 1] = col_dir
        T[:3, 2] = normal_dir
        T[:3, 3] = ipp
        return T

    def update_base_mesh(self):
        """Single entry point for re-meshing. Dispatches to fusion or single path."""
        if self._loading_session:
            return
        self._clear_undo_stack()
        if self.fusion_enabled and len(self.all_series_data) > 0:
            self._update_fused_meshes()
        else:
            self._update_single_mesh()

    def _update_single_mesh(self):
        """Build self.base_mesh from self.volume_grid (single-series mode)."""
        if self.volume_grid is None:
            return

        # Any active bone separation is now stale (mask/threshold/series changed).
        self._auto_clear_separation_on_remesh("parameters changed")

        # Drop any actors left over from fusion mode before single-mesh draws.
        if self.fusion_actors:
            for actor in self.fusion_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.fusion_actors.clear()
            self.fusion_meshes.clear()

        values = self._compute_masked_values(self.current_image_hu)
        self.volume_grid.point_data["masked"] = values

        self.base_mesh = self.volume_grid.contour(
            [self.current_min_threshold], scalars="masked"
        )
        self.base_mesh = self._apply_smoothing(self.base_mesh)
        self.base_mesh = self._apply_stage_a(self.base_mesh)

        self.update_rendered_mesh()

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
        grid coordinate frame using each series' LPS transform.
        """
        self._auto_clear_separation_on_remesh("parameters changed")

        # First call after a patient load: dump per-series geometry once so
        # we can diagnose alignment issues from the console.
        if not getattr(self, '_fusion_diag_printed', False):
            self._print_fusion_diagnostics()
            self._fusion_diag_printed = True

        # Remove all previous fusion actors and reset the rendered actor used by
        # the single-mesh path so we don't end up with duplicates.
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

        # Lazy-init series_volume_grids slot list
        if len(self.series_volume_grids) != len(self.all_series_data):
            self.series_volume_grids = [None] * len(self.all_series_data)

        # Base series transform (target frame). If it lacks DICOM geometry we
        # silently fall back to identity for all series (overlap at origin).
        base_idx = max(0, min(self.base_series_index, len(self.all_series_data) - 1))
        base_meta = self.all_series_data[base_idx]['meta']
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_base_inv = np.linalg.inv(T_base) if T_base is not None else None

        crop_bounds_for_render = self.cropping_bounds
        crop_active = (
            crop_bounds_for_render is not None and self.crop_checkbox.isChecked()
        )

        any_rendered = False
        base_mesh_for_picking = None

        # Defensive length match
        if len(self.fusion_include_flags) != len(self.all_series_data):
            self.fusion_include_flags = [True] * len(self.all_series_data)

        for i, sd in enumerate(self.all_series_data):
            # Skip series the user (or Mako filter) has excluded
            if not self.fusion_include_flags[i]:
                continue

            image_hu = sd['image_hu']
            spacing = sd['spacing']
            meta = sd['meta']

            # 1) Reuse cached ImageData if present (only HU array identity needs to match)
            grid = self.series_volume_grids[i]
            if grid is None:
                grid = self._build_image_data(image_hu, spacing)
                self.series_volume_grids[i] = grid

            # 2) Apply Stage C masking
            grid.point_data["masked"] = self._compute_masked_values(image_hu)

            # 3) Contour
            try:
                mesh = grid.contour(
                    [self.current_min_threshold], scalars="masked"
                )
            except Exception as e:
                print(f"[Fusion] Series {i} contour failed: {e}")
                continue

            if mesh is None or mesh.n_points == 0:
                continue

            # 4) Smoothing + Stage A (per series)
            mesh = self._apply_smoothing(mesh)
            mesh = self._apply_stage_a(mesh)
            if mesh is None or mesh.n_points == 0:
                continue

            # 5) Transform into base series' grid frame
            if i != base_idx and T_base_inv is not None:
                T_i = self._series_grid_to_lps_matrix(meta)
                if T_i is not None:
                    T_composite = T_base_inv @ T_i
                    mesh.transform(T_composite, inplace=True)
                else:
                    print(f"[Fusion] Series {i} missing DICOM geometry; skipped.")
                    continue

            # 6) Optional crop (acts in the base grid frame)
            if crop_active:
                try:
                    mesh = mesh.clip_box(crop_bounds_for_render, invert=False)
                except Exception:
                    pass
                if mesh is None or mesh.n_points == 0:
                    continue

            # 7) Add to scene
            actor = self.plotter.add_mesh(
                mesh,
                color="ivory",
                specular=0.5,
                smooth_shading=True,
            )
            self.fusion_meshes.append((i, mesh))
            self.fusion_actors.append(actor)
            any_rendered = True

            # One-time per-fusion diagnostics: where did this mesh land?
            try:
                b = mesh.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax) in base grid
                desc = self.all_series_data[i].get('meta', {}).get(
                    'series_description', 'N/A'
                )
                print(
                    f"[Fusion] series[{i}] '{desc}' → "
                    f"npts={mesh.n_points}, ncells={mesh.n_cells}, "
                    f"bounds(base grid mm) = "
                    f"X[{b[0]:.1f},{b[1]:.1f}] "
                    f"Y[{b[2]:.1f},{b[3]:.1f}] "
                    f"Z[{b[4]:.1f},{b[5]:.1f}]"
                )
            except Exception:
                pass

            if i == base_idx:
                base_mesh_for_picking = mesh

        # base_mesh is used by landmark picking, hover preview and click-to-remove.
        # In fusion mode we anchor it on the base series so coords stay consistent
        # with the single-series mode contract.
        if base_mesh_for_picking is not None:
            self.base_mesh = base_mesh_for_picking
        elif self.fusion_meshes:
            # Geometry-less fallback: just use the first rendered series
            self.base_mesh = self.fusion_meshes[0][1]

        if not self._camera_initialized and any_rendered:
            self.plotter.reset_camera()
            self._camera_initialized = True

        self.plotter.update()

    def update_rendered_mesh(self):
        # Fusion mode owns the scene actors via _update_fused_meshes;
        # this single-mesh renderer must stay out of its way.
        if self.fusion_enabled:
            return
        if self.base_mesh is None:
            return

        mesh = self.base_mesh
        
        # Apply cropping only when the crop toggle is currently ON.
        # cropping_bounds is preserved across toggle off/on for restoration,
        # but should not actually clip the mesh while crop is disabled.
        if self.cropping_bounds is not None and self.crop_checkbox.isChecked():
            mesh = mesh.clip_box(self.cropping_bounds, invert=False)
        
        # Update Plotter
        if self.current_mesh_actor:
            self.plotter.remove_actor(self.current_mesh_actor)
            
        # Add the new mesh
        self.current_mesh_actor = self.plotter.add_mesh(
            mesh, 
            color="ivory", 
            specular=0.5,
            smooth_shading=True
        )
        
        # Only reset camera if it's the first time rendering this volume
        if not hasattr(self, '_camera_initialized') or not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True
            
        self.plotter.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
