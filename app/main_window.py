"""Main application window: state initialization and control panel layout."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSlider,
    QTableWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QListWidget,
)
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pyvistaqt import QtInteractor

from app.mixins import (
    MeshPipelineMixin,
    SessionIoMixin,
    FusionMixin,
    ExportScoutMixin,
    PatientLoadMixin,
    BoneSeparationMixin,
    LandmarksMixin,
    CroppingMixin,
    ParticleRemovalMixin,
    SliceViewerMixin,
    AiSegmentationMixin,
)
from app.ui.collapsible import CollapsibleSection


class MainWindow(
    MeshPipelineMixin,
    SessionIoMixin,
    FusionMixin,
    ExportScoutMixin,
    PatientLoadMixin,
    BoneSeparationMixin,
    LandmarksMixin,
    CroppingMixin,
    ParticleRemovalMixin,
    SliceViewerMixin,
    AiSegmentationMixin,
    QMainWindow,
):
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
        # Volume rendering actors. In fusion mode this is a single
        # vtkMultiVolume (correct depth compositing across series); in
        # single mode a single vtkVolume. self._volume_props holds the
        # shared vtkVolumeProperty objects so W/L opacity updates can
        # reach the multi-volume (whose actor exposes no GetProperty).
        self.volume_actors = []
        self._volume_props = []
    
        self.base_mesh = None
        self.cropping_bounds = None
        self.last_box_bounds = None
        self.scout_arrays = []
        self.current_scout_index = 0
    
        # Bone classification cutoff. Drives the binary volume opacity
        # (value ≥ threshold = solid bone). 50 is a low default that
        # captures soft cancellous bone too; the user can raise it.
        self.current_min_threshold = 50
    
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
        # Stage C+: optional closing to fill internal bone gaps before meshing
        self.closing_enabled = False
        self.closing_iterations = 1
        # Stage D: mesh-level hole filling after contouring.
        # Default ON with a very large size so all topological surface holes
        # (boundary edges from threshold artifacts) are filled. Closed inner
        # cavities like the medullary canal have no boundary loop and are
        # always preserved regardless of this setting.
        self.mesh_fill_holes_enabled = True
        self.mesh_fill_holes_size = 1000000
        # Stage A: mesh-level post-cleanup
        self.mesh_cleanup_enabled = False
        self.keep_largest_only = False
        self.min_fragment_faces = 100
        # Stage B: interactive click-to-remove
        self.picking_enabled = False
        self._manual_undo_stack = []  # snapshots of base_mesh before each manual removal
        self._max_undo = 20

        # ===== Bone Separation State =====
        # When ON, the single base mesh is replaced by N per-bone meshes
        # (one PolyData + actor per connected anatomical bone) so the user
        # can toggle individual bones on/off and inspect them in isolation.
        #
        # Pipeline: build the same pre-smoothing base mesh that update_base_mesh
        # would produce (same threshold / closing / particle-removal applied),
        # then connectivity-split it. Each connected component becomes one
        # bone with smoothing + Stage A re-applied per CC. The separation
        # therefore tracks every voxel-level option from the main controls.
        self.bone_separation_enabled = False
        self.min_bone_voxels = 500               # min mesh cell count per bone
        # Each entry: uid, id, mesh, actor, visible, color, voxel_count, name, series_index
        self.separated_bones = []
        # AI 뼈 분할(학습 모델) 활성 여부
        self.ai_segmentation_active = False
        # AI 우선 모드: 로드 시 자동 AI 분할 + 옛 threshold/분리 UI 숨김
        self.ai_first_mode = True
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, self._enter_ai_first_mode)
        # Phase 2: apply mesh-level Stage A to each bone when separating
        self.separation_apply_stage_a = False
        # Snapshot of the original single-mesh actor's visibility so we can
        # restore it when the user clicks "Clear Separation".
        self._presep_base_actor_was_visible = True
        # Fusion: parallel visibility flags for fusion_actors before separation.
        self._presep_fusion_visibility = []

        # ===== Click-to-Restore State =====
        # Two-step UX: click to select (highlight), Enter to execute.
        self.restore_picking_enabled = False
        self.restore_iterations = 5    # max growth distance (voxels)
        self.restore_method = 'hu_grow'
        self.vote_threshold = 20       # HU offset (threshold - offset = 복원 기준)
        self._restore_undo_stack = []  # list of (bone_uid, old_mesh_copy) or ('__merge__', ...)
        self._max_restore_undo = 20
        self._restore_selected_bone = None   # currently highlighted bone entry
        self._restore_highlight_actor = None  # wireframe highlight actor
        self._list_highlight_actors = []      # 리스트 선택 시 3D 하이라이트
        # Merge & Fill: closing iterations for bridging gaps between bones
        self.merge_fill_iterations = 5

        # ===== Landmark Picking State =====
        self.landmark_picking_enabled = False
        # Each landmark is a dict:
        #   {'name': str,
        #    'grid': np.ndarray(3,)  PyVista grid coords (mm, origin (0,0,0)),
        #    'lps':  np.ndarray(3,) or None  DICOM patient LPS (mm),
        #    'ras':  np.ndarray(3,) or None  DICOM patient RAS (mm)}
        self.landmark_data = []
        self.landmark_actors = []   # parallel list of sphere actors for visualization
        self.landmark_radius_factor = 0.00125  # sphere radius = factor * diag(bounds)
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

        # ===== Hover Preview State =====
        # Single yellow sphere that follows the mouse over the bone surface
        # while landmark picking mode is ON, giving visual feedback before clicking.
        self._hover_sphere_actor = None
        self._hover_picker = None
        self._hover_observer_id = None
        self._hover_last_render = 0.0
        self._hover_min_interval = 1.0 / 60.0  # cap re-render rate to ~60Hz

        # ===== Volume Landmark Picking State =====
        # 3D landmark picking on the volume (no surface mesh) uses a
        # vtkVolumePicker driven by our own click/move observers.
        self._lm_volume_picker = None
        self._lm_cell_picker = None
        self._lm_native_iren = None
        self._lm_obs_ids = []
        self._lm_press_pos = None

        # ===== 2D Slice Viewer State =====
        # Separate top-level QMainWindow built once in setup_controls()
        # via _build_slice_window(). Shows the active series' HU volumes
        # as a UNIFIED z-ordered stack — all included series concatenated
        # by patient z-position so the user scrolls continuously through
        # the whole study with one slider (MicroDicom behaviour). The
        # currently displayed slice is mirrored as a red translucent quad
        # in the 3D plotter at its physical location.
        #
        # Matplotlib FigureCanvas (NOT a second QtInteractor) avoids the
        # WGL OpenGL context conflict on Windows.
        self.slice_window = None
        self.slice_fig = None
        self.slice_canvas = None
        self.slice_ax = None
        self.slice_im = None
        self.slice_slider = None
        # Unified stack: list of (series_idx, local_slice_idx) sorted by
        # the first-slice LPS-z of each series, then by local index.
        self.slice_stack = []
        self.slice_unified_idx = 0
        # Tracks which series the imshow currently displays — used to
        # decide between fast set_data (same series) and full imshow
        # recreate (series transition with different shape/extent).
        self._last_rendered_series_idx = None
        # Red translucent quad in self.plotter showing slice position.
        # Cached vtkPolyData is mutated in-place during drag (no add_mesh
        # per frame) so the 3D camera/zoom is preserved.
        self.slice_3d_actor = None
        self._slice_3d_poly = None
        # Defaults to OFF: indicator + clip plane both add 3D-render
        # cost on every slice change; the user can opt in via checkboxes.
        self.slice_show_3d_indicator = False
        # "Hide above current slice" GPU clip plane on bone mesh mappers.
        # Shared vtkPlane: same reference attached to every bone mapper,
        # so updating its origin propagates to all of them on next render.
        self.slice_hide_above_enabled = False
        self._slice_clip_plane = None
        # Filled cross-section ("cap") rendered at the clip plane so the
        # bones don't look hollow when cut. Single ivory actor that we
        # rebuild on every throttled frame while clipping is active.
        self._slice_cap_actor = None
        # 2D landmark picking — when ON, left-click on the slice canvas
        # places a landmark via the existing on_landmark_picked pipeline
        # (sphere in 3D + row in the landmark table). Independent toggle
        # from the 3D "Pick Landmark" button — both can be ON at once.
        self.slice_landmark_pick_enabled = False
        # Matplotlib artists (markers + name labels) currently drawn on
        # top of the 2D slice. Re-created from self.landmark_data every
        # time the slice or the landmark list changes.
        self._slice_2d_marker_artists = []
        # Distance line / angle arc artists for the currently selected
        # measurement (driven by self.landmark_table selection).
        self._slice_2d_measurement_artists = []
        # Bone preset matches the default 3D rendering target.
        self.window_width = 1500
        self.window_level = 300
        # Mouse drag bookkeeping
        self._slice_drag_mode = None  # None | 'slice' | 'wl'
        self._slice_drag_start = None
        self._slice_drag_start_idx = 0
        self._slice_drag_start_ww = self.window_width
        self._slice_drag_start_wl = self.window_level
        self._slice_last_render = 0.0
        self._slice_min_render_interval = 1.0 / 60.0

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
            "physical position. Click-to-remove is disabled in fusion;\n"
            "cropping, landmarks, and bone separation work in base frame."
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

        self._lbl_smoothing = QLabel("Smoothing:")
        self.control_layout.addWidget(self._lbl_smoothing)
        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems([
            "None (Raw)", "Light", "Strong"
        ])
        # Volume rendering: smoothing = Gaussian blur of the HU field
        # (None→σ0, Light→σ0.8, Strong→σ1.5). Default = None so the raw
        # HU values are used as-is — any smoothing bleeds neighbouring
        # soft tissue (muscle, HU 30-60) above a low cutoff like 50,
        # making non-bone show up as fake bone.
        self.smooth_combo.setCurrentIndex(0)
        self.smooth_combo.currentIndexChanged.connect(self.on_smooth_changed)
        self.control_layout.addWidget(self.smooth_combo)

        # ============================================================
        # 3) Thresholds (always visible — most frequently used)
        # ============================================================
        self.control_layout.addSpacing(8)
        self._lbl_thr1 = QLabel("<b>Bone Threshold</b>")
        self.control_layout.addWidget(self._lbl_thr1)

        self._lbl_thr2 = QLabel("Higher = only dense bone:")
        self.control_layout.addWidget(self._lbl_thr2)
        min_layout = QHBoxLayout()
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(-1000, 3000)
        self.min_slider.setValue(50)
        self.min_spinbox = QSpinBox()
        self.min_spinbox.setRange(-1000, 3000)
        self.min_spinbox.setValue(50)
        self.min_spinbox.setKeyboardTracking(False)
        min_layout.addWidget(self.min_slider)
        min_layout.addWidget(self.min_spinbox)
        self.control_layout.addLayout(min_layout)
        self.min_slider.valueChanged.connect(self.on_min_slider_changed)
        self.min_spinbox.valueChanged.connect(self.on_min_spinbox_changed)


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
        self.separation_section = CollapsibleSection("Bone Editor", expanded=False)

        min_vox_layout = QHBoxLayout()
        min_vox_layout.addWidget(QLabel("Min bone cells:"))
        self.min_bone_vox_spinbox = QSpinBox()
        self.min_bone_vox_spinbox.setRange(0, 1000000)
        self.min_bone_vox_spinbox.setValue(int(self.min_bone_voxels))
        self.min_bone_vox_spinbox.setSingleStep(100)
        self.min_bone_vox_spinbox.setKeyboardTracking(False)
        self.min_bone_vox_spinbox.setToolTip(
            "Mesh connected components with fewer triangle cells than this "
            "are discarded as noise. Bone separation runs as a CC split of "
            "the base mesh — all voxel-level options (closing, particle "
            "removal, threshold) come from the main controls and apply "
            "identically to each separated bone."
        )
        self.min_bone_vox_spinbox.valueChanged.connect(self.on_min_bone_vox_changed)
        min_vox_layout.addWidget(self.min_bone_vox_spinbox)
        self.separation_section.addLayout(min_vox_layout)

        self.sep_stage_a_checkbox = QCheckBox("Stage A cleanup per bone")
        self.sep_stage_a_checkbox.setChecked(self.separation_apply_stage_a)
        self.sep_stage_a_checkbox.setToolTip(
            "When enabled, runs mesh fragment cleanup (Stage A settings from\n"
            "Particle Removal) on each separated bone after contouring."
        )
        self.sep_stage_a_checkbox.stateChanged.connect(self.on_sep_stage_a_toggled)
        self.sep_stage_a_checkbox.setVisible(False)  # 현재 미사용
        self.separation_section.addWidget(self.sep_stage_a_checkbox)

        # ---- Per-bone hole filling (applied inside _build_separated_bone_meshes) ----
        # These run AFTER separation so each bone is processed in isolation —
        # closing and fill_holes are physically bounded by the individual bone
        # mask and cannot merge with neighbouring bones.
        # ---- Fill bone gaps / Fill surface holes (현재 미사용 — 코드 유지) ----
        self.closing_checkbox = QCheckBox("Fill bone gaps (closing)")
        self.closing_checkbox.setChecked(self.closing_enabled)
        self.closing_checkbox.stateChanged.connect(self.on_closing_toggled)
        self.closing_checkbox.setVisible(False)
        self.separation_section.addWidget(self.closing_checkbox)

        closing_iter_layout = QHBoxLayout()
        closing_iter_layout.addWidget(QLabel("  Iterations:"))
        self.closing_iter_spinbox = QSpinBox()
        self.closing_iter_spinbox.setRange(1, 5)
        self.closing_iter_spinbox.setValue(self.closing_iterations)
        self.closing_iter_spinbox.setKeyboardTracking(False)
        self.closing_iter_spinbox.setEnabled(self.closing_enabled)
        self.closing_iter_spinbox.valueChanged.connect(self.on_closing_iter_changed)
        closing_iter_layout.addWidget(self.closing_iter_spinbox)
        self._closing_iter_widget = QWidget()
        self._closing_iter_widget.setLayout(closing_iter_layout)
        self._closing_iter_widget.setVisible(False)
        self.separation_section.addWidget(self._closing_iter_widget)

        self.fill_holes_checkbox = QCheckBox("Fill surface holes (mesh)")
        self.fill_holes_checkbox.setChecked(self.mesh_fill_holes_enabled)
        self.fill_holes_checkbox.stateChanged.connect(self.on_fill_holes_toggled)
        self.fill_holes_checkbox.setVisible(False)
        self.separation_section.addWidget(self.fill_holes_checkbox)

        holes_size_layout = QHBoxLayout()
        holes_size_layout.addWidget(QLabel("  Max hole size:"))
        self.fill_holes_spinbox = QSpinBox()
        self.fill_holes_spinbox.setRange(1, 1000000)
        self.fill_holes_spinbox.setValue(self.mesh_fill_holes_size)
        self.fill_holes_spinbox.setSingleStep(1000)
        self.fill_holes_spinbox.setKeyboardTracking(False)
        self.fill_holes_spinbox.setEnabled(self.mesh_fill_holes_enabled)
        self.fill_holes_spinbox.valueChanged.connect(self.on_fill_holes_size_changed)
        holes_size_layout.addWidget(self.fill_holes_spinbox)
        self._holes_size_widget = QWidget()
        self._holes_size_widget.setLayout(holes_size_layout)
        self._holes_size_widget.setVisible(False)
        self.separation_section.addWidget(self._holes_size_widget)

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

        # ── AI Bone Segmentation (bundled nnU-Net model) ──
        self.separation_section.addWidget(QLabel("<b>AI Bone Segmentation (trained model)</b>"))
        ai_btn_row = QHBoxLayout()
        self.ai_seg_btn = QPushButton("Run AI Segmentation")
        self.ai_seg_btn.setToolTip(
            "Segment the current patient's CT into individual bones with the "
            "bundled nnU-Net model.\n"
            "Uses the GPU when available (first run per patient takes a few "
            "minutes, then it is cached)."
        )
        self.ai_seg_btn.clicked.connect(self.apply_ai_segmentation)
        ai_btn_row.addWidget(self.ai_seg_btn)
        self.ai_clear_btn = QPushButton("Clear AI")
        self.ai_clear_btn.setToolTip("Remove AI bones and restore the threshold rendering.")
        self.ai_clear_btn.clicked.connect(self.clear_ai_segmentation)
        ai_btn_row.addWidget(self.ai_clear_btn)
        self.separation_section.addLayout(ai_btn_row)

        # Phase 2: per-bone list (multi-select for merge, no checkbox interference)
        self.bone_list_widget = QListWidget()
        self.bone_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.bone_list_widget.setEnabled(False)
        self.bone_list_widget.setToolTip(
            "Click to select/deselect (multi-select supported).\n"
            "Clicking a bone in the 3D view selects it here too.\n"
            "Double-click to rename. Press 'H' to hide/show."
        )
        self.bone_list_widget.itemDoubleClicked.connect(self._on_bone_list_item_double_clicked)
        self.bone_list_widget.itemSelectionChanged.connect(self._on_bone_list_selection_changed)
        self.separation_section.addWidget(self.bone_list_widget)

        sep_tools_row = QHBoxLayout()
        self.rename_bone_btn = QPushButton("Rename")
        self.rename_bone_btn.setEnabled(False)
        self.rename_bone_btn.setToolTip("Rename the selected bone in the list.")
        self.rename_bone_btn.clicked.connect(self.on_rename_bone_clicked)
        sep_tools_row.addWidget(self.rename_bone_btn)

        self.toggle_vis_btn = QPushButton("Hide/Show")
        self.toggle_vis_btn.setEnabled(False)
        self.toggle_vis_btn.setToolTip("Show/hide the selected bone(s) (or press 'H').")
        self.toggle_vis_btn.clicked.connect(self.on_toggle_bone_visibility_clicked)
        sep_tools_row.addWidget(self.toggle_vis_btn)

        self.merge_bones_btn = QPushButton("Merge Selected")
        self.merge_bones_btn.setEnabled(False)
        self.merge_bones_btn.setToolTip(
            "Merge two or more selected bones into one mesh (Ctrl+click to multi-select)."
        )
        self.merge_bones_btn.clicked.connect(self.on_merge_bones_clicked)
        sep_tools_row.addWidget(self.merge_bones_btn)
        self.separation_section.addLayout(sep_tools_row)

        self.export_bones_stl_btn = QPushButton("Export Bones STL…")
        self.export_bones_stl_btn.setEnabled(False)
        self.export_bones_stl_btn.setToolTip(
            "Export each visible separated bone as its own STL file into a folder."
        )
        self.export_bones_stl_btn.clicked.connect(self.on_export_separated_bones_stl)
        self.separation_section.addWidget(self.export_bones_stl_btn)

        # ---- Click-to-Restore (per-bone hole filling) ----
        self.separation_section.addWidget(QLabel("<i>── Click-to-Restore ──</i>"))

        restore_btn_row = QHBoxLayout()
        self.restore_pick_btn = QPushButton("Restore Mode: OFF")
        self.restore_pick_btn.setCheckable(True)
        self.restore_pick_btn.setEnabled(False)
        self.restore_pick_btn.setToolTip(
            "When on, click a bone to fill holes in it.\n"
            "Each click applies closing cumulatively.\n"
            "Use Undo to revert."
        )
        self.restore_pick_btn.toggled.connect(self.on_restore_picking_toggled)
        restore_btn_row.addWidget(self.restore_pick_btn)

        self.restore_undo_btn = QPushButton("Undo")
        self.restore_undo_btn.setEnabled(False)
        self.restore_undo_btn.clicked.connect(self.on_restore_undo_clicked)
        restore_btn_row.addWidget(self.restore_undo_btn)
        self.separation_section.addLayout(restore_btn_row)

        # ── HU Region Growing 복원 알고리즘 ──
        # 현재 뼈(threshold 기준)에서 출발, 인접 voxel 중 HU >= (threshold - offset)인 것만 추가.
        # 원본 CT 데이터의 HU 값이 자연 경계 → 뼈 바깥(공기)은 절대 안 채워짐.

        restore_dist_row = QHBoxLayout()
        restore_dist_row.addWidget(QLabel("  Max growth:"))
        self.restore_iter_spinbox = QSpinBox()
        self.restore_iter_spinbox.setRange(1, 50)
        self.restore_iter_spinbox.setValue(self.restore_iterations)
        self.restore_iter_spinbox.setKeyboardTracking(False)
        self.restore_iter_spinbox.setSuffix(" vox")
        self.restore_iter_spinbox.setToolTip(
            "Maximum growth distance (in voxels).\n"
            "  Small (3-5): restore only near the surface\n"
            "  Medium (10): default, handles most gaps\n"
            "  Large (20-50): fills big gaps (can be slow)"
        )
        self.restore_iter_spinbox.valueChanged.connect(
            lambda v: setattr(self, 'restore_iterations', int(v))
        )
        restore_dist_row.addWidget(self.restore_iter_spinbox)
        self.separation_section.addLayout(restore_dist_row)

        hu_offset_row = QHBoxLayout()
        hu_offset_row.addWidget(QLabel("  HU offset:"))
        self.vote_threshold_spinbox = QSpinBox()
        self.vote_threshold_spinbox.setRange(0, 500)
        self.vote_threshold_spinbox.setValue(self.vote_threshold)
        self.vote_threshold_spinbox.setKeyboardTracking(False)
        self.vote_threshold_spinbox.setSuffix(" HU")
        self.vote_threshold_spinbox.setToolTip(
            "Restore cutoff = (current threshold) − (this value).\n"
            "Includes voxels with HU below the current threshold that may still "
            "be bone.\n\n"
            "  0 = same as current threshold (no extra restore)\n"
            "  50 = default (partial-volume correction, safe)\n"
            "  100-200 = aggressive (osteoporotic / thin bone)\n"
            "  300+ = very aggressive (may include soft tissue)\n\n"
            "e.g. threshold=200, offset=50 → restore voxels with HU ≥ 150"
        )
        self.vote_threshold_spinbox.valueChanged.connect(
            lambda v: setattr(self, 'vote_threshold', int(v))
        )
        hu_offset_row.addWidget(self.vote_threshold_spinbox)
        self.separation_section.addLayout(hu_offset_row)

        # ---- Merge & Fill Gap ----
        self.separation_section.addWidget(QLabel("<i>── Merge &amp; Fill ──</i>"))

        self.merge_fill_btn = QPushButton("Merge && Fill Gap")
        self.merge_fill_btn.setEnabled(False)
        self.merge_fill_btn.setToolTip(
            "Select 2+ bones in the list (Ctrl+click), then click.\n"
            "Merges them into one and smoothly fills the space between.\n"
            "Fills gaps with voxel-level closing, then re-meshes."
        )
        self.merge_fill_btn.clicked.connect(self.on_merge_fill_clicked)
        self.separation_section.addWidget(self.merge_fill_btn)

        merge_iter_row = QHBoxLayout()
        merge_iter_row.addWidget(QLabel("  Fill iterations:"))
        self.merge_fill_iter_spinbox = QSpinBox()
        self.merge_fill_iter_spinbox.setRange(1, 20)
        self.merge_fill_iter_spinbox.setValue(self.merge_fill_iterations)
        self.merge_fill_iter_spinbox.setKeyboardTracking(False)
        self.merge_fill_iter_spinbox.setToolTip(
            "Closing iterations used to fill the space between bones.\n"
            "  3 = small gaps\n"
            "  5 = medium gaps (default)\n"
            "  10+ = large gaps"
        )
        self.merge_fill_iter_spinbox.valueChanged.connect(
            lambda v: setattr(self, 'merge_fill_iterations', int(v))
        )
        merge_iter_row.addWidget(self.merge_fill_iter_spinbox)
        self.separation_section.addLayout(merge_iter_row)

        self.control_layout.addWidget(self.separation_section)

        # ============================================================
        # 5) Particle Removal (collapsible — per user request)
        # ============================================================
        self.particle_section = CollapsibleSection("Particle Removal", expanded=False)

        # Stage C — morphological opening (erode N × dilate N) on the bone mask
        self.voxel_cleanup_checkbox = QCheckBox("CC particle removal (voxel-level)")
        self.voxel_cleanup_checkbox.setChecked(self.particle_removal_enabled)
        self.voxel_cleanup_checkbox.setToolTip(
            "Remove isolated voxel clusters from the bone mask BEFORE meshing.\n"
            "Morphological opening (erode → dilate) removes small noise particles.\n"
            "Keep iterations small to preserve thin bones."
        )
        self.voxel_cleanup_checkbox.stateChanged.connect(self.on_voxel_cleanup_toggled)
        self.particle_section.addWidget(self.voxel_cleanup_checkbox)

        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("  Iterations:"))
        self.opening_iter_spinbox = QSpinBox()
        self.opening_iter_spinbox.setRange(0, 3)
        self.opening_iter_spinbox.setValue(int(self.opening_iterations))
        self.opening_iter_spinbox.setSingleStep(1)
        self.opening_iter_spinbox.setKeyboardTracking(False)
        self.opening_iter_spinbox.setToolTip(
            "Opening iterations (erode → dilate kernel size).\n"
            "  0 = off\n"
            "  1 = remove 1-voxel spurs/particles (recommended)\n"
            "  2 = remove clusters ≤ 2 voxels (careful with thin bone)\n"
            "Higher removes more but can damage thin bones."
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
            "Adjacency used when labeling connected components:\n"
            "6  – face neighbors only   (conservative, splits more components)\n"
            "18 – + edge neighbors\n"
            "26 – + corner neighbors    (permissive, merges thin diagonal links)"
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

        self.particle_section.setVisible(False)  # 현재 미사용 — 코드는 유지
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

        # Key events for Restore mode (Enter to confirm, Escape to cancel)
        self.plotter.add_key_event('Return', self._on_restore_confirm)
        self.plotter.add_key_event('Escape', self._on_restore_cancel)
        self.plotter.add_key_event('h', self.on_toggle_bone_visibility_clicked)

        self.control_layout.addStretch()

        bottom_row = QHBoxLayout()
        self.scout_btn = QPushButton("Open Scout Viewer")
        self.scout_btn.clicked.connect(self.open_scout_window)
        bottom_row.addWidget(self.scout_btn)

        self.info_btn = QPushButton("Open Patient Info")
        self.info_btn.clicked.connect(self.open_info_window)
        bottom_row.addWidget(self.info_btn)
        self.control_layout.addLayout(bottom_row)

        slice_row = QHBoxLayout()
        self.slice_viewer_btn = QPushButton("Open 2D Viewer")
        self.slice_viewer_btn.setToolTip(
            "Axial slice viewer of the active series.\n"
            "Left-drag: scroll slices · Right-drag: window/level · Wheel: ±1 slice"
        )
        self.slice_viewer_btn.clicked.connect(self.open_slice_viewer)
        slice_row.addWidget(self.slice_viewer_btn)
        self.control_layout.addLayout(slice_row)

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

        # 2D Slice Viewer window (axial). Built once, shown on demand.
        self._build_slice_window()

