import os
import traceback

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from app.constants import BASE_DATA_DIR
from dicom_utils import get_pixels_hu, get_series_info, load_scan, load_series

class PatientLoadMixin:
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
    

    def update_thresholds(self):
        self.current_min_threshold = self.min_slider.value()

        if self.volume_grid is not None:
            self.update_base_mesh()


    def on_load_clicked(self):
        patient_id = self.patient_combo.currentText()
        if not patient_id or patient_id == "Data folder not found":
            return
        
        patient_dir = os.path.join(BASE_DATA_DIR, patient_id)
    
        if not os.path.exists(patient_dir):
            return

        # Disable picking modes during a fresh load to avoid stale picking on old mesh
        if self.picking_enabled:
            self.pick_btn.setChecked(False)
        if getattr(self, 'restore_picking_enabled', False):
            self.restore_pick_btn.setChecked(False)
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

        # Drop the old patient's 2D slice indicator + clip cap from the
        # 3D view. The new patient's stack will rebuild them via
        # on_series_switched / update_base_mesh hooks.
        if hasattr(self, '_remove_3d_slice_indicator'):
            self._remove_3d_slice_indicator()
        if hasattr(self, '_remove_clip_cap_mesh'):
            self._remove_clip_cap_mesh()
        # Volume actors are bound to the previous patient's grids; drop
        # them before we wipe self.all_series_data. The next call to
        # update_base_mesh rebuilds them for the new patient.
        for vol_actor in getattr(self, 'volume_actors', []) or []:
            try:
                self.plotter.remove_actor(vol_actor)
            except Exception:
                pass
        self.volume_actors = []
        self.slice_stack = []
        self.slice_unified_idx = 0
        self._last_rendered_series_idx = None

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

