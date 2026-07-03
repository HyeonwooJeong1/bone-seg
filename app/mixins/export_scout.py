import os
from datetime import datetime

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from app.constants import BASE_DATA_DIR

class ExportScoutMixin:
    def open_scout_window(self):
        self.scout_window.show()
        self.scout_window.raise_()
        self.scout_window.activateWindow()


    def open_info_window(self):
        self.info_window.show()
        self.info_window.raise_()
        self.info_window.activateWindow()

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

