"""Simple mesh-level merge, STL export, undo helper for merge operations.

Methods grouped here:
  - on_merge_bones_clicked: pure mesh union (no gap-fill)
  - on_export_separated_bones_stl: per-bone STL + history report
  - _undo_merge: restore original bones from a __merge__ undo stack entry
    (used by both simple merge and Merge & Fill)
"""

import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
)


class BoneMergeMixin:
    def on_merge_bones_clicked(self):
        """Merge all selected list items into one bone mesh (undo 지원)."""
        if not self.bone_separation_enabled:
            return
        items = self.bone_list_widget.selectedItems()
        if len(items) < 2:
            QMessageBox.information(
                self,
                "Merge Bones",
                "Select at least two bones (Ctrl+click) in the list, then click Merge.",
            )
            return

        uids = [it.data(Qt.UserRole) for it in items]
        bones = [self._bone_by_uid(uid) for uid in uids]
        bones = [b for b in bones if b is not None]
        if len(bones) < 2:
            return

        try:
            merged_mesh = bones[0]['mesh'].copy(deep=True)
            for bone in bones[1:]:
                try:
                    merged_mesh = merged_mesh.merge(bone['mesh'])
                except Exception:
                    merged_mesh = merged_mesh + bone['mesh']
        except Exception as e:
            QMessageBox.critical(self, "Merge Failed", str(e))
            return

        if merged_mesh is None or merged_mesh.n_points == 0:
            QMessageBox.warning(self, "Merge Failed", "Merged mesh is empty.")
            return

        merged_names = [b['name'] for b in bones]
        default_name = " + ".join(merged_names[:3])
        if len(merged_names) > 3:
            default_name += f" (+{len(merged_names) - 3} more)"
        name, ok = QInputDialog.getText(
            self,
            "Merged Bone Name",
            "Name for the merged bone:",
            text=default_name,
        )
        if not ok:
            return
        merged_name = name.strip() or default_name
        total_voxels = sum(int(b.get('voxel_count', 0)) for b in bones)

        # 카메라 시점 저장
        cam_pos = self.plotter.camera_position

        # Undo: 원본 뼈들의 전체 상태를 저장
        # ('__merge__', merged_bone_uid, [original_bone_snapshots])
        original_snapshots = []
        for bone in bones:
            original_snapshots.append({
                'uid': bone['uid'],
                'id': bone.get('id', 0),
                'mesh': bone['mesh'].copy(deep=True),
                'visible': bone.get('visible', True),
                'color': bone.get('color', (1, 1, 1)),
                'voxel_count': bone.get('voxel_count', 0),
                'name': bone['name'],
                'series_index': bone.get('series_index'),
                'raw_mesh': bone['raw_mesh'].copy(deep=True) if bone.get('raw_mesh') is not None else None,
            })

        merged_uid = self._new_bone_uid()

        # Remove old actors and entries
        uid_set = set(b['uid'] for b in bones)
        for bone in list(self.separated_bones):
            if bone.get('uid') in uid_set:
                actor = bone.get('actor')
                if actor is not None:
                    try:
                        self.plotter.remove_actor(actor)
                    except Exception:
                        pass
        self.separated_bones = [
            b for b in self.separated_bones if b.get('uid') not in uid_set
        ]

        color = self._bone_color_palette(1)[0]
        actor = self.plotter.add_mesh(
            merged_mesh,
            color=color,
            specular=0.5,
            smooth_shading=True,
            reset_camera=False,
        )
        self.separated_bones.append({
            'uid': merged_uid,
            'id': 0,
            'mesh': merged_mesh,
            'actor': actor,
            'visible': True,
            'color': color,
            'voxel_count': total_voxels,
            'name': merged_name,
            'series_index': bones[0].get('series_index'),
        })

        # Undo 스택에 merge 항목 추가
        self._restore_undo_stack.append(
            ('__merge__', merged_uid, original_snapshots)
        )
        if len(self._restore_undo_stack) > self._max_restore_undo:
            self._restore_undo_stack = self._restore_undo_stack[-self._max_restore_undo:]
        self.restore_undo_btn.setEnabled(True)

        self._refresh_separation_list()
        # 카메라 시점 복원
        self.plotter.camera_position = cam_pos
        self.separation_status_label.setText(
            f"Merged {len(bones)} bones → \"{merged_name}\" "
            f"({len(self.separated_bones)} total)"
        )
        self.plotter.render()

    def on_export_separated_bones_stl(self):
        """Export each visible separated bone as an individual STL file."""
        if not self.bone_separation_enabled or not self.separated_bones:
            QMessageBox.information(
                self, "Export Bones",
                "Run Separate Bones first."
            )
            return

        visible = [b for b in self.separated_bones if b.get('visible', True)]
        if not visible:
            QMessageBox.information(
                self, "Export Bones", "No visible bones to export."
            )
            return

        base_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Folder for Separated Bones"
        )
        if not base_dir:
            return

        name = "UnknownPatient"
        date_str = "UnknownDate"
        if self.current_meta_info:
            name = self.current_meta_info.get('patient_name', name)
            date_str = self.current_meta_info.get('study_date', date_str)
        name = "".join(c for c in str(name) if c.isalnum() or c in (' ', '-', '_')).strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(
            base_dir,
            f"{name}_{date_str}_{timestamp}_Bones_Export",
        )
        os.makedirs(export_dir, exist_ok=True)

        used_names = set()
        exported = []
        try:
            for bone in visible:
                base = self._sanitize_bone_export_name(bone['name'])
                fname = base
                n = 2
                while fname.lower() in used_names:
                    fname = f"{base}_{n}"
                    n += 1
                used_names.add(fname.lower())
                path = os.path.join(export_dir, f"{fname}.stl")
                bone['mesh'].save(path)
                exported.append((
                    bone['name'], fname, path, bone.get('voxel_count', 0)
                ))

            report_path = os.path.join(export_dir, "bones_export_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("--- Separated Bones STL Export ---\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Count: {len(exported)}\n\n")
                for disp, fname, path, vox in exported:
                    f.write(f"{disp}\n  file: {os.path.basename(path)}\n")
                    f.write(f"  voxels: {vox}\n\n")
                # 작업 이력
                undo_stack = getattr(self, '_restore_undo_stack', [])
                if undo_stack:
                    f.write("--- Operation History (undo stack) ---\n")
                    for j, entry in enumerate(undo_stack):
                        if (isinstance(entry, tuple) and len(entry) == 3
                                and entry[0] == '__merge__'):
                            names = [s['name'] for s in entry[2]]
                            f.write(f"  {j+1}. Merge: {', '.join(names)}\n")
                        else:
                            uid, _ = entry
                            bone = self._bone_by_uid(uid)
                            bname = bone['name'] if bone else uid[:8]
                            f.write(f"  {j+1}. Restore: {bname}\n")
                    f.write("\n")

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(exported)} bone(s) to:\n{export_dir}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _undo_merge(self, merged_uid, original_snapshots, cam_pos):
        """Merge 작업을 되돌림: 합쳐진 뼈를 제거하고 원본 뼈들을 복원."""
        # 합쳐진 뼈 제거
        merged_bone = self._bone_by_uid(merged_uid)
        if merged_bone is not None:
            actor = merged_bone.get('actor')
            if actor is not None:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.separated_bones = [
                b for b in self.separated_bones if b.get('uid') != merged_uid
            ]

        # 원본 뼈들 복원
        for snap in original_snapshots:
            mesh = snap['mesh']
            color = snap.get('color', (1, 1, 1))
            actor = self.plotter.add_mesh(
                mesh, color=color,
                specular=0.5, smooth_shading=True,
                reset_camera=False,
            )
            restored = {
                'uid': snap['uid'],
                'id': snap.get('id', 0),
                'mesh': mesh,
                'actor': actor,
                'visible': snap.get('visible', True),
                'color': color,
                'voxel_count': snap.get('voxel_count', mesh.n_cells),
                'name': snap['name'],
                'series_index': snap.get('series_index'),
            }
            if snap.get('raw_mesh') is not None:
                restored['raw_mesh'] = snap['raw_mesh']
            if not restored['visible']:
                try:
                    actor.SetVisibility(False)
                except Exception:
                    pass
            self.separated_bones.append(restored)

        self._refresh_separation_list()
        self.plotter.camera_position = cam_pos
        self.plotter.render()
        if hasattr(self, 'separation_status_label'):
            names = [s['name'] for s in original_snapshots]
            self.separation_status_label.setText(
                f"Undo merge → {len(names)}개 뼈 복원: {', '.join(names[:3])}"
                + (f" (+{len(names)-3})" if len(names) > 3 else "")
            )
