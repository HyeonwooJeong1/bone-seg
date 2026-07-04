"""
ai_segmentation.py — AI 뼈 분할(통합 nnU-Net 모델) 결과를 앱에 통합하는 mixin.

기존 threshold/volume 파이프라인은 그대로 두고, "AI 뼈 분할" 모드를 추가한다.
  1) infer_app.py(subprocess, 오프라인·GPU/CPU 자동)로 환자 CT → 라벨 npz 생성
     (한 번 만든 뒤 <환자>_ai_labels.npz로 캐시 → 다음엔 재추론 없이 로드)
  2) 현재 시리즈(스테이션)에 매칭되는 블록 라벨을 찾아
  3) 뼈마다 marching cubes → 약한 taubin → 의미론적 색·이름으로 표시
     (기존 separated_bones 구조 재사용 → 랜드마크·목록·세션 등 기존 도구 그대로 동작)

라벨은 이미 좌우 통합(R)·후처리 완료 상태로 npz에 담겨 옴.
"""
import os
import json
import subprocess
import sys
import time

import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication, QProgressDialog
from PyQt5.QtCore import Qt


# 뼈 id(통합 21라벨) → 의미론적 색 (L/R 통합이라 R id만 등장하지만 둘 다 정의)
_AI_COLORS = {
    1: (0.85, 0.75, 0.55), 2: (0.85, 0.75, 0.55),   # Femur — tan
    3: (0.90, 0.60, 0.25), 4: (0.90, 0.60, 0.25),   # Hip — orange
    5: (0.95, 0.85, 0.35),                          # Sacrum — yellow
    6: (0.85, 0.20, 0.20), 7: (0.85, 0.20, 0.20),   # Patella — red
    8: (0.55, 0.35, 0.75), 9: (0.55, 0.35, 0.75),   # Tibia — purple
    10: (0.55, 0.35, 0.20), 11: (0.55, 0.35, 0.20), # Fibula — brown
    12: (0.30, 0.65, 0.80), 13: (0.30, 0.65, 0.80), # Talus — cyan
    14: (0.90, 0.45, 0.70), 15: (0.90, 0.45, 0.70), # Calcaneus — pink
    16: (0.60, 0.80, 0.35), 17: (0.60, 0.80, 0.35), # Tarsals — green
    18: (0.40, 0.55, 0.85), 19: (0.40, 0.55, 0.85), # Metatarsals — blue
    20: (0.80, 0.55, 0.30), 21: (0.80, 0.55, 0.30), # Phalanges — amber
}


class AiSegmentationMixin:

    # ── AI 우선 모드: 옛 뼈구분 UI 숨김 ─────────────────────────
    def _enter_ai_first_mode(self):
        """HU threshold·smoothing·particle 등 옛 뼈구분 UI를 숨긴다(코드는 보존).

        AI가 기본 뼈 소스이므로 threshold 슬라이더·smoothing·particle 제거,
        그리고 Bone Editor 안의 기하학적 분리(Separate Bones)·closing·fill holes도 숨김.
        랜드마크·뼈목록·크롭·세션·시리즈 선택 등은 유지.
        """
        legacy = [
            "_lbl_thr1", "_lbl_thr2", "min_slider", "min_spinbox",
            "_lbl_smoothing", "smooth_combo", "particle_section",
            "separate_btn", "clear_separation_btn",
            "sep_stage_a_checkbox", "closing_checkbox", "_closing_iter_widget",
            "fill_holes_checkbox", "_holes_size_widget", "min_bone_vox_spinbox",
            # 시리즈/fusion 컨트롤 — AI가 전체 하지를 한 번에 표시하므로 불필요·충돌 방지
            "fusion_checkbox", "series_combo", "series_include_section",
            "mako_only_checkbox",
        ]
        for attr in legacy:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setVisible(False)
                except Exception:
                    pass

    # ── 경로/상태 ────────────────────────────────────────────────
    def _ai_npz_path(self):
        """현재 환자의 AI 라벨 캐시 npz 경로."""
        pid = self.patient_combo.currentText() if hasattr(self, "patient_combo") else None
        if not pid or pid == "Data folder not found":
            return None
        from app.constants import BASE_DATA_DIR
        return os.path.join(BASE_DATA_DIR, pid, f"{pid}_ai_labels.npz")

    def _ai_infer_script(self):
        """번들된 infer_app.py 경로 (repo/ai_bone/infer_app.py)."""
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(here, "ai_bone", "infer_app.py")

    # ── ① 추론 실행 (없으면 생성) ────────────────────────────────
    def run_ai_inference(self, folds=None, force=False, silent=False, cache_only=False):
        """infer_app.py를 subprocess로 실행해 <환자>_ai_labels.npz 생성.

        folds=None → GPU면 5-fold, CPU면 단일 fold 자동. force=True면 캐시 무시.
        silent=True → 팝업 없이 실패 시 조용히 None 반환(자동 로드용).
        cache_only=True → 캐시가 있으면 그 경로, 없으면 추론하지 않고 None(자동 로드용).
        """
        if getattr(self, "_ai_busy", False):
            return None       # 이미 추론 중 → 중복 실행 방지
        npz = self._ai_npz_path()
        if npz is None:
            if not silent:
                QMessageBox.warning(self, "AI Segmentation", "Load a patient first.")
            return None
        if os.path.exists(npz) and not force:
            return npz  # 캐시 사용
        if cache_only:
            return None  # 캐시 없음 → 자동모드에선 추론 생략

        pid = self.patient_combo.currentText()
        from app.constants import BASE_DATA_DIR
        dicom_dir = os.path.join(BASE_DATA_DIR, pid)
        script = self._ai_infer_script()
        if not os.path.exists(script):
            if not silent:
                QMessageBox.critical(self, "AI Segmentation",
                                     f"Inference script not found:\n{script}")
            return None

        # 추론 런타임 확인 (torch/nnunetv2). 없으면 안내하고 중단(캐시 환자는 계속 동작).
        try:
            import importlib.util
            has_nnunet = importlib.util.find_spec("nnunetv2") is not None
        except Exception:
            has_nnunet = False
        if not has_nnunet:
            if not silent:
                QMessageBox.information(
                    self, "AI runtime required",
                    "This environment has no AI inference runtime (torch/nnU-Net).\n"
                    "Run new-CT segmentation in the environment created by "
                    "setup_and_run.bat.\n"
                    "(Already-segmented patients still load instantly from cache.)")
            return None

        # device·folds 자동
        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        if folds is None:
            folds = "0"   # 단일 fold = 빠름(정확도 충분). 더 높이려면 "0,1,2" 등.

        cmd = [sys.executable, script, dicom_dir, npz,
               "--folds", str(folds), "--device", device]
        dlg = QProgressDialog(
            f"Running AI bone segmentation… (device={device})\n"
            f"The first run for a patient may take a few minutes.",
            None, 0, 0, self)
        dlg.setWindowTitle("AI Segmentation")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.show()
        QApplication.processEvents()
        self._ai_busy = True
        # 출력은 로그 파일로 보냄 — PIPE를 안 읽으면 버퍼가 차서 추론이 멈추므로(중요!)
        logpath = npz + ".log"
        try:
            with open(logpath, "w", encoding="utf-8", errors="replace") as lf:
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
                while proc.poll() is None:      # GUI 응답 유지하며 대기
                    QApplication.processEvents()
                    time.sleep(0.1)
        finally:
            self._ai_busy = False
            dlg.close()
        if proc.returncode != 0 or not os.path.exists(npz):
            err = ""
            try:
                with open(logpath, encoding="utf-8", errors="replace") as lf:
                    err = lf.read()[-1500:]
            except Exception:
                pass
            if not silent:
                QMessageBox.critical(
                    self, "AI Segmentation failed",
                    f"Inference failed (code {proc.returncode}).\n\n{err}")
            self.statusBar().showMessage("AI inference failed", 5000)
            return None
        self.statusBar().showMessage("AI inference complete", 4000)
        return npz

    # ── ② npz 로드 + 시리즈 매칭 ─────────────────────────────────
    def _load_ai_blocks(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        n = int(d["n_blocks"])
        id2name = json.loads(str(d["id2name"]))
        blocks = []
        for b in range(n):
            entry = {
                "label": d[f"block{b}_label"],           # (nz,ny,nx) uint8
                "zrange": tuple(float(x) for x in d[f"block{b}_zrange"]),
                "shape": tuple(int(x) for x in d[f"block{b}_shape"]),
                "spacing": None,
            }
            key = f"block{b}_spacing"
            if key in d.files:
                entry["spacing"] = tuple(float(x) for x in d[key])   # (z,y,x)
            blocks.append(entry)
        return blocks, id2name

    def _match_ai_block(self, blocks):
        """현재 시리즈에 맞는 블록 선택: shape 정확 일치 우선, 없으면 z-범위 겹침."""
        if self.current_image_hu is None:
            return None
        shp = tuple(self.current_image_hu.shape)
        for blk in blocks:
            if blk["shape"] == shp:
                return blk
        # z-범위 겹침 폴백 (world z: 시리즈 meta에서 z_min/z_max)
        meta = {}
        try:
            meta = self.all_series_data[self.series_combo.currentIndex()].get("meta", {})
        except Exception:
            pass
        zmin = meta.get("z_min"); zmax = meta.get("z_max")
        if zmin is not None and zmax is not None:
            best, ov = None, 0.0
            for blk in blocks:
                lo, hi = blk["zrange"]
                o = max(0.0, min(hi, zmax) - max(lo, zmin))
                if o > ov:
                    best, ov = blk, o
            if best is not None:
                return best
        return None

    def _ai_base_transform(self, series_idx):
        """4x4 transform: series-idx grid(mm) → base series grid(mm), or None.

        Uses the same LPS basis the slice indicator / landmarks / fused volume
        use, so AI bones land in the app's coordinate frame. None if geometry
        is missing (caller then falls back to physical-z placement).
        """
        if series_idx is None:
            return None
        try:
            base_idx = int(getattr(self, "base_series_index", 0))
            base_meta = self.all_series_data[base_idx].get("meta", {}) or {}
            i_meta = self.all_series_data[series_idx].get("meta", {}) or {}
            T_base = self._series_grid_to_lps_matrix(base_meta)
            T_i = self._series_grid_to_lps_matrix(i_meta)
            if T_base is None or T_i is None:
                return None
            return np.linalg.inv(T_base) @ T_i
        except Exception:
            return None

    # ── ③ 라벨 → 뼈별 의미론적 메시 표시 ─────────────────────────
    def apply_ai_segmentation(self, auto=False):
        """현재 시리즈에 AI 라벨을 적용 — 뼈마다 색·이름 메시로 표시.

        auto=True(로드시 자동): 팝업 없이, 실패하면 조용히 기존 렌더 유지.
        """
        # 캐시 있으면 즉시, 없으면 추론 실행(자동 로드·버튼 모두). 실패 시 안내.
        npz = self.run_ai_inference(silent=False, cache_only=False)
        if npz is None:
            return
        try:
            blocks, id2name = self._load_ai_blocks(npz)
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "AI Segmentation",
                                     f"Failed to load labels:\n{e}")
            return
        if not blocks:
            return

        # 기존 뼈/메시 actor 정리 + volume 숨김
        self._clear_separated_actors()
        self.separated_bones = []
        self._hide_volume_for_ai()

        # 각 블록을 로드된 시리즈(같은 z-gap 분할)에 매칭 → 그 시리즈 grid에서
        # 메쉬 생성 후 앱 기준 프레임(base grid, LPS 변환)으로 옮김 → 2D 슬라이스
        # 뷰어·랜드마크·볼륨과 좌표 정합. 매칭/지오메트리 실패 시 물리z 폴백.
        import pyvista as pv
        for blk in blocks:
            label = blk["label"]                              # (nz,ny,nx)
            series_idx = None
            for si, sd in enumerate(getattr(self, "all_series_data", []) or []):
                ih = sd.get("image_hu")
                if ih is not None and tuple(ih.shape) == tuple(label.shape):
                    series_idx = si
                    break
            T = self._ai_base_transform(series_idx) if series_idx is not None else None
            if T is not None:
                sp = self.all_series_data[series_idx]["spacing"]   # (z,y,x)
                origin = (0.0, 0.0, 0.0)
            else:
                sp = blk.get("spacing") or self.current_spacing
                origin = (0.0, 0.0, float(blk["zrange"][0]))
            nz, ny, nx = label.shape
            for cid in [int(x) for x in np.unique(label) if x > 0]:
                mask = (label == cid).astype(np.float32)
                if mask.sum() < 50:
                    continue
                grid = pv.ImageData(
                    dimensions=(nx, ny, nz),
                    spacing=(float(sp[2]), float(sp[1]), float(sp[0])),
                    origin=origin,
                )
                grid.point_data["values"] = mask.flatten(order="C")
                try:
                    s = grid.contour([0.5], scalars="values")
                except Exception:
                    continue
                if s is None or s.n_points == 0:
                    continue
                if T is not None:
                    try:
                        ph = np.hstack([s.points, np.ones((s.n_points, 1))])
                        s.points = (T @ ph.T).T[:, :3]
                    except Exception:
                        pass
                try:
                    # Taubin(부피 보존) 표면 스무딩.
                    s = s.smooth_taubin(n_iter=20, pass_band=0.1)
                except Exception:
                    pass
                color = _AI_COLORS.get(cid, (0.8, 0.8, 0.8))
                actor = self.plotter.add_mesh(
                    s, color=color, specular=0.3, smooth_shading=True)
                name = id2name.get(str(cid), f"Bone {cid}")
                self.separated_bones.append({
                    "uid": self._new_bone_uid(),
                    "id": cid,
                    "mesh": s,
                    "raw_mesh": s.copy(deep=True),
                    "actor": actor,
                    "visible": True,
                    "color": color,
                    "voxel_count": int(mask.sum()),
                    "name": name,
                    "series_index": None,
                })

        if not self.separated_bones:
            QMessageBox.warning(self, "AI Segmentation", "No bones to display.")
            self._restore_volume_after_ai()
            return

        self.ai_segmentation_active = True
        self.bone_separation_enabled = True
        if hasattr(self, "clear_separation_btn"):
            self.clear_separation_btn.setEnabled(True)
        self._set_separation_tools_enabled(True)
        if hasattr(self, "separation_status_label"):
            self.separation_status_label.setText(
                f"AI: {len(self.separated_bones)} bone(s)")
        self._refresh_separation_list()
        # Enable click-to-select in the 3D view (click a bone -> selected in list)
        if hasattr(self, "_enable_bone_click_selection"):
            self._enable_bone_click_selection()
        self._update_info_panel()
        try:
            if not getattr(self, "_camera_initialized", False):
                self.plotter.reset_camera()
                self._camera_initialized = True
            self.plotter.update()
        except Exception:
            pass

    # ── volume 숨김/복원 ────────────────────────────────────────
    def _hide_volume_for_ai(self):
        for actor in getattr(self, "volume_actors", []) or []:
            try:
                actor.SetVisibility(False)
            except Exception:
                pass

    def _restore_volume_after_ai(self):
        for actor in getattr(self, "volume_actors", []) or []:
            try:
                actor.SetVisibility(True)
            except Exception:
                pass

    def clear_ai_segmentation(self):
        """Remove AI bones and restore the threshold volume rendering."""
        if hasattr(self, "_disable_bone_click_selection"):
            self._disable_bone_click_selection()
        self._clear_separated_actors()
        self.separated_bones = []
        self.ai_segmentation_active = False
        self._restore_volume_after_ai()
        if hasattr(self, "separation_status_label"):
            self.separation_status_label.setText("")
        if hasattr(self, "_refresh_separation_list"):
            self._refresh_separation_list()
        self._update_info_panel()
        try:
            self.plotter.update()
        except Exception:
            pass

    # ── Selection Info panel + landmark memo ────────────────────
    def _update_info_panel(self):
        """Refresh the Selection Info panel from the current selection.

        Landmark selection takes priority; otherwise show bone selection;
        otherwise a short summary of the current state.
        """
        if not hasattr(self, "info_label"):
            return
        # Landmark rows selected?
        lm_rows = []
        if hasattr(self, "landmark_table") and self.landmark_table is not None:
            sm = self.landmark_table.selectionModel()
            if sm is not None:
                lm_rows = sorted({i.row() for i in sm.selectedRows()
                                  if 0 <= i.row() < len(self.landmark_data)})
        if lm_rows:
            self._info_show_landmarks(lm_rows)
            return
        bone_items = (self.bone_list_widget.selectedItems()
                      if hasattr(self, "bone_list_widget") else [])
        if bone_items:
            self._info_show_bones(bone_items)
            return
        self._info_show_summary()

    def _info_disable_memo(self):
        self._memo_row = None
        if hasattr(self, "memo_edit"):
            self.memo_edit.blockSignals(True)
            self.memo_edit.setPlainText("")
            self.memo_edit.blockSignals(False)
            self.memo_edit.setEnabled(False)

    def _info_show_landmarks(self, rows):
        parts = []
        for r in rows:
            e = self.landmark_data[r]
            g = e.get("grid")
            gtxt = ", ".join(f"{float(v):.1f}" for v in g) if g is not None else "-"
            parts.append(f"<b>{e.get('name','')}</b> &nbsp; grid=({gtxt})")
        self.info_label.setText("Landmark(s) selected:<br>" + "<br>".join(parts))
        if len(rows) == 1 and hasattr(self, "memo_edit"):
            self._memo_row = rows[0]
            self.memo_edit.blockSignals(True)
            self.memo_edit.setPlainText(str(self.landmark_data[rows[0]].get("memo", "")))
            self.memo_edit.blockSignals(False)
            self.memo_edit.setEnabled(True)
        else:
            self._info_disable_memo()

    def _info_show_bones(self, items):
        self._info_disable_memo()
        parts = []
        for it in items:
            b = self._bone_by_uid(it.data(Qt.UserRole))
            if b is None:
                continue
            vis = "shown" if b.get("visible", True) else "hidden"
            parts.append(f"<b>{b.get('name','')}</b> &nbsp; "
                         f"{b.get('voxel_count', 0):,} vox &nbsp; ({vis})")
        self.info_label.setText("Bone(s) selected:<br>" + "<br>".join(parts))

    def _info_show_summary(self):
        self._info_disable_memo()
        nb = len(getattr(self, "separated_bones", []))
        nl = len(getattr(self, "landmark_data", []))
        mode = "AI segmentation" if getattr(self, "ai_segmentation_active", False) else "Threshold"
        self.info_label.setText(
            f"<i>Nothing selected.</i><br>Mode: <b>{mode}</b><br>"
            f"Bones: {nb} &nbsp;·&nbsp; Landmarks: {nl}")

    def _on_memo_changed(self):
        if getattr(self, "_memo_row", None) is None:
            return
        if 0 <= self._memo_row < len(self.landmark_data):
            self.landmark_data[self._memo_row]["memo"] = self.memo_edit.toPlainText()
