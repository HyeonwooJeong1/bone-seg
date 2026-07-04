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

import numpy as np
from PyQt5.QtWidgets import QMessageBox, QApplication
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
        npz = self._ai_npz_path()
        if npz is None:
            if not silent:
                QMessageBox.warning(self, "AI 분할", "먼저 환자를 선택/로드하세요.")
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
                QMessageBox.critical(self, "AI 분할", f"추론 스크립트 없음:\n{script}")
            return None

        # device·folds 자동
        device = "cpu"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            pass
        if folds is None:
            folds = "0,1,2,3,4" if device == "cuda" else "0"

        cmd = [sys.executable, script, dicom_dir, npz,
               "--folds", str(folds), "--device", device]
        self.statusBar().showMessage(
            f"AI 추론 중… (device={device}, folds={folds}) — 수 분 소요", 0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            QApplication.restoreOverrideCursor()
        if proc.returncode != 0 or not os.path.exists(npz):
            if not silent:
                QMessageBox.critical(
                    self, "AI 분할 실패",
                    f"추론 실패 (code {proc.returncode}).\n\n{proc.stderr[-1500:]}")
            self.statusBar().showMessage("AI 추론 실패", 5000)
            return None
        self.statusBar().showMessage("AI 추론 완료", 4000)
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

    # ── ③ 라벨 → 뼈별 의미론적 메시 표시 ─────────────────────────
    def apply_ai_segmentation(self, auto=False):
        """현재 시리즈에 AI 라벨을 적용 — 뼈마다 색·이름 메시로 표시.

        auto=True(로드시 자동): 팝업 없이, 실패하면 조용히 기존 렌더 유지.
        """
        # 자동(로드시)엔 캐시 있을 때만; 버튼 클릭(auto=False)이면 없으면 추론 실행
        npz = self.run_ai_inference(silent=auto, cache_only=auto)
        if npz is None:
            return
        try:
            blocks, id2name = self._load_ai_blocks(npz)
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "AI 분할", f"라벨 로드 실패:\n{e}")
            return
        if not blocks:
            return

        # 기존 뼈/메시 actor 정리 + volume 숨김
        self._clear_separated_actors()
        self.separated_bones = []
        self._hide_volume_for_ai()

        # 모든 스테이션(블록)을 물리 z 위치에 배치 → 전체 하지 한 번에 표시.
        # (블록마다 z-spacing이 다르므로 저장된 spacing 사용, 없으면 현재 시리즈 것)
        import pyvista as pv
        for blk in blocks:
            label = blk["label"]                              # (nz,ny,nx)
            sp = blk.get("spacing") or self.current_spacing   # (z,y,x)
            oz = float(blk["zrange"][0])
            nz, ny, nx = label.shape
            for cid in [int(x) for x in np.unique(label) if x > 0]:
                mask = (label == cid).astype(np.float32)
                if mask.sum() < 50:
                    continue
                grid = pv.ImageData(
                    dimensions=(nx, ny, nz),
                    spacing=(float(sp[2]), float(sp[1]), float(sp[0])),
                    origin=(0.0, 0.0, oz),
                )
                grid.point_data["values"] = mask.flatten(order="C")
                try:
                    s = grid.contour([0.5], scalars="values")
                except Exception:
                    continue
                if s is None or s.n_points == 0:
                    continue
                try:
                    s = s.smooth_taubin(n_iter=12, pass_band=0.1)
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
            QMessageBox.warning(self, "AI 분할", "표시할 뼈가 없습니다.")
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
        """AI 뼈 제거 + threshold volume 복원."""
        self._clear_separated_actors()
        self.separated_bones = []
        self.ai_segmentation_active = False
        self._restore_volume_after_ai()
        if hasattr(self, "separation_status_label"):
            self.separation_status_label.setText("")
        if hasattr(self, "_refresh_separation_list"):
            self._refresh_separation_list()
        try:
            self.plotter.update()
        except Exception:
            pass
