"""
mesh_pipeline.py — HU 배열 → 3D Mesh 생성 파이프라인

데이터 흐름 순서:
  ⑤ _build_image_data          HU 배열 + spacing → PyVista ImageData 격자
  ⑥ _compute_masked_values     bone_mask → global closing(복원) → global opening(파티클 제거)
  ⑦ contour (marching cubes)   ImageData → PolyData isosurface
  ⑧ mesh 후처리                _close_surface → _apply_smoothing → _close_surface → _apply_stage_a
  ⑨ 자동 뼈 분리 시각화         mesh CC → 색상별 표시 + UI bone list

Spacing 컨벤션:
  이 모듈에 전달되는 spacing은 (z, y, x) 순서.
  PyVista ImageData는 (x, y, z) 순서를 요구하므로 _build_image_data에서 뒤집음.

뼈 분리 방식:
  Voxel CC는 noise bridge를 통해 가까운 뼈들을 같은 그룹으로 묶는 문제가 있음.
  대신 mesh-level CC (mesh.connectivity)를 사용 — marching cubes 이후
  물리적으로 분리된 surface가 자연스럽게 독립 component가 됨.
"""

import numpy as np
import pyvista as pv
import vtk
from scipy.ndimage import (binary_closing, binary_opening,
                           gaussian_filter, generate_binary_structure)


class MeshPipelineMixin:

    # ──────────────────────────────────────────────────────────────────
    # ⑤ HU 배열 → PyVista ImageData 격자
    # ──────────────────────────────────────────────────────────────────

    def _build_image_data(self, image_hu, spacing):
        """HU 배열을 PyVista ImageData로 래핑.

        좌표 매핑:
          image_hu.shape == (nz, ny, nx)     — DICOM: slices, rows, cols
          spacing        == (z_sp, y_sp, x_sp)

          PyVista grid axis (i, j, k):
            i (x) → col 방향
            j (y) → row 방향
            k (z) → slice 방향

        Parameters
        ----------
        image_hu : np.ndarray, shape (nz, ny, nx)
        spacing  : tuple (z_sp, y_sp, x_sp) in mm

        Returns
        -------
        pv.ImageData
            point_data["values"]에 HU 값이 flatten(order="C")으로 저장됨
        """
        nz, ny, nx = image_hu.shape
        sz, sy, sx = spacing  # (z, y, x) 언패킹

        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(sx, sy, sz),   # PyVista는 (x, y, z) 순서
        )
        grid.point_data["values"] = image_hu.flatten(order="C")
        return grid

    # ──────────────────────────────────────────────────────────────────
    # ⑥ Voxel 가공 — global closing(복원) → global opening(파티클 제거)
    # ──────────────────────────────────────────────────────────────────

    def _compute_masked_values(self, image_hu):
        """HU 배열에 voxel-level 가공을 적용하여 contour 입력용 값 배열을 생성.

        처리 순서:
          1) bone_mask 생성: HU >= min_threshold인 voxel
          2) Global closing: 뼈 내부 빈 공간 메움 (dilation → erosion)
          3) Global opening: noise 파티클 제거 (erosion → dilation)

        뼈 분리는 voxel 단계에서 하지 않음.
        Voxel CC는 noise bridge 때문에 가까운 뼈들을 같은 그룹으로 묶는 문제가 있어,
        mesh-level CC (⑨단계)에서 marching cubes 이후 물리적 분리로 처리.

        Parameters
        ----------
        image_hu : np.ndarray, shape (nz, ny, nx), dtype int16

        Returns
        -------
        np.ndarray
            shape (nz*ny*nx,), dtype int16 — flatten(order="C") 된 masked HU 값
        """
        values = image_hu.astype(np.int16, copy=True).ravel(order="C")
        bone_mask = image_hu >= self.current_min_threshold

        # ── Global closing (복원) ──
        # 뼈 내부 빈 공간을 메움. 6-connectivity (struct=1) 커널 사용.
        if getattr(self, 'closing_enabled', False) and getattr(self, 'closing_iterations', 0) > 0:
            c_iters = int(self.closing_iterations)
            struct_close = generate_binary_structure(3, 1)
            try:
                bone_mask = binary_closing(bone_mask, structure=struct_close,
                                           iterations=c_iters)
            except Exception as e:
                print(f"[Global closing] failed: {e}")

        # ── Global opening (파티클 제거) ──
        # noise voxel 제거. connectivity는 UI에서 선택 (1=6-conn, 2=18, 3=26).
        if self.particle_removal_enabled and self.opening_iterations > 0:
            o_iters = int(self.opening_iterations)
            struct_open = generate_binary_structure(3, self.opening_connectivity)
            try:
                bone_mask = binary_opening(bone_mask, structure=struct_open,
                                           iterations=o_iters)
            except Exception as e:
                print(f"[Global opening] failed: {e}")

        mask_out = ~bone_mask
        values[mask_out.ravel(order="C")] = self.current_min_threshold - 1
        return values

    # ──────────────────────────────────────────────────────────────────
    # ⑧ Mesh 후처리 — smoothing, hole filling, fragment removal
    # ──────────────────────────────────────────────────────────────────

    def _close_surface(self, mesh):
        """Mesh의 topological hole(boundary edge)을 채우고 clean.

        vtkFillHolesFilter는 boundary edge loop만 채움.
        medullary canal처럼 닫힌 내부 공동은 boundary edge가 없으므로
        size와 관계없이 절대 채워지지 않음.

        Parameters
        ----------
        mesh : pv.PolyData or None

        Returns
        -------
        pv.PolyData
        """
        if mesh is None or mesh.n_points == 0:
            return mesh
        try:
            if getattr(self, 'mesh_fill_holes_enabled', True):
                size = float(getattr(self, 'mesh_fill_holes_size', 1e10))
                if size > 0:
                    mesh = mesh.fill_holes(size)
            mesh = mesh.clean()
        except Exception as e:
            print(f"[close_surface] {e}")
        return mesh

    def _apply_smoothing(self, mesh):
        """Smoothing 적용 + smoothing으로 생긴 micro-fragment 제거.

        Smoothing 방식은 smooth_combo UI에서 선택:
          - Laplacian: mesh.smooth(n_iter=100)
          - Windowed Sinc (Taubin): mesh.smooth_taubin(n_iter=50, pass_band=0.05)
          - None: 그대로 반환

        Smoothing 후 50 faces 미만의 micro-fragment는 자동 제거
        (aggressive smoothing이 얇은 뼈 구조를 끊어서 생기는 파편).
        """
        if mesh is None or mesh.n_points == 0:
            return mesh
        if not hasattr(self, 'smooth_combo'):
            return mesh

        method = self.smooth_combo.currentText()
        if "Laplacian" in method:
            mesh = mesh.smooth(n_iter=100)
        elif "Windowed Sinc" in method:
            mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.05)
        else:
            return mesh

        # Smoothing 후 micro-fragment 제거 (50 faces 미만)
        try:
            labeled = mesh.connectivity(extraction_mode='all')
            rids = labeled.cell_data.get('RegionId')
            if rids is not None and len(rids) > 0 and rids.max() > 0:
                counts = np.bincount(rids.astype(int), minlength=int(rids.max()) + 1)
                counts[0] = 99999  # region 0은 제거하지 않음
                tiny_ids = np.where(counts < 50)[0]
                if 0 < len(tiny_ids) < len(counts):
                    cells_to_remove = np.where(np.isin(rids, tiny_ids))[0]
                    mesh = labeled.remove_cells(cells_to_remove)
                    if not isinstance(mesh, pv.PolyData):
                        mesh = mesh.extract_surface()
                    for arr in ('RegionId',):
                        if arr in mesh.point_data:
                            del mesh.point_data[arr]
                        if arr in mesh.cell_data:
                            del mesh.cell_data[arr]
        except Exception as e:
            print(f"[Smoothing post-cleanup] {e}")

        return mesh

    def _apply_stage_a(self, mesh):
        """Stage A: mesh fragment 크기 기반 제거.

        모드:
          - keep_largest_only: 가장 큰 component만 유지
          - min_fragment_faces: 이 수 미만의 faces를 가진 component 제거

        Parameters
        ----------
        mesh : pv.PolyData or None

        Returns
        -------
        pv.PolyData
        """
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

    # ──────────────────────────────────────────────────────────────────
    # ⑨ 자동 뼈 분리 + 색상별 시각화
    # ──────────────────────────────────────────────────────────────────

    def _auto_separate_and_display(self):
        """base_mesh를 mesh-level CC로 분리하여 뼈별 색상 표시.

        Marching cubes 이후 물리적으로 분리된 surface는 자연스럽게
        독립 connected component가 됨. Voxel CC와 달리 noise bridge를
        통해 가까운 뼈들이 묶이는 문제가 없음.
        """
        if self.base_mesh is None or self.base_mesh.n_points == 0:
            return

        # 기존 actors 정리
        self._clear_separated_actors()
        if self.current_mesh_actor:
            try:
                self.plotter.remove_actor(self.current_mesh_actor)
            except Exception:
                pass
            self.current_mesh_actor = None

        # Crop 적용
        mesh = self.base_mesh
        if self.cropping_bounds is not None and self.crop_checkbox.isChecked():
            try:
                mesh = mesh.clip_box(self.cropping_bounds, invert=False)
            except Exception:
                pass

        if mesh is None or mesh.n_points == 0:
            return

        # ── Mesh-level CC 분리 ──
        try:
            labeled = mesh.connectivity(extraction_mode='all')
        except Exception as e:
            print(f"[Auto separate] connectivity failed: {e}")
            self._display_single_mesh(mesh)
            return

        rids = labeled.cell_data.get('RegionId')
        if rids is None or len(rids) == 0:
            self._display_single_mesh(mesh)
            return

        rids_arr = np.asarray(rids)
        unique_ids, counts = np.unique(rids_arr, return_counts=True)

        if len(unique_ids) == 0:
            self._display_single_mesh(mesh)
            return

        # 크기순 내림차순 정렬
        order = np.argsort(-counts)
        unique_ids = unique_ids[order]
        counts = counts[order]

        # min cells 필터
        min_cells = max(1, int(getattr(self, 'min_bone_voxels', 1)))
        keep_mask = counts >= min_cells
        if not keep_mask.any():
            keep_mask[0] = True  # 최소한 가장 큰 것은 유지

        colors = self._bone_color_palette(int(keep_mask.sum()))
        color_idx = 0

        for rid, count, keep in zip(unique_ids, counts, keep_mask):
            if not keep:
                continue
            try:
                cells_idx = np.where(rids_arr == rid)[0]
                sub = labeled.extract_cells(cells_idx)
                if not isinstance(sub, pv.PolyData):
                    sub = sub.extract_surface()
                for arr in ('RegionId',):
                    if arr in sub.point_data:
                        del sub.point_data[arr]
                    if arr in sub.cell_data:
                        del sub.cell_data[arr]
            except Exception:
                continue

            if sub is None or sub.n_points == 0:
                continue

            color = colors[color_idx]
            color_idx += 1
            bone_num = color_idx

            actor = self.plotter.add_mesh(
                sub, color=color, specular=0.5, smooth_shading=True
            )

            self.separated_bones.append({
                'uid': self._new_bone_uid(),
                'id': bone_num,
                'mesh': sub,
                'raw_mesh': sub.copy(deep=True),
                'actor': actor,
                'visible': True,
                'color': color,
                'voxel_count': int(count),
                'name': f"Bone {bone_num}",
                'series_index': None,
            })

        if not self.separated_bones:
            self._display_single_mesh(mesh)
            return

        # UI 상태 업데이트
        self.bone_separation_enabled = True
        if hasattr(self, 'clear_separation_btn'):
            self.clear_separation_btn.setEnabled(True)
        self._set_separation_tools_enabled(True)
        if hasattr(self, 'separation_status_label'):
            self.separation_status_label.setText(
                f"{len(self.separated_bones)} bone(s) separated"
            )
        self._refresh_separation_list()

        # 카메라
        if not hasattr(self, '_camera_initialized') or not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True

        self.plotter.update()

    def _display_single_mesh(self, mesh):
        """Fallback: CC 분리 실패 시 단일 ivory mesh로 표시."""
        if self.current_mesh_actor:
            try:
                self.plotter.remove_actor(self.current_mesh_actor)
            except Exception:
                pass
        self.current_mesh_actor = self.plotter.add_mesh(
            mesh, color="ivory", specular=0.5, smooth_shading=True
        )
        if not hasattr(self, '_camera_initialized') or not self._camera_initialized:
            self.plotter.reset_camera()
            self._camera_initialized = True
        self.plotter.update()

    # ──────────────────────────────────────────────────────────────────
    # ⑦+⑧+⑨ 파이프라인 실행 — masking → contour → 후처리 → 시각화
    # ──────────────────────────────────────────────────────────────────

    def update_base_mesh(self):
        """Re-render 단일 진입점 — VOLUME-ONLY 파이프라인.

        설계 결정 (사용자 요청):
          • Particle removal / bone separation / marching cubes 메쉬는
            **하지 않는다**. 화면에 보이는 것은 오직 binary-solid volume.
          • Smoothing은 메쉬가 아니라 volume HU 데이터에 Gaussian으로 적용,
            그리고 volume은 뼈 bbox로 크롭됨 (_build_render_grid).
          • Bone 분류 cutoff = current_min_threshold (threshold 슬라이더).
          • Landmark picking은 vtkVolumePicker 기반 (메쉬가 없으므로).
        """
        if self._loading_session:
            return
        self._clear_undo_stack()

        suppress_attr_existed = False
        if hasattr(self.plotter, 'suppress_rendering'):
            suppress_attr_existed = True
            old_suppress = self.plotter.suppress_rendering
            self.plotter.suppress_rendering = True

        try:
            # Remove any leftover mesh / separation / fusion-mesh actors
            # from older builds or a previous (mesh-based) session.
            self._clear_mesh_and_separation_actors()
            self._update_volume_render()
        finally:
            if suppress_attr_existed:
                self.plotter.suppress_rendering = old_suppress

        try:
            self.plotter.render()
        except Exception:
            pass

        # Re-attach the 2D "hide above" clip plane to the fresh volumes.
        if hasattr(self, '_reapply_slice_clipping_after_mesh_rebuild'):
            self._reapply_slice_clipping_after_mesh_rebuild()

    def _clear_mesh_and_separation_actors(self):
        """Drop every marching-cubes / separated-bone / fusion-mesh actor.

        Volume mode shows none of these; clearing them keeps the scene
        to just the volume actors (+ landmark spheres + 2D indicators).
        """
        # Single combined mesh
        if getattr(self, 'current_mesh_actor', None) is not None:
            try:
                self.plotter.remove_actor(self.current_mesh_actor)
            except Exception:
                pass
            self.current_mesh_actor = None
        # Fusion mesh actors
        for actor in list(getattr(self, 'fusion_actors', []) or []):
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        if hasattr(self, 'fusion_actors'):
            self.fusion_actors.clear()
        if hasattr(self, 'fusion_meshes'):
            self.fusion_meshes.clear()
        # Separated-bone actors (via the dedicated helper if present)
        if hasattr(self, '_clear_separated_actors'):
            try:
                self._clear_separated_actors()
            except Exception:
                pass
        if hasattr(self, 'separated_bones'):
            self.separated_bones = []

    # ──────────────────────────────────────────────────────────────────
    # Volume rendering pipeline (replaces visible marching-cubes mesh)
    # ──────────────────────────────────────────────────────────────────
    def _hide_all_mesh_actors_for_volume_mode(self):
        """Hide every bone-mesh actor so the volume actors are the sole
        visible bone representation. Mesh actors are kept alive — STL
        export, bone-separation list, landmark surface picking all rely
        on them existing."""
        if getattr(self, 'current_mesh_actor', None) is not None:
            try:
                self.current_mesh_actor.SetVisibility(False)
            except Exception:
                pass
        for actor in getattr(self, 'fusion_actors', []) or []:
            try:
                actor.SetVisibility(False)
            except Exception:
                pass
        for entry in getattr(self, 'separated_bones', []) or []:
            if isinstance(entry, dict):
                actor = entry.get('actor')
                if actor is not None:
                    try:
                        actor.SetVisibility(False)
                    except Exception:
                        pass

    def _update_volume_render(self):
        """Rebuild the volume rendering for the current view.

        Single series → one vtkVolume. Multiple (fusion) series → a single
        vtkMultiVolume so overlapping series composite with correct depth.
        Rendering each series as its own vtkVolume (the previous approach)
        let the back series bleed through the front and the alpha-blend
        darkened the colour — the root cause of the "see-through" and
        "too dark" complaints in fusion.
        """
        # 1) Tear down old volume actors.
        for actor in list(getattr(self, 'volume_actors', []) or []):
            try:
                self.plotter.renderer.RemoveVolume(actor)
            except Exception:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
        self.volume_actors = []
        self._volume_props = []

        if not self.all_series_data:
            return

        # 2) Collect (cropped grid, base-frame transform) for each series.
        entries = []
        if not self.fusion_enabled:
            grid = self._build_render_grid(self.current_image_hu, self.current_spacing)
            if grid is not None:
                entries.append((grid, None))
        else:
            base_idx = max(0, min(self.base_series_index, len(self.all_series_data) - 1))
            base_meta = self.all_series_data[base_idx].get('meta', {}) or {}
            T_base = self._series_grid_to_lps_matrix(base_meta)
            T_base_inv = np.linalg.inv(T_base) if T_base is not None else None

            if len(self.fusion_include_flags) != len(self.all_series_data):
                self.fusion_include_flags = [True] * len(self.all_series_data)

            # base first → it becomes port 0 (identity) in the multi-volume.
            order = [base_idx] + [
                i for i in range(len(self.all_series_data)) if i != base_idx
            ]
            for i in order:
                if not self.fusion_include_flags[i]:
                    continue
                sd = self.all_series_data[i]
                grid = self._build_render_grid(sd['image_hu'], sd['spacing'])
                if grid is None:
                    continue
                transform = None
                if i != base_idx and T_base_inv is not None:
                    T_i = self._series_grid_to_lps_matrix(sd.get('meta', {}) or {})
                    if T_i is not None:
                        transform = T_base_inv @ T_i
                entries.append((grid, transform))

        if not entries:
            return

        # 3) Shared property (same ivory + shading for every series).
        prop = self._make_volume_property()
        self._volume_props = [prop]

        # 4) Build the actor(s).
        built = False
        if len(entries) > 1 and hasattr(vtk, 'vtkMultiVolume'):
            try:
                self._build_multi_volume(entries, prop)
                built = True
            except Exception as e:
                print(f"[volume] vtkMultiVolume failed ({e}); using separate volumes")
                for a in list(self.volume_actors):
                    try:
                        self.plotter.renderer.RemoveVolume(a)
                    except Exception:
                        pass
                self.volume_actors = []
        if not built:
            for grid, transform in entries:
                vol = self._build_single_volume(grid, transform, prop)
                if vol is not None:
                    self.volume_actors.append(vol)

        if not getattr(self, '_camera_initialized', False):
            try:
                self.plotter.reset_camera()
                self._camera_initialized = True
            except Exception:
                pass

        try:
            self.plotter.render()
        except Exception:
            pass

    def _make_volume_property(self):
        """Shared vtkVolumeProperty: flat-ivory colour + binary opacity +
        Phong shading. One instance is shared by every sub-volume so the
        bone looks identical across series (and so W/L updates touch a
        single object)."""
        r, g, b = self._volume_color_points()[0][1:4]
        ctf = vtk.vtkColorTransferFunction()
        for hu in (-1024.0, 0.0, 500.0, 1500.0, 3000.0):
            ctf.AddRGBPoint(hu, r, g, b)

        otf = vtk.vtkPiecewiseFunction()
        for x, a in self._volume_opacity_points():
            otf.AddPoint(float(x), float(a))

        prop = vtk.vtkVolumeProperty()
        prop.SetColor(ctf)
        prop.SetScalarOpacity(otf)
        # NOTE: no SetGradientOpacity — vtkMultiVolume rejects gradient
        # opacity, and our gof was a constant 1 anyway (no effect).
        prop.SetInterpolationTypeToLinear()
        # Depth/shape comes from the AMBIENT↔DIFFUSE contrast: a face
        # turned toward the (head)light gets ambient+diffuse ≈ 1.0 (bright
        # ivory); a face turned away gets ambient only ≈ 0.5 (still a light
        # cream, not black, because the colour is near-white). Earlier the
        # ambient was pushed to 0.85 to fix "too dark", but that flattened
        # the contrast to 0.85–1.0 → no visible relief. 0.5/0.55 restores
        # the relief while the bright colour keeps the dim side acceptable.
        prop.ShadeOn()
        prop.SetAmbient(0.5)
        prop.SetDiffuse(0.55)
        prop.SetSpecular(0.0)
        return prop

    def _tune_volume_mapper(self, mapper, spacings):
        """Composite mode + fine fixed sample distance.

        Sampling at half the smallest voxel (≥2 samples/voxel) fixes both
        the rotate-time shimmer and the "thin cortical shell lets the bone
        behind show through" problem (too-coarse sampling never accumulates
        full opacity across a 1-2 voxel wall)."""
        try:
            mapper.SetBlendModeToComposite()
        except Exception:
            pass
        try:
            fine = (min(spacings) * 0.5) if spacings else 0.5
            if hasattr(mapper, 'SetSampleDistance'):
                mapper.SetSampleDistance(fine)
            if hasattr(mapper, 'SetAutoAdjustSampleDistances'):
                mapper.SetAutoAdjustSampleDistances(False)
            if hasattr(mapper, 'SetInteractiveAdjustSampleDistances'):
                mapper.SetInteractiveAdjustSampleDistances(False)
        except Exception as e:
            print(f"[volume] sample-distance tuning skipped: {e}")

    def _np_to_vtk_matrix(self, transform):
        m = vtk.vtkMatrix4x4()
        for r in range(4):
            for c in range(4):
                m.SetElement(r, c, float(transform[r, c]))
        return m

    def _spacings_of(self, grid):
        try:
            return [float(s) for s in grid.spacing if float(s) > 0]
        except Exception:
            return []

    def _build_single_volume(self, grid, transform, prop):
        """One vtkVolume → renderer. Used in single mode, single-series
        fusion, and as the vtkMultiVolume fallback."""
        if grid is None or grid.n_points == 0:
            return None
        try:
            grid.set_active_scalars('values')
        except Exception:
            pass
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(grid)
        self._tune_volume_mapper(mapper, self._spacings_of(grid))
        vol = vtk.vtkVolume()
        vol.SetMapper(mapper)
        vol.SetProperty(prop)
        vol.SetPickable(True)
        if transform is not None:
            try:
                vol.SetUserMatrix(self._np_to_vtk_matrix(transform))
            except Exception as e:
                print(f"[volume] SetUserMatrix failed: {e}")
        try:
            self.plotter.renderer.AddVolume(vol)
        except Exception as e:
            print(f"[volume] AddVolume failed: {e}")
            return None
        return vol

    def _build_multi_volume(self, entries, prop):
        """Render every series through ONE vtkGPUVolumeRayCastMapper +
        vtkMultiVolume so overlapping series composite with correct depth
        (separate vtkVolumes don't depth-intermix — the back series bleeds
        through and the blend darkens)."""
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        spacings = []
        for grid, _ in entries:
            spacings += self._spacings_of(grid)
        self._tune_volume_mapper(mapper, spacings)

        multi = vtk.vtkMultiVolume()
        multi.SetMapper(mapper)
        for port, (grid, transform) in enumerate(entries):
            try:
                grid.set_active_scalars('values')
            except Exception:
                pass
            mapper.SetInputDataObject(port, grid)
            sub = vtk.vtkVolume()
            sub.SetProperty(prop)
            if transform is not None:
                sub.SetUserMatrix(self._np_to_vtk_matrix(transform))
            multi.SetVolume(sub, port)
        multi.SetPickable(True)
        self.plotter.renderer.AddVolume(multi)
        self.volume_actors.append(multi)

    def _update_volume_opacity_transfer(self):
        """Re-apply the opacity TF after a W/L change. Operates on the
        shared property objects, so it works for both single volumes and
        the multi-volume (whose actor has no GetProperty)."""
        otf = vtk.vtkPiecewiseFunction()
        for x, a in self._volume_opacity_points():
            otf.AddPoint(float(x), float(a))
        for prop in getattr(self, '_volume_props', None) or []:
            try:
                prop.SetScalarOpacity(otf)
            except Exception:
                pass

    def _volume_cutoff(self):
        """HU value at/above which a voxel is classified as bone for the
        binary volume. Driven by the threshold slider (current_min_threshold)
        so the user adjusts bone inclusion exactly like the old mesh path."""
        return float(getattr(self, 'current_min_threshold', 150))

    def _volume_smooth_sigma(self):
        """Gaussian sigma for the current Smoothing combo (0=None / Light / Strong)."""
        smooth_idx = 0
        if hasattr(self, 'smooth_combo'):
            try:
                smooth_idx = int(self.smooth_combo.currentIndex())
            except Exception:
                smooth_idx = 0
        return {0: 0.0, 1: 0.8}.get(smooth_idx, 1.5)

    def _build_render_grid(self, image_hu, spacing):
        """Build a pv.ImageData for volume rendering, CROPPED to the bone
        bounding box and (optionally) Gaussian-smoothed.

        Why crop: voxels below the bone cutoff are fully transparent, but
        an uncropped grid still makes the ray-caster set up over the full
        512×512×N extent and wastes memory. Cropping to the bone's tight
        bbox (+ a small margin) drops the empty borders → less memory,
        faster rendering, faster volume picking.

        Coordinate handling: the cropped grid's ORIGIN is set to
        (x0·sx, y0·sy, z0·sz) so its points still sit at the *same* world
        positions they would in the full grid. That keeps the fusion
        ``SetUserMatrix`` transform (which assumes original series-grid
        coordinates) correct without modification.

        Returns None if no voxel reaches the bone cutoff.
        """
        if image_hu is None:
            return None
        cutoff = self._volume_cutoff()
        mask = image_hu >= cutoff
        if not mask.any():
            return None

        nz, ny, nx = image_hu.shape
        zs = np.any(mask, axis=(1, 2))
        ys = np.any(mask, axis=(0, 2))
        xs = np.any(mask, axis=(0, 1))
        margin = 2
        z0 = max(0, int(np.argmax(zs)) - margin)
        z1 = min(nz, nz - int(np.argmax(zs[::-1])) + margin)
        y0 = max(0, int(np.argmax(ys)) - margin)
        y1 = min(ny, ny - int(np.argmax(ys[::-1])) + margin)
        x0 = max(0, int(np.argmax(xs)) - margin)
        x1 = min(nx, nx - int(np.argmax(xs[::-1])) + margin)

        sub = image_hu[z0:z1, y0:y1, x0:x1].astype(np.float32)

        sigma = self._volume_smooth_sigma()
        if sigma > 0.0:
            try:
                sub = gaussian_filter(sub, sigma=sigma)
            except Exception as e:
                print(f"[volume] smoothing failed: {e}")

        csz, csy, csx = sub.shape
        sz, sy, sx = float(spacing[0]), float(spacing[1]), float(spacing[2])
        grid = pv.ImageData(
            dimensions=(csx, csy, csz),
            spacing=(sx, sy, sz),
            origin=(x0 * sx, y0 * sy, z0 * sz),
        )
        grid.point_data['values'] = sub.ravel(order='C')
        return grid

    def _volume_opacity_points(self):
        """BINARY opacity transfer function — "this voxel is bone or it
        isn't", exactly like the old above-threshold classification.

        Below the cutoff every voxel is fully transparent; at/above it
        every voxel is FULLY OPAQUE. No HU-gradient translucency, so:
          • bones look like solid objects (and stay solid when clipped),
          • bone-vs-bone boundaries (low-HU gaps) read cleanly,
          • rays terminate at the first bone voxel → lighter to render,
            which also cuts down the coarse-sampling "breakup" while the
            camera is moving.
        """
        cutoff = self._volume_cutoff()
        return [
            (-1000.0,      0.0),
            (cutoff - 1.0, 0.0),
            (cutoff,       1.0),   # hard step: bone = fully opaque
            (3000.0,       1.0),
        ]

    def _volume_color_points(self):
        """Single near-white ivory for every bone voxel — depth comes from
        Phong shading, not colour variation. Kept very bright so that even
        the shaded (dim) side of the bone stays a light cream rather than
        going dark."""
        r, g, b = 1.0, 0.98, 0.92
        return [
            (-1000.0, r, g, b),
            (3000.0,  r, g, b),
        ]

    def _ivory_cmap(self):
        """A flat warm-ivory matplotlib Colormap. Used as ``cmap=`` for
        ``add_volume`` so the bone colour is baked into the colour TF that
        pyvista builds itself — much more reliable than a post-hoc
        ``prop.SetColor()`` override (which kept being overwritten and made
        the bones look grey / brown / black at low HU)."""
        from matplotlib.colors import ListedColormap
        r, g, b = self._volume_color_points()[0][1:4]
        return ListedColormap(np.tile([r, g, b, 1.0], (256, 1)))

    def _update_single_mesh(self):
        """Single-series 모드의 전체 mesh 생성 파이프라인.

        실행 순서:
          ⑥ _compute_masked_values  → global closing(복원) → global opening(파티클 제거)
          ⑦ volume_grid.contour     → marching cubes isosurface
          ⑧ _close_surface          → hole filling (1차)
             _apply_smoothing       → surface smoothing
             _close_surface         → hole filling (2차)
             _apply_stage_a         → 작은 fragment 제거
          ⑨ mesh CC → 뼈 자동 분리 + 색상별 시각화
        """
        if self.volume_grid is None:
            return

        # fusion 모드 actor 잔여물 정리
        if self.fusion_actors:
            for actor in self.fusion_actors:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
            self.fusion_actors.clear()
            self.fusion_meshes.clear()

        # ⑥ Masking (global closing → global opening)
        values = self._compute_masked_values(self.current_image_hu)
        self.volume_grid.point_data["masked"] = values

        # ⑦ Marching cubes
        self.base_mesh = self.volume_grid.contour(
            [self.current_min_threshold], scalars="masked"
        )

        # ⑧ 후처리
        self.base_mesh = self._close_surface(self.base_mesh)
        self.base_mesh = self._apply_smoothing(self.base_mesh)
        self.base_mesh = self._close_surface(self.base_mesh)
        self.base_mesh = self._apply_stage_a(self.base_mesh)

        # ⑨ mesh CC → 뼈 자동 분리 + 색상별 시각화
        self._auto_separate_and_display()

    def update_rendered_mesh(self):
        """base_mesh가 변경될 때 호출 (click-to-remove, cropping 등).

        자동 뼈 분리를 다시 수행하여 색상별 표시를 갱신.
        Fusion 모드에서는 _update_fused_meshes가 직접 관리.
        """
        if self.fusion_enabled:
            return
        if self.base_mesh is None:
            return
        self._auto_separate_and_display()

    # ──────────────────────────────────────────────────────────────────
    # LPS 좌표 변환 — fusion / bone separation에서 사용
    # ──────────────────────────────────────────────────────────────────

    def _series_grid_to_lps_matrix(self, meta):
        """PyVista grid 좌표 → DICOM LPS(mm) 변환 4x4 affine 행렬 생성.

        LPS = ipp + x·row_dir + y·col_dir + z·normal_dir

        Parameters
        ----------
        meta : dict
            DICOM geometry 정보 (ipp_first, row_dir, col_dir, normal_dir)

        Returns
        -------
        np.ndarray (4, 4) or None
            DICOM geometry가 없거나 불완전하면 None
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
