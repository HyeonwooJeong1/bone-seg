"""Coordinate transforms between fusion base grid and per-series grids.

Methods grouped here:
  - _get_bone_series_data: HU/spacing/meta lookup for a bone entry
  - _bone_pts_to_grid_coords: base grid → original series grid (inverse fusion)
  - _grid_coords_to_bone_frame: original series grid → base grid

All mesh.points stored in self.separated_bones are in BASE GRID coordinates,
so these helpers exist to round-trip when voxel work happens in the original
series' grid space.
"""

import numpy as np


class BoneCoordsMixin:
    def _get_bone_series_data(self, bone_entry):
        """뼈 entry에 해당하는 HU 배열, spacing, meta를 반환."""
        series_idx = bone_entry.get('series_index')
        if (series_idx is not None
                and hasattr(self, 'all_series_data')
                and series_idx < len(self.all_series_data)):
            sd = self.all_series_data[series_idx]
            return sd['image_hu'], sd['spacing'], sd.get('meta') or {}
        return self.current_image_hu, self.current_spacing, self.current_meta_info or {}

    def _bone_pts_to_grid_coords(self, bone_entry, pts):
        """Fusion 모드에서 mesh 좌표를 원래 시리즈 grid 좌표로 역변환.

        Single 모드나 base series면 변환 없이 그대로 반환.
        """
        series_idx = bone_entry.get('series_index')
        if (not getattr(self, 'fusion_enabled', False)
                or series_idx is None
                or series_idx == self.base_series_index):
            return pts

        _, _, meta = self._get_bone_series_data(bone_entry)
        base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(meta)

        if T_base is None or T_i is None:
            return pts

        T_composite = np.linalg.inv(T_base) @ T_i
        T_inv = np.linalg.inv(T_composite)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (T_inv @ pts_h.T).T[:, :3]

    def _grid_coords_to_bone_frame(self, bone_entry, pts):
        """원래 시리즈 grid 좌표를 fusion base frame으로 변환 (역변환의 반대)."""
        series_idx = bone_entry.get('series_index')
        if (not getattr(self, 'fusion_enabled', False)
                or series_idx is None
                or series_idx == self.base_series_index):
            return pts

        _, _, meta = self._get_bone_series_data(bone_entry)
        base_meta = self.all_series_data[self.base_series_index].get('meta') or {}
        T_base = self._series_grid_to_lps_matrix(base_meta)
        T_i = self._series_grid_to_lps_matrix(meta)

        if T_base is None or T_i is None:
            return pts

        T_composite = np.linalg.inv(T_base) @ T_i
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        return (T_composite @ pts_h.T).T[:, :3]
