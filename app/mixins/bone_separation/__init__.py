"""Bone separation feature, split into focused sub-mixins.

`BoneSeparationMixin` is composed via multiple inheritance from:
  - BoneListUIMixin    (list_ui.py)     — palette/uid/list/visibility/rename/3D click
  - BoneSeparateMixin  (separate.py)    — connected-component separation
  - BoneMergeMixin     (merge.py)       — simple merge + STL export + undo merge
  - BoneCoordsMixin    (coords.py)      — fusion coord transforms
  - BoneVoxelOpsMixin  (voxel_ops.py)   — HU-CC exclusion, region grow, opposing axes
  - BoneRestoreMixin   (restore.py)     — restore mode UI + voxel restore + undo
  - BoneMergeFillMixin (merge_fill.py)  — merge & fill gap

External imports (e.g. MainWindow) should keep using
`from app.mixins.bone_separation import BoneSeparationMixin` — that path
resolves to this package and the composed class below.
"""

from app.mixins.bone_separation.coords import BoneCoordsMixin
from app.mixins.bone_separation.list_ui import BoneListUIMixin
from app.mixins.bone_separation.merge import BoneMergeMixin
from app.mixins.bone_separation.merge_fill import BoneMergeFillMixin
from app.mixins.bone_separation.restore import BoneRestoreMixin
from app.mixins.bone_separation.separate import BoneSeparateMixin
from app.mixins.bone_separation.voxel_ops import BoneVoxelOpsMixin


class BoneSeparationMixin(
    BoneListUIMixin,
    BoneSeparateMixin,
    BoneMergeMixin,
    BoneCoordsMixin,
    BoneVoxelOpsMixin,
    BoneRestoreMixin,
    BoneMergeFillMixin,
):
    """Composed mixin for all bone separation features."""
    pass


__all__ = ['BoneSeparationMixin']
