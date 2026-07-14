import numpy as np
import pytest


def test_nib_to_sitk_orthonormalizes_and_preserves_array():
    nib = pytest.importorskip("nibabel")
    import SimpleITK as sitk
    from ai_bone.nifti_io import _nib_to_sitk

    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)   # (X,Y,Z)
    # slightly non-orthonormal direction like TotalSeg (~2e-4 off), spacing 1.5
    D = np.array([[0.94909, 0.31479, 0.0],
                  [-0.31499, 0.94916, 0.0],
                  [0.0, 0.0, 1.0]])
    aff = np.eye(4)
    aff[:3, :3] = D * 1.5
    aff[:3, 3] = [10.0, 20.0, 30.0]

    img = _nib_to_sitk(nib.Nifti1Image(arr, aff))
    assert img.GetSize() == (4, 5, 6)
    assert np.allclose(img.GetSpacing(), (1.5, 1.5, 1.5), atol=1e-3)
    # direction is now exactly orthonormal (ITK-safe)
    Dout = np.array(img.GetDirection()).reshape(3, 3)
    assert np.allclose(Dout @ Dout.T, np.eye(3), atol=1e-6)
    # voxel data preserved (sitk zyx → back to xyz)
    back = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    assert np.array_equal(back, arr)
