"""Combine per-structure binary masks into one unified-label segmentation.

TotalSegmentator (and CADS) store one binary mask per anatomical structure
(segmentations/<structure>.nii.gz), not a single multi-label volume. This module
merges the bone structures into a single array carrying our unified taxonomy ids,
so the result can flow through build_raw with an IDENTITY label_map.

TotalSegmentator v2 structure names map cleanly (skull, sternum, sacrum,
vertebrae_C1.., rib_left_1.., hip_left/right). CADS uses numeric `part_NNN` files
whose number→structure table must be supplied separately (deferred).
"""
from ai_bone import taxonomy_v1 as tx


def _ts_name_to_unified():
    m = {"skull": "Skull", "sternum": "Sternum", "sacrum": "Sacrum",
         "vertebrae_S1": "Sacrum", "hip_left": "Hip_L", "hip_right": "Hip_R"}
    for i in range(1, 8):  m[f"vertebrae_C{i}"] = f"C{i}"
    for i in range(1, 13): m[f"vertebrae_T{i}"] = f"T{i}"
    for i in range(1, 6):  m[f"vertebrae_L{i}"] = f"L{i}"
    for i in range(1, 13): m[f"rib_left_{i}"] = f"Rib_L_{i}"
    for i in range(1, 13): m[f"rib_right_{i}"] = f"Rib_R_{i}"
    return m

TS_NAME_TO_UNIFIED = _ts_name_to_unified()


def combine_arrays(binaries, name_to_unified=TS_NAME_TO_UNIFIED):
    """binaries: {structure_name: boolean/int mask array} → single uint8 array of
    unified ids. Later structures overwrite earlier on overlap (bones rarely do)."""
    import numpy as np
    combined = None
    for name, mask in binaries.items():
        uni = name_to_unified.get(name)
        if uni is None:
            continue
        m = np.asarray(mask) > 0
        if combined is None:
            combined = np.zeros(m.shape, dtype=np.uint8)
        combined[m] = tx.NAME_TO_ID[uni]
    return combined


def combine_case(seg_dir, name_to_unified=TS_NAME_TO_UNIFIED, reader=None):
    """Read per-structure .nii.gz in seg_dir and combine → (sitk image of unified ids).
    Geometry taken from the first present mask. Returns None if no mapped mask found."""
    import os
    import SimpleITK as sitk
    read = reader or (lambda p: sitk.ReadImage(p))
    binaries, ref = {}, None
    for name in name_to_unified:
        p = os.path.join(seg_dir, name + ".nii.gz")
        if not os.path.exists(p):
            continue
        img = read(p)
        binaries[name] = sitk.GetArrayFromImage(img)
        ref = ref or img
    if ref is None:
        return None
    combined = combine_arrays(binaries, name_to_unified)
    out = sitk.GetImageFromArray(combined)
    out.CopyInformation(ref)
    return out


def main():
    """Combine TotalSegmentator per-structure masks → combined/<case>.nii.gz (unified ids).
    Usage: python -m ai_bone.datasets.combine --root <extracted_totalseg> --out <combined_dir>"""
    import argparse, glob, os
    import SimpleITK as sitk
    ap = argparse.ArgumentParser(description="Combine per-structure bone masks (TotalSeg).")
    ap.add_argument("--root", required=True, help="extracted TotalSegmentator dir (sXXXX/)")
    ap.add_argument("--out", required=True, help="output dir for combined segs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for d in sorted(glob.glob(os.path.join(args.root, "s*"))):
        seg_dir = os.path.join(d, "segmentations")
        if not os.path.isdir(seg_dir):
            continue
        img = combine_case(seg_dir)
        if img is None:
            continue
        sitk.WriteImage(img, os.path.join(args.out, os.path.basename(d) + ".nii.gz"))
        n += 1
    print(f"combined {n} cases → {args.out}")


if __name__ == "__main__":
    main()
