"""Spine-Mets-CT-SEG ETL: DICOM CT series + DICOM-SEG → (CT, unified-id seg).

Spine-Mets stores, per patient, one CT series (many .dcm slices) and one DICOM-SEG
object (a single multi-frame .dcm) whose segments are labelled per vertebra
('T1 vertebra', ..., 'L5 vertebra'). We:
  1. read the CT series into a SimpleITK image,
  2. decode the SEG (pydicom_seg) into per-segment binaries,
  3. paint each segment onto a unified-taxonomy-id volume (vertebrae outside the
     taxonomy — L6/T13/sacrum — become IGNORE, not background),
  4. resample that volume onto the CT grid so CT and seg share geometry.

The result flows through build_raw with the IDENTITY-style spinemets label_map
(values already unified ids; 255 passed through as ignore), exactly like the
TotalSegmentator combine path.

Heavy DICOM deps (SimpleITK, pydicom, pydicom_seg) are imported lazily so the
pure label parser can be unit-tested in a GPU-free env without them.
"""
import re

from ai_bone import taxonomy_v1 as tx

IGNORE = tx.IGNORE_LABEL

# 'T1 vertebra', 'L5 vertebra', 'C7 vertebra', 'S1 vertebra', ...
_VERT_RE = re.compile(r"^\s*([CTLS])\s*(\d+)\s+vertebra\s*$", re.IGNORECASE)
_MAX = {"C": 7, "T": 12, "L": 5}          # taxonomy vertebra counts (no S* / L6 / T13)


def segment_label_to_unified(label):
    """Map a DICOM-SEG SegmentLabel to a taxonomy name, the sentinel '__ignore__'
    (a vertebra the taxonomy does not model — L6, T13, sacral S*, 'sacrum'), or
    None if the label is not a vertebra at all (e.g. a lesion segment)."""
    if not label:
        return None
    m = _VERT_RE.match(label)
    if not m:
        return "__ignore__" if "sacrum" in label.lower() else None
    prefix, num = m.group(1).upper(), int(m.group(2))
    if prefix in _MAX and 1 <= num <= _MAX[prefix]:
        return f"{prefix}{num}"
    return "__ignore__"                    # C8/L6/T13/S1 → present bone, not in taxonomy


def combine_segments(seg_binaries):
    """seg_binaries: list of (SegmentLabel, boolean mask array, all same shape) →
    single uint8 array of unified ids. Non-taxonomy vertebrae → IGNORE(255).
    Segments painted low→high id so a rare overlap resolves deterministically."""
    import numpy as np
    combined = None
    painted = []
    for label, mask in seg_binaries:
        uni = segment_label_to_unified(label)
        if uni is None:
            continue
        val = IGNORE if uni == "__ignore__" else tx.NAME_TO_ID[uni]
        m = np.asarray(mask) > 0
        if combined is None:
            combined = np.zeros(m.shape, dtype=np.uint8)
        painted.append((val, m))
    if combined is None:
        return None
    for val, m in sorted(painted, key=lambda x: (x[0] != IGNORE, x[0])):
        combined[m] = val                  # IGNORE first, real ids overwrite it (win on overlap)
    return combined


def read_ct_series(series_dir):
    """Read a directory of CT DICOM slices → SimpleITK image (sorted by geometry)."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir)
    if not files:
        raise RuntimeError(f"no DICOM series found in {series_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def seg_to_unified_image(seg_path, ref_ct):
    """Decode a DICOM-SEG file → SimpleITK image of unified ids resampled onto the
    CT grid (nearest-neighbour), so it shares CT geometry/shape exactly."""
    import numpy as np
    import pydicom
    import pydicom_seg
    import SimpleITK as sitk

    ds = pydicom.dcmread(seg_path)
    result = pydicom_seg.SegmentReader().read(ds)
    label_of = {s.SegmentNumber: getattr(s, "SegmentLabel", "")
                for s in ds.SegmentSequence}
    binaries, ref = [], None
    for num in result.available_segments:
        seg_img = result.segment_image(num)          # sitk binary in SEG geometry
        binaries.append((label_of.get(num, ""), sitk.GetArrayFromImage(seg_img)))
        ref = ref or seg_img
    combined = combine_segments(binaries)
    if combined is None:
        return None
    seg_img = sitk.GetImageFromArray(combined)
    seg_img.CopyInformation(ref)
    # place on the CT grid → guarantees size_match with the CT in build_raw/verify
    return sitk.Resample(seg_img, ref_ct, sitk.Transform(), sitk.sitkNearestNeighbor,
                         0, sitk.sitkUInt8)


def convert_case(ct_series_dir, seg_path):
    """(ct_series_dir, seg_path) → (ct sitk image, unified-id seg sitk image on CT grid).
    Returns (ct, None) if the SEG has no taxonomy-relevant segments."""
    ct = read_ct_series(ct_series_dir)
    seg = seg_to_unified_image(seg_path, ct)
    return ct, seg


def find_cases(root):
    """Discover (case_id, ct_series_dir, seg_dcm_path) triples under a Spine-Mets
    DICOM tree. Layout-agnostic: every directory that directly contains *.dcm is a
    series; one slice is read per series for Modality/UIDs. The CT series and the
    SEG object sharing a StudyInstanceUID are paired; case_id = PatientID."""
    import glob
    import os
    import pydicom
    series_dirs = sorted({os.path.dirname(p)
                          for p in glob.glob(os.path.join(root, "**", "*.dcm"),
                                             recursive=True)})
    studies = {}
    for d in series_dirs:
        dcms = sorted(glob.glob(os.path.join(d, "*.dcm")))
        ds = pydicom.dcmread(dcms[0], stop_before_pixels=True, force=True)
        key = (getattr(ds, "PatientID", ""), getattr(ds, "StudyInstanceUID", ""))
        slot = studies.setdefault(key, {})
        mod = getattr(ds, "Modality", "")
        if mod == "SEG":
            slot["seg"] = dcms[0]
        elif mod == "CT":
            slot["ct"] = d
    out = []
    for (patient, study), slot in sorted(studies.items()):
        if "ct" in slot and "seg" in slot:
            out.append((patient or study, slot["ct"], slot["seg"]))
    return out


def main():
    """Convert a Spine-Mets DICOM tree → staged CT/seg NIfTI + a build_raw pairs
    manifest. Usage: python -m ai_bone.datasets.dicom_seg --root <dicom_root>
    --staging <nii_out> --pairs <pairs.json>"""
    import argparse
    import json
    import os
    import SimpleITK as sitk
    ap = argparse.ArgumentParser(description="Spine-Mets DICOM/SEG → NIfTI pairs.")
    ap.add_argument("--root", required=True, help="Spine-Mets DICOM root")
    ap.add_argument("--staging", required=True, help="dir for <case>_ct/_seg.nii.gz")
    ap.add_argument("--pairs", required=True, help="output pairs manifest json")
    args = ap.parse_args()
    os.makedirs(args.staging, exist_ok=True)
    pairs = []
    cases = find_cases(args.root)
    print(f"found {len(cases)} CT+SEG cases", flush=True)
    for cid, ctd, segp in cases:
        try:
            ct, seg = convert_case(ctd, segp)
        except Exception as e:
            print(f"[{cid}] ERROR {e}", flush=True)
            continue
        if seg is None:
            print(f"[{cid}] no taxonomy segments, skip", flush=True)
            continue
        cp = os.path.join(args.staging, f"{cid}_ct.nii.gz")
        sp = os.path.join(args.staging, f"{cid}_seg.nii.gz")
        sitk.WriteImage(ct, cp)
        sitk.WriteImage(seg, sp)
        pairs.append((cp, sp, str(cid)))
        print(f"[{cid}] converted", flush=True)
    with open(args.pairs, "w", encoding="utf-8") as f:
        json.dump([[c, s, i] for c, s, i in pairs], f, indent=1)
    print(f"SPINEMETS_CONVERT_DONE {len(pairs)} cases → {args.pairs}")


if __name__ == "__main__":
    main()
