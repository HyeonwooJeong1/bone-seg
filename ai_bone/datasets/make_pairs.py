"""Build the [ct, seg, case_id] pairs manifest that build_raw consumes.

Two dataset layouts:
  - single multi-label seg (CTPelvic1K, VerSe, RibSeg, CTSpine1K, Spine-Mets):
    match each CT to its seg by a shared case token → pairs point at the seg file.
  - per-structure binaries (TotalSegmentator, CADS): run `combine` first to write
    one unified-id seg per case, then pair CT with the combined seg.

Explicit pairs avoid the wrong-CT-file bug (build_raw contract).
"""
import glob
import json
import os


def _token(path, strip):
    b = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            b = b[: -len(ext)]
            break
    return b.replace(strip, "")


def match_by_token(ct_paths, seg_paths, ct_strip="", seg_strip=""):
    """Pair CT↔seg files that share a case token (basename minus the given suffix)."""
    seg_by = {_token(s, seg_strip): s for s in seg_paths}
    pairs = []
    for c in ct_paths:
        t = _token(c, ct_strip)
        if t in seg_by:
            pairs.append((c, seg_by[t], t))
    return pairs


def totalseg_pairs(root, combined_dir):
    """root = extracted TotalSeg (sXXXX/ct.nii.gz); combined_dir = combine.py output."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "s*"))):
        case = os.path.basename(d)
        ct = os.path.join(d, "ct.nii.gz")
        seg = os.path.join(combined_dir, case + ".nii.gz")
        if os.path.exists(ct) and os.path.exists(seg):
            out.append((ct, seg, case))
    return out


def ctpelvic1k_pairs(ct_root, mask_root):
    """Match CTPelvic1K CT (*_data.nii.gz) ↔ mask (*_mask*label*.nii.gz) by case token."""
    ct = glob.glob(os.path.join(ct_root, "**", "*_data.nii.gz"), recursive=True)
    seg = glob.glob(os.path.join(mask_root, "**", "*mask*label*.nii.gz"), recursive=True)
    # strip the two distinguishing suffixes down to e.g. 'dataset6_CLINIC_0060'
    def norm(paths, strip_pats):
        out = {}
        for p in paths:
            t = os.path.basename(p)
            for e in (".nii.gz", ".nii"):
                if t.endswith(e):
                    t = t[:-len(e)]; break
            for s in strip_pats:
                t = t.replace(s, "")
            out[t] = p
        return out
    ctm = norm(ct, ["_data"])
    sgm = norm(seg, ["_mask_4label", "_mask"])
    return [(ctm[t], sgm[t], t) for t in ctm if t in sgm]


def ctspine1k_pairs(vol_root, label_root):
    """Match CTSpine1K CT (raw_data/volumes/<SRC>/<UID>.nii.gz) ↔ seg
    (raw_data/labels/<SRC>/<UID>_seg.nii.gz) by the shared <UID> token.
    Volumes and labels live in disjoint trees, searched recursively."""
    ct = glob.glob(os.path.join(vol_root, "**", "*.nii.gz"), recursive=True)
    seg = glob.glob(os.path.join(label_root, "**", "*_seg.nii.gz"), recursive=True)
    return match_by_token(sorted(ct), sorted(seg), ct_strip="", seg_strip="_seg")


def ribseg_pairs(ct_root, seg_dir):
    """Match RibFrac CT (RibFracXXX-image.nii.gz, unzipped from ribfrac_ct) ↔
    RibSeg mask (RibFracXXX-rib-seg.nii.gz) by the shared 'RibFracXXX' token.

    RibFrac ships CT across several image zips; point ct_root at wherever they
    were extracted (searched recursively). seg_dir is ribseg_v2/seg/."""
    ct = glob.glob(os.path.join(ct_root, "**", "RibFrac*-image.nii.gz"), recursive=True)
    seg = glob.glob(os.path.join(seg_dir, "**", "RibFrac*-rib-seg.nii.gz"), recursive=True)
    return match_by_token(sorted(ct), sorted(seg),
                          ct_strip="-image", seg_strip="-rib-seg")


def write_pairs(pairs, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([[c, s, i] for c, s, i in pairs], f, indent=1)
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build a [ct,seg,id] pairs manifest.")
    ap.add_argument("--dataset", required=True,
                    choices=["totalseg", "ctpelvic1k", "ribseg", "ctspine1k", "generic"])
    ap.add_argument("--out", required=True)
    # totalseg
    ap.add_argument("--root", help="extracted dataset root")
    ap.add_argument("--combined", help="combine.py output dir (totalseg)")
    # ctpelvic1k
    ap.add_argument("--ct-root")
    ap.add_argument("--mask-root")
    # generic single-multilabel
    ap.add_argument("--ct-glob")
    ap.add_argument("--seg-glob")
    ap.add_argument("--ct-strip", default="")
    ap.add_argument("--seg-strip", default="")
    args = ap.parse_args()

    if args.dataset == "totalseg":
        pairs = totalseg_pairs(args.root, args.combined)
    elif args.dataset == "ctpelvic1k":
        pairs = ctpelvic1k_pairs(args.ct_root, args.mask_root)
    elif args.dataset == "ribseg":
        pairs = ribseg_pairs(args.ct_root, args.mask_root)
    elif args.dataset == "ctspine1k":
        pairs = ctspine1k_pairs(args.ct_root, args.mask_root)
    else:
        pairs = match_by_token(sorted(glob.glob(args.ct_glob)),
                               sorted(glob.glob(args.seg_glob)),
                               args.ct_strip, args.seg_strip)
    write_pairs(pairs, args.out)
    print(f"{args.dataset}: {len(pairs)} pairs → {args.out}")


if __name__ == "__main__":
    main()
