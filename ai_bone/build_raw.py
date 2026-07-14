import json, os
from ai_bone import taxonomy_v1 as tx

def write_dataset_json(out_dir, num_training: int, present_union) -> dict:
    labels = {name: i for i, name in tx.UNIFIED_V1.items()}   # background..Hip_R
    labels["ignore"] = tx.IGNORE_LABEL
    d = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return d

def write_present_sidecar(out_dir, case_id: str, present_labels) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{case_id}.present.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"present_labels": list(present_labels)}, f)
    return p


def _process_one_case(task):
    """Multiprocessing worker (real SimpleITK IO). Picklable: takes paths, not images.
    task = (ct_path, seg_path, cid, lm, images_dir, labels_dir, spacing, hu_thr).
    Returns (cid, ok: bool, payload) — payload = present_labels(list) if ok else report."""
    import os as _os
    import SimpleITK as sitk
    from ai_bone.harmonize import harmonize_case
    from ai_bone.verify_dataset import verify_case, is_pass
    from ai_bone.nifti_io import read_sitk
    ct_path, seg_path, cid, lm, images_dir, labels_dir, spacing, hu_thr, overlap_thr = task
    try:
        out_ct, out_seg = harmonize_case(read_sitk(ct_path), read_sitk(seg_path),
                                         lm, spacing_mm=spacing)
    except Exception as e:                           # unreadable / geometry failure
        return (cid, False, {"error": str(e)})
    rep = verify_case(out_ct, out_seg, hu_thr=hu_thr)
    if not is_pass(rep, overlap_thr):
        return (cid, False, rep)
    sitk.WriteImage(out_ct, _os.path.join(images_dir, f"{cid}_0000.nii.gz"))
    sitk.WriteImage(sitk.Cast(out_seg, sitk.sitkUInt8), _os.path.join(labels_dir, f"{cid}.nii.gz"))
    write_present_sidecar(labels_dir, cid, list(lm.present_labels))
    return (cid, True, list(lm.present_labels))


def build_from_pairs(pairs, lm, raw_dir, spacing=0.6, hu_thr=200,
                     reader=None, writer=None, logf=print, workers=1,
                     overlap_thr=0.5) -> dict:
    """End-to-end build of one nnU-Net raw dataset from explicit (ct, seg, id) pairs.

    For each pair: read CT+seg → `harmonize_case` (remap+align+isotropic) →
    `verify_case` gate → write imagesTr/<id>_0000.nii.gz + labelsTr/<id>.nii.gz +
    present sidecar. Cases failing the gate are skipped and logged (not written).
    Finally writes dataset.json. `pairs` are EXPLICIT (id, ct, seg) so the
    wrong-CT-file bug (multiple CTs in one folder) cannot occur here — discovery
    is the caller's responsibility.

    workers>1 fans the (CPU-heavy) per-case harmonize/verify/write across a process
    pool — use it on the many-core server. The parallel path always uses real
    SimpleITK IO, so it is taken only when reader/writer are NOT injected (tests
    inject them and run the sequential path).

    Returns {"written": int, "skipped": [(id, report)], "present_union": set}.
    """
    import SimpleITK as sitk
    from ai_bone.harmonize import harmonize_case
    from ai_bone.verify_dataset import verify_case, is_pass
    images_dir = os.path.join(raw_dir, "imagesTr")
    labels_dir = os.path.join(raw_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    present_union, skipped, n_ok = set(), [], 0

    if workers and workers > 1 and reader is None and writer is None:
        from multiprocessing import Pool
        tasks = [(cp, sp, cid, lm, images_dir, labels_dir, spacing, hu_thr, overlap_thr)
                 for cp, sp, cid in pairs]
        with Pool(workers) as pool:
            for cid, ok, payload in pool.imap_unordered(_process_one_case, tasks):
                if ok:
                    n_ok += 1; present_union |= set(payload)
                else:
                    skipped.append((cid, payload)); logf(f"[{cid}] SKIP: {payload}")
    else:
        from ai_bone.nifti_io import read_sitk
        read = reader or read_sitk
        write = writer or (lambda img, p: sitk.WriteImage(img, p))
        for ct_path, seg_path, cid in pairs:
            try:
                out_ct, out_seg = harmonize_case(read(ct_path), read(seg_path), lm, spacing_mm=spacing)
            except Exception as e:                   # unreadable / geometry failure
                logf(f"[{cid}] ERROR harmonize: {e}"); skipped.append((cid, {"error": str(e)})); continue
            rep = verify_case(out_ct, out_seg, hu_thr=hu_thr)
            if not is_pass(rep, overlap_thr):
                logf(f"[{cid}] SKIP (verify): empty={rep['empty']} size_match={rep['size_match']} "
                     f"overlap={rep['overlap_ratio']:.2f}")
                skipped.append((cid, rep)); continue
            write(out_ct, os.path.join(images_dir, f"{cid}_0000.nii.gz"))
            write(sitk.Cast(out_seg, sitk.sitkUInt8), os.path.join(labels_dir, f"{cid}.nii.gz"))
            write_present_sidecar(labels_dir, cid, lm.present_labels)
            present_union |= set(lm.present_labels)
            n_ok += 1
    write_dataset_json(raw_dir, n_ok, present_union)
    logf(f"built {raw_dir}: {n_ok} written, {len(skipped)} skipped")
    return {"written": n_ok, "skipped": skipped, "present_union": present_union}


def _load_pairs(pairs_json):
    """Read a pairs manifest: JSON list of [ct_path, seg_path, case_id]."""
    data = json.loads(open(pairs_json, encoding="utf-8").read())
    return [(row[0], row[1], row[2]) for row in data]


def main():
    import argparse
    from ai_bone.label_map import load_label_map
    from ai_bone.datasets.registry import DATASETS
    ap = argparse.ArgumentParser(description="Build an nnU-Net raw dataset (GPU-free).")
    ap.add_argument("--pairs", required=True,
                    help="JSON manifest: list of [ct_path, seg_path, case_id]")
    ap.add_argument("--dataset", required=True,
                    help="registered dataset name (selects its label_map.json)")
    ap.add_argument("--out", required=True, help="output nnUNet_raw dataset dir")
    ap.add_argument("--spacing", type=float, default=0.6)
    ap.add_argument("--hu-thr", type=int, default=200)
    ap.add_argument("--overlap-thr", type=float, default=0.5,
                    help="verify overlap gate (lower for whole-bone/marrow masks, "
                         "e.g. 0.25 for TotalSeg; ~0 catches gross misalignment)")
    # Shared server: default to a modest core count and leave the rest for others.
    # The box has ~124 cores; do NOT crank this near that. 16 is a polite default.
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel CPU workers for the build (keep well below the "
                         "machine's core count on a shared server)")
    args = ap.parse_args()
    lm = load_label_map(DATASETS[args.dataset].label_map_path)
    res = build_from_pairs(_load_pairs(args.pairs), lm, args.out,
                           spacing=args.spacing, hu_thr=args.hu_thr, workers=args.workers,
                           overlap_thr=args.overlap_thr)
    print(f"written={res['written']} skipped={len(res['skipped'])}")


if __name__ == "__main__":
    main()
