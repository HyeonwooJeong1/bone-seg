"""Merge several per-dataset nnU-Net raw datasets into one fine-tune dataset.

Each expert dataset was built into its own raw (e.g. Dataset512_TotalSeg,
Dataset511_CTPelvic1K) with unified-taxonomy labels + per-case `present_labels`
sidecars. This step assembles them into one Dataset510_AxialFT:
  - case ids are prefixed `dataset__case` so ids never collide across sources,
  - files are hard-linked (no copy) when on the same filesystem,
  - present sidecars are merged, and a `case_datasets.json` maps case→dataset
    (used by MERIT conflict estimation to group gradients per source),
  - dataset.json is written with numTraining = total and the unified labels.

NOTE (partial labels): datasets that annotate only some bones (e.g. CTPelvic1K =
pelvis only) carry a `present_labels` subset. Non-annotated classes are currently
background in the seg; a marginal/ignore-aware loss at training time should use
`present_labels` so such cases don't teach "spine = background". The sidecars +
case_datasets.json preserve exactly the info that loss needs.
"""
import glob
import json
import os
import shutil

from ai_bone.build_raw import write_dataset_json


def _link_or_copy(src, dst, link=True):
    if os.path.exists(dst):
        os.remove(dst)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def merge_raw(sources, out_dir, link=True, logf=print):
    """sources: {dataset_name: raw_dir}. Returns {total, present_union, per_dataset}."""
    img_out = os.path.join(out_dir, "imagesTr")
    lab_out = os.path.join(out_dir, "labelsTr")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lab_out, exist_ok=True)
    present_union, case2ds, per_dataset, total = set(), {}, {}, 0
    for name, src in sources.items():
        n = 0
        for ip in sorted(glob.glob(os.path.join(src, "imagesTr", "*_0000.nii.gz"))):
            base = os.path.basename(ip)[: -len("_0000.nii.gz")]
            lp = os.path.join(src, "labelsTr", base + ".nii.gz")
            if not os.path.exists(lp):
                continue
            newcase = f"{name}__{base}"
            _link_or_copy(ip, os.path.join(img_out, newcase + "_0000.nii.gz"), link)
            _link_or_copy(lp, os.path.join(lab_out, newcase + ".nii.gz"), link)
            sp = os.path.join(src, "labelsTr", base + ".present.json")
            if os.path.exists(sp):
                pl = json.loads(open(sp, encoding="utf-8").read())["present_labels"]
                with open(os.path.join(lab_out, newcase + ".present.json"), "w",
                          encoding="utf-8") as f:
                    json.dump({"present_labels": pl}, f)
                present_union |= set(pl)
            case2ds[newcase] = name
            n += 1
            total += 1
        per_dataset[name] = n
        logf(f"[{name}] merged {n} cases")
    write_dataset_json(out_dir, total, present_union)
    with open(os.path.join(out_dir, "case_datasets.json"), "w", encoding="utf-8") as f:
        json.dump(case2ds, f, indent=1)
    logf(f"merged {total} cases from {len(sources)} datasets → {out_dir}")
    return {"total": total, "present_union": present_union, "per_dataset": per_dataset}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Merge per-dataset raws into one FT dataset.")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="name=raw_dir entries, e.g. totalseg=/data1/.../Dataset512_TotalSeg")
    ap.add_argument("--out", required=True, help="output merged nnUNet_raw dir")
    ap.add_argument("--copy", action="store_true", help="copy instead of hard-link")
    args = ap.parse_args()
    sources = {}
    for s in args.sources:
        name, path = s.split("=", 1)
        sources[name] = path
    res = merge_raw(sources, args.out, link=not args.copy)
    print(f"total={res['total']} per_dataset={res['per_dataset']}")


if __name__ == "__main__":
    main()
