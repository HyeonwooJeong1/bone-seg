"""Per-dataset download sources for the v1 pipeline.

CAUTION: record IDs / URLs below are BEST-KNOWN but NOT yet verified against a
real download. Every entry has `verified: False`; the download CLI refuses to
fetch an unverified source unless `--force` is given, and always prints the
`landing_url` so a human can confirm the correct record before downloading.

`method` semantics:
  - "zenodo"    : automatable. `record` = Zenodo record id; the CLI fetches
                  https://zenodo.org/api/records/<record> and downloads its files.
  - "manual"    : needs auth / a special client (TCIA NBIA, HuggingFace hub,
                  Google Drive, Figshare) → the CLI prints instructions, does
                  NOT attempt an automatic download.
"""

SOURCES = {
    # Pretrain — HuggingFace dataset (needs `huggingface_hub`), large.
    "cads": {
        "method": "manual", "verified": False,
        "landing_url": "https://huggingface.co/datasets/mrmrx/CADS-dataset",
        "notes": "HuggingFace hub. Axial-bone subset only needed for pretrain; "
                 "consider subsampling (~3-5k scans) — full 22k not required.",
    },
    # Expert GT (fine-tune)
    "totalseg": {
        "method": "zenodo", "record": "10047292", "verified": True,
        "landing_url": "https://zenodo.org/records/10047292",
        "notes": "TotalSegmentatorV2: 1228 CT, 117 structures, CC BY 4.0. Record "
                 "id confirmed. The internal bone label VALUES are still confirmed "
                 "at build time via the verify_dataset overlap gate.",
    },
    "verse": {
        "method": "manual", "verified": False,
        "landing_url": "https://github.com/anjany/verse",
        "urls": ["https://osf.io/nqjyw/", "https://osf.io/t98fz/"],
        "notes": "3D CT+masks are on OSF, NOT Zenodo: VerSe'19 = osf.io/nqjyw, "
                 "VerSe'20 = osf.io/t98fz (rawdata=CT .nii.gz, derivatives=seg "
                 "masks + centroid .json). Zenodo 3759104 is only the challenge "
                 "doc and 8115942 is 2D projections — neither is the volume data. "
                 "Download via `osfclient` or the browser.",
    },
    "ctspine1k": {
        "method": "manual", "verified": False,
        "landing_url": "https://github.com/MIRACLE-Center/CTSpine1K",
        "notes": "Images pulled from public sources (COLONOG, MSD, etc.) via "
                 "the repo's download scripts + Google Drive label links.",
    },
    "ribseg": {
        "method": "manual", "verified": False,
        "landing_url": "https://github.com/HINTLab/RibSeg",
        "ribfrac_records": ["3893508", "3893498", "3893496"],
        "notes": "Two-part source. (1) CT images = RibFrac on Zenodo — train "
                 "3893508 (part1, 300) + 3893498 (part2, 120) + 3893496 (val, 80); "
                 "these ARE zenodo-automatable if you add them as separate entries. "
                 "(2) RIB SEGMENTATION masks (what we map to Rib_L/R): RibSeg v2 = "
                 "Google Drive (link in HINTLab/RibSeg README); RibSeg v1 = Zenodo "
                 "5336592. Masks are NIfTI (512,512,N) rib labels, NOT RibFrac's "
                 "fracture labels. GDrive → use `gdown`.",
    },
    "ctpelvic1k": {
        "method": "zenodo", "record": "4588403", "verified": True,
        "landing_url": "https://zenodo.org/records/4588403",
        "notes": "CTPelvic1K annotations + clinical data (sacrum/L+R hip; lumbar "
                 "grouped→ignore). Record id confirmed.",
    },
    "spinemets": {
        "method": "manual", "verified": False,
        "landing_url": "https://doi.org/10.7937/kh36-ds04",
        "notes": "TCIA — download via NBIA Data Retriever (DICOM + DICOM-SEG).",
    },
    "mug500": {
        "method": "manual", "verified": False,
        "landing_url": "https://figshare.com/articles/dataset/MUG500_/9616319",
        "notes": "Figshare — 500 skull segmentations.",
    },
}

_ALLOWED_METHODS = {"zenodo", "manual"}


def get_source(name):
    if name not in SOURCES:
        raise KeyError(f"no download source registered for dataset {name!r}")
    return SOURCES[name]
