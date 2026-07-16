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
        "method": "huggingface", "repo_id": "mrmrx/CADS-dataset", "verified": True,
        "landing_url": "https://huggingface.co/datasets/mrmrx/CADS-dataset",
        # allow_patterns subsamples the snapshot. None = download EVERYTHING (very
        # large — 22k scans). For pretrain a subset is enough; set a pattern here
        # or pass --allow on the CLI (e.g. a shard/subject-range glob).
        "allow_patterns": None,
        "notes": "HuggingFace (needs `huggingface_hub`). ⚠ full set is very large "
                 "(22k scans) — pretrain only needs ~3-5k, so subsample via "
                 "allow_patterns / --allow before pulling everything.",
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
        "method": "http", "verified": True,
        "urls": [
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip",
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip",
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip",
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20training.zip",
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20validation.zip",
            "https://s3.bonescreen.de/public/VerSe-complete/dataset-verse20test.zip",
        ],
        "landing_url": "https://github.com/anjany/verse",
        "notes": "VerSe'19+'20 COMPLETE (CT .nii.gz + seg masks + centroid .json, "
                 "BIDS layout) via direct bonescreen S3 zips — the OSF projects put "
                 "the big data behind external storage that osfclient can't fetch "
                 "(403). CC BY-SA 4.0. Vertebra labels 1-28 (25=L6, 26=sacrum, "
                 "28=T13); confirm exact set at build time.",
    },
    "ctspine1k": {
        "method": "huggingface", "repo_id": "alexanderdann/CTSpine1K",
        "verified": True,
        "landing_url": "https://huggingface.co/datasets/alexanderdann/CTSpine1K",
        # Pull only the raw NIfTI tree (raw_data/ = images + labels/*_seg.nii.gz,
        # 1005 cases ~150GB) + metadata (split file). Skip the repo's Arrow
        # export which the default snapshot would otherwise fetch.
        "allow_patterns": ["raw_data/**", "metadata/**"],
        "notes": "CTSpine1K on HuggingFace (public, not gated). 1005 CT + vertebra "
                 "seg (.nii.gz), 512x512xN. Top dir is raw_data/ (underscore): "
                 "images + labels/<SRC>/*_seg.nii.gz across COLONOG/HNSCC/MSD/COVID. "
                 "metadata/data_split.txt gives train/val/test. Restrict to "
                 "raw_data/** + metadata/** to avoid the large Arrow export.",
    },
    # RibFrac CT images (auto). Confirmed Zenodo records. NOTE: each record also
    # ships ribfrac-*-labels.zip = FRACTURE labels (ignore those); the rib
    # SEGMENTATION masks we need are the separate RibSeg release (see "ribseg").
    "ribfrac_ct": {
        "method": "zenodo",
        "records": ["3893508", "3893498", "3893496"], "verified": True,
        "landing_url": "https://zenodo.org/records/3893508",
        "notes": "RibFrac CT: 3893508 train-p1 (300) + 3893498 train-p2 (120) + "
                 "3893496 val (80). Use ribfrac-train-images-*.zip; the rib "
                 "segmentation masks come from RibSeg (dataset 'ribseg').",
    },
    "ribseg": {
        "method": "gdrive", "file_id": "1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm",
        "verified": True,
        "landing_url": "https://github.com/HINTLab/RibSeg",
        "notes": "RibSeg v2 RIB SEGMENTATION masks (NIfTI 512x512xN rib labels) via "
                 "Google Drive. CT images come from RibFrac (dataset 'ribfrac_ct', "
                 "already auto-downloaded). Pair the two for build.",
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
        "notes": "SKIP recommended: STL surface meshes (not voxel masks), ~210GB, "
                 "only contributes the single 'Skull' label which TotalSeg already "
                 "annotates. Not worth voxelizing for v1. Figshare art. 9616319.",
    },
}

_ALLOWED_METHODS = {"zenodo", "huggingface", "gdrive", "osf", "http", "manual"}


def get_source(name):
    if name not in SOURCES:
        raise KeyError(f"no download source registered for dataset {name!r}")
    return SOURCES[name]
