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
        "method": "zenodo", "record": "10047292", "verified": False,
        "landing_url": "https://zenodo.org/records/10047292",
        "notes": "TotalSegmentator v2 training data (1204). CONFIRM record id "
                 "on the landing page before download (CC BY 4.0).",
    },
    "verse": {
        "method": "zenodo", "record": "10159290", "verified": False,
        "landing_url": "https://github.com/anjany/verse",
        "notes": "VerSe'19/'20. Data hosted via the GitHub repo's links / OSF / "
                 "Zenodo — confirm the actual record before download.",
    },
    "ctspine1k": {
        "method": "manual", "verified": False,
        "landing_url": "https://github.com/MIRACLE-Center/CTSpine1K",
        "notes": "Images pulled from public sources (COLONOG, MSD, etc.) via "
                 "the repo's download scripts + Google Drive label links.",
    },
    "ribseg": {
        "method": "zenodo", "record": "3893508", "verified": False,
        "landing_url": "https://github.com/M3DV/RibSeg",
        "notes": "RibSeg v2 masks build on RibFrac CT (Zenodo). Confirm the "
                 "RibFrac image records + RibSeg label release.",
    },
    "ctpelvic1k": {
        "method": "zenodo", "record": "4588403", "verified": False,
        "landing_url": "https://github.com/MIRACLE-Center/CTPelvic1K",
        "notes": "Annotations + new clinical data via a Zenodo link; confirm id.",
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
