import os
from pathlib import Path

def parse_zenodo_manifest(record_json: dict):
    out = []
    for f in record_json.get("files", []):
        out.append({"name": f["key"], "url": f["links"]["self"], "size": f.get("size")})
    return out

def download_dataset(name, dest_root, session=None, force=False, logf=print,
                     allow_patterns=None, max_workers=None):
    """Download one dataset's files per its registered source.

    - "zenodo": fetch the record manifest and download every file (resumable).
    - "manual": print instructions and return without downloading.
    Refuses an unverified source unless `force=True` (prints the landing URL so
    a human can confirm the record first). Returns a list of downloaded paths.
    """
    from ai_bone.datasets.sources import get_source
    src = get_source(name)
    landing = src.get("landing_url", "?")
    if not src.get("verified") and not force:
        logf(f"[{name}] source UNVERIFIED — confirm at {landing} then re-run "
             f"with --force. Notes: {src.get('notes','')}")
        return []
    if src["method"] == "manual":
        logf(f"[{name}] manual download required (auth/special client).\n"
             f"       Landing: {landing}\n       Notes: {src.get('notes','')}")
        return []
    if src["method"] == "huggingface":
        import os
        from huggingface_hub import snapshot_download   # server dep
        logf(f"[{name}] huggingface snapshot_download {src['repo_id']} ...")
        path = snapshot_download(
            repo_id=src["repo_id"], repo_type="dataset",
            local_dir=os.path.join(dest_root, name),
            allow_patterns=allow_patterns or src.get("allow_patterns"),
            max_workers=max_workers or 4,   # fewer concurrent HEADs → fewer 429s
        )
        return [str(path)]
    if src["method"] == "gdrive":
        import os
        import gdown                                   # server dep
        dest = os.path.join(dest_root, name)
        os.makedirs(dest, exist_ok=True)
        out = []
        for fid in (src.get("file_ids") or [src["file_id"]]):
            logf(f"[{name}] gdrive file {fid} → {dest}")
            out.append(str(gdown.download(id=fid, output=dest + os.sep, quiet=False)))
        return out
    if src["method"] == "osf":
        import os, subprocess
        out = []
        for proj in src["osf_projects"]:
            d = os.path.join(dest_root, name, proj)
            os.makedirs(d, exist_ok=True)
            logf(f"[{name}] osf clone {proj} → {d}")
            subprocess.run(["osf", "-p", proj, "clone", d], check=True)
            out.append(d)
        return out
    if src["method"] == "zenodo":
        import os
        if session is None:
            import requests
            session = requests.Session()
        records = src.get("records") or [src["record"]]   # one or many
        out = []
        for rec in records:
            rec_url = f"https://zenodo.org/api/records/{rec}"
            with session.get(rec_url, stream=False, headers={}, timeout=60) as r:
                r.raise_for_status()
                record_json = r.json()
            for entry in parse_zenodo_manifest(record_json):
                dest = os.path.join(dest_root, name, entry["name"])
                logf(f"[{name}] record {rec}: downloading {entry['name']} ...")
                out.append(str(download_file(entry["url"], dest, session=session,
                                             expected_size=entry.get("size"))))
        return out
    logf(f"[{name}] unknown method {src['method']!r}")
    return []


def download_file(url, dest, resume=True, session=None, chunk=1 << 20,
                  expected_size=None, max_retries=6):
    """Resumable download that RETRIES on dropped connections (large Zenodo zips
    routinely break mid-stream). Resumes from the partial file via HTTP Range;
    if `expected_size` is known, verifies completeness and skips already-complete
    files (avoids a 416 error when re-run)."""
    import time
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if session is None:
        import requests
        session = requests.Session()
    last_err = None
    for attempt in range(max_retries):
        pos = dest.stat().st_size if (resume and dest.exists()) else 0
        if expected_size is not None and pos >= expected_size:
            return dest                       # already complete
        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            with session.get(url, stream=True, headers=headers, timeout=60) as r:
                if getattr(r, "status_code", 200) == 416:
                    return dest               # range not satisfiable → complete
                r.raise_for_status()
                # Append only when the server honored Range (206); a 200 means it
                # ignored Range and sent the full body, so truncate to avoid dup.
                mode = "ab" if (pos and getattr(r, "status_code", 206) == 206) else "wb"
                with open(dest, mode) as f:
                    for c in r.iter_content(chunk):
                        if c: f.write(c)
            if expected_size is None or dest.stat().st_size >= expected_size:
                return dest                   # done (or size unknown → assume ok)
            last_err = f"incomplete {dest.stat().st_size}/{expected_size}"
        except Exception as e:                # dropped connection etc. → resume-retry
            last_err = e
        if attempt < max_retries - 1:
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"download failed after {max_retries} tries: {url} ({last_err})")


def main():
    import argparse
    from ai_bone.datasets.sources import SOURCES
    ap = argparse.ArgumentParser(description="Download a dataset's files (GPU-free).")
    ap.add_argument("name", help="dataset name or 'all'")
    ap.add_argument("--dest", default="ai_bone/data", help="destination root dir")
    ap.add_argument("--force", action="store_true",
                    help="download even if the source is marked unverified")
    ap.add_argument("--allow", nargs="*", default=None,
                    help="huggingface allow_patterns (subsample, e.g. a shard glob)")
    ap.add_argument("--max-workers", type=int, default=None,
                    help="huggingface concurrent workers (lower = fewer 429s)")
    args = ap.parse_args()
    names = list(SOURCES) if args.name == "all" else [args.name]
    for n in names:
        download_dataset(n, args.dest, force=args.force, allow_patterns=args.allow,
                         max_workers=args.max_workers)


if __name__ == "__main__":
    main()
