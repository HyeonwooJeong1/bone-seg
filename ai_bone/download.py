import os
from pathlib import Path

def parse_zenodo_manifest(record_json: dict):
    out = []
    for f in record_json.get("files", []):
        out.append({"name": f["key"], "url": f["links"]["self"], "size": f.get("size")})
    return out

def download_dataset(name, dest_root, session=None, force=False, logf=print,
                     allow_patterns=None):
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
        )
        return [str(path)]
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
                out.append(str(download_file(entry["url"], dest, session=session)))
        return out
    logf(f"[{name}] unknown method {src['method']!r}")
    return []


def download_file(url, dest, resume=True, session=None, chunk=1 << 20):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if session is None:
        import requests
        session = requests.Session()
    pos = dest.stat().st_size if (resume and dest.exists()) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with session.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        # Append only when the server honored the Range request (206 Partial).
        # If it ignored Range and returned the full body (200), truncate to
        # avoid doubling the already-downloaded prefix.
        mode = "ab" if (pos and getattr(r, "status_code", 206) == 206) else "wb"
        with open(dest, mode) as f:
            for c in r.iter_content(chunk):
                if c: f.write(c)
    return dest


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
    args = ap.parse_args()
    names = list(SOURCES) if args.name == "all" else [args.name]
    for n in names:
        download_dataset(n, args.dest, force=args.force, allow_patterns=args.allow)


if __name__ == "__main__":
    main()
