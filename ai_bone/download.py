import os
from pathlib import Path

def parse_zenodo_manifest(record_json: dict):
    out = []
    for f in record_json.get("files", []):
        out.append({"name": f["key"], "url": f["links"]["self"], "size": f.get("size")})
    return out

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
