from ai_bone.download import download_dataset

class _Resp:
    status_code = 200
    def __init__(self, manifest): self._m = manifest
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self): pass
    def json(self): return self._m
    def iter_content(self, n): yield b"abc"

class _Sess:
    def __init__(self, manifest): self._m = manifest
    def get(self, url, stream=False, headers=None, timeout=None): return _Resp(self._m)

_MANIFEST = {"files": [{"key": "a.bin", "links": {"self": "https://x/a.bin"}, "size": 3}]}

def test_unverified_without_force_skips(tmp_path):
    # 'verse' is an unverified source → must short-circuit BEFORE any network/session.
    logs = []
    out = download_dataset("verse", str(tmp_path), force=False, logf=logs.append)
    assert out == []
    assert any("UNVERIFIED" in m for m in logs)

def test_manual_source_prints_instructions(tmp_path):
    # 'spinemets' is a manual source → prints instructions, no download/network.
    logs = []
    out = download_dataset("spinemets", str(tmp_path), force=True, logf=logs.append)
    assert out == []
    assert any("manual download" in m for m in logs)

def test_zenodo_force_downloads(tmp_path):
    sess = _Sess(_MANIFEST)
    out = download_dataset("totalseg", str(tmp_path), session=sess, force=True, logf=lambda *a: None)
    assert len(out) == 1
    p = tmp_path / "totalseg" / "a.bin"
    assert p.exists() and p.read_bytes() == b"abc"

def test_zenodo_multirecord_loops_all_records(tmp_path):
    # ribfrac_ct has 3 records; each fake record yields 1 file → 3 downloads.
    sess = _Sess(_MANIFEST)
    out = download_dataset("ribfrac_ct", str(tmp_path), session=sess, force=True, logf=lambda *a: None)
    assert len(out) == 3
    assert (tmp_path / "ribfrac_ct" / "a.bin").exists()
