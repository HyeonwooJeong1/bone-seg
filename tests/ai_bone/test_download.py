from ai_bone.download import parse_zenodo_manifest, download_file

def test_parse_zenodo():
    rec = {"files":[{"key":"a.zip","links":{"self":"https://x/a.zip"},"size":10}]}
    out = parse_zenodo_manifest(rec)
    assert out == [{"name":"a.zip","url":"https://x/a.zip","size":10}]

class _FakeResp:
    status_code=200; headers={"content-length":"3"}
    def iter_content(self, n): yield b"abc"
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def raise_for_status(self): pass
class _FakeSession:
    def get(self, url, stream, headers, timeout): return _FakeResp()

def test_download_writes(tmp_path):
    dest = tmp_path/"a.bin"
    download_file("https://x/a.bin", dest, resume=False, session=_FakeSession())
    assert dest.read_bytes()==b"abc"
