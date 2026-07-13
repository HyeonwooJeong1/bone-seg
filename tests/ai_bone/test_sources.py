from ai_bone.datasets import registry
from ai_bone.datasets.sources import SOURCES, get_source, _ALLOWED_METHODS

def test_every_registered_dataset_has_a_source():
    for name in registry.DATASETS:
        assert name in SOURCES, f"missing download source for {name}"

def test_source_entries_well_formed():
    for name, src in SOURCES.items():
        assert src["method"] in _ALLOWED_METHODS
        assert isinstance(src["verified"], bool)
        assert src.get("landing_url")
        if src["method"] == "zenodo":
            assert src.get("record") or src.get("records")

def test_ribfrac_ct_is_auto_multirecord():
    src = SOURCES["ribfrac_ct"]
    assert src["method"] == "zenodo" and src["verified"] is True
    assert src["records"] == ["3893508", "3893498", "3893496"]

def test_cads_is_huggingface():
    src = SOURCES["cads"]
    assert src["method"] == "huggingface" and src.get("repo_id") == "mrmrx/CADS-dataset"

def test_get_source_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_source("not_a_dataset")
