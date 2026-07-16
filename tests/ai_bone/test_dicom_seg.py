import numpy as np
from ai_bone.datasets.dicom_seg import segment_label_to_unified, combine_segments
from ai_bone import taxonomy_v1 as tx


def test_label_parses_taxonomy_vertebrae():
    assert segment_label_to_unified("T1 vertebra") == "T1"
    assert segment_label_to_unified("L5 vertebra") == "L5"
    assert segment_label_to_unified("C7 vertebra") == "C7"
    assert segment_label_to_unified("t12 VERTEBRA") == "T12"   # case-insensitive


def test_label_ignores_non_taxonomy_vertebrae():
    for lbl in ("L6 vertebra", "T13 vertebra", "S1 vertebra", "Sacrum"):
        assert segment_label_to_unified(lbl) == "__ignore__"


def test_label_none_for_non_vertebra():
    assert segment_label_to_unified("Metastatic lesion") is None
    assert segment_label_to_unified("") is None
    assert segment_label_to_unified("C8 vertebra") == "__ignore__"   # out-of-range vertebra


def test_combine_paints_ids_and_ignore():
    shape = (4, 4, 4)
    t1 = np.zeros(shape, bool); t1[0] = True
    l6 = np.zeros(shape, bool); l6[1] = True
    lesion = np.zeros(shape, bool); lesion[2] = True
    out = combine_segments([("T1 vertebra", t1), ("L6 vertebra", l6),
                            ("Metastatic lesion", lesion)])
    assert out[0].max() == tx.NAME_TO_ID["T1"] and (out[0] == tx.NAME_TO_ID["T1"]).all()
    assert (out[1] == tx.IGNORE_LABEL).all()          # L6 → ignore
    assert (out[2] == 0).all()                        # lesion segment dropped → background
    assert out.dtype == np.uint8


def test_combine_real_ids_win_over_ignore_on_overlap():
    shape = (2, 2, 2)
    t1 = np.ones(shape, bool)
    l6 = np.ones(shape, bool)                          # fully overlaps T1
    out = combine_segments([("L6 vertebra", l6), ("T1 vertebra", t1)])
    assert (out == tx.NAME_TO_ID["T1"]).all()         # real id beats IGNORE regardless of order


def test_combine_none_when_no_relevant_segments():
    shape = (2, 2, 2)
    assert combine_segments([("Metastatic lesion", np.ones(shape, bool))]) is None
