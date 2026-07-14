import numpy as np
from ai_bone import taxonomy_v1 as tx
from ai_bone.eval import bone_groups as bg
from ai_bone.eval.evaluate import region_summary, difficulty_summary

def test_group_sizes():
    assert len(bg.CERVICAL) == 7
    assert len(bg.THORACIC) == 12
    assert len(bg.LUMBAR) == 5
    assert len(bg.RIBS) == 24
    assert len(bg.VERTEBRAE) == 24            # C+T+L, no sacrum
    assert len(bg.LR_PAIRS) == 13             # 12 rib pairs + hips
    assert bg.FLOATING_RIBS == [tx.NAME_TO_ID[n] for n in
                                ("Rib_L_11", "Rib_L_12", "Rib_R_11", "Rib_R_12")]

def test_transition_zones_are_enumeration_boundaries():
    names = [(tx.id_to_name(a), tx.id_to_name(b)) for a, b in bg.TRANSITION_ZONES]
    assert names == [("C7", "T1"), ("T12", "L1"), ("L5", "Sacrum")]

def test_region_and_difficulty_summary_macro_average():
    per_class = {n: {"dice": 0.9, "nsd": 0.8, "hd95": 2.0} for n in tx.FG_NAMES}
    reg = region_summary(per_class, "dice")
    assert set(reg) == set(bg.REGION_GROUPS)
    assert abs(reg["ribs"] - 0.9) < 1e-9
    diff = difficulty_summary(per_class, "nsd")
    assert abs(diff["individual_vertebrae"] - 0.8) < 1e-9
