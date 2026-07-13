from ai_bone import taxonomy_v1 as tx

def test_counts():
    assert tx.NUM_CLASSES == 54          # bg + 53 fg
    assert len(tx.FG_NAMES) == 53
    assert tx.IGNORE_LABEL == 255

def test_roundtrip_unique():
    # id<->name 왕복 + 중복 없음
    for i, name in tx.UNIFIED_V1.items():
        assert tx.name_to_id(name) == i
        assert tx.id_to_name(i) == name
    assert len(set(tx.UNIFIED_V1.values())) == len(tx.UNIFIED_V1)

def test_key_labels_present():
    for n in ["Skull","C1","C7","T1","T12","L1","L5","Sacrum",
              "Rib_L_1","Rib_R_12","Sternum","Hip_L","Hip_R"]:
        assert n in tx.NAME_TO_ID

def test_validate_ok():
    tx.validate()  # raises on any inconsistency
