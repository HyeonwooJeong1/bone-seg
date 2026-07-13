"""통합 전신 뼈 taxonomy v1 (축골격 + 골반대). 배경=0, 전경 53클래스."""

IGNORE_LABEL = 255

def _build():
    d = {0: "background", 1: "Skull"}
    nid = 2
    for i in range(1, 8):   d[nid] = f"C{i}";  nid += 1      # 2..8
    for i in range(1, 13):  d[nid] = f"T{i}";  nid += 1      # 9..20
    for i in range(1, 6):   d[nid] = f"L{i}";  nid += 1      # 21..25
    d[nid] = "Sacrum"; nid += 1                              # 26
    for i in range(1, 13):  d[nid] = f"Rib_L_{i}"; nid += 1  # 27..38
    for i in range(1, 13):  d[nid] = f"Rib_R_{i}"; nid += 1  # 39..50
    d[nid] = "Sternum"; nid += 1                             # 51
    d[nid] = "Hip_L"; nid += 1                               # 52
    d[nid] = "Hip_R"; nid += 1                               # 53
    return d

UNIFIED_V1 = _build()
NAME_TO_ID = {v: k for k, v in UNIFIED_V1.items()}
NUM_CLASSES = len(UNIFIED_V1)          # 54
FG_NAMES = [UNIFIED_V1[i] for i in range(1, NUM_CLASSES)]

def name_to_id(name): return NAME_TO_ID[name]
def id_to_name(i): return UNIFIED_V1[i]

def validate():
    assert NUM_CLASSES == 54, NUM_CLASSES
    assert len(set(UNIFIED_V1.values())) == NUM_CLASSES, "duplicate label name"
    assert set(UNIFIED_V1) == set(range(NUM_CLASSES)), "ids must be contiguous 0..53"
    assert IGNORE_LABEL not in UNIFIED_V1
