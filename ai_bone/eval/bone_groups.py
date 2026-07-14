"""Taxonomy-derived bone groups for region-averaged reporting, left/right swap
evaluation, and difficulty-stratified analysis (see docs/experiment_design.md §4).
Built from taxonomy names so ids stay correct if the taxonomy changes."""
from ai_bone import taxonomy_v1 as tx

def _id(name):
    return tx.NAME_TO_ID[name]

SKULL     = [_id("Skull")]
CERVICAL  = [_id(f"C{i}") for i in range(1, 8)]
THORACIC  = [_id(f"T{i}") for i in range(1, 13)]
LUMBAR    = [_id(f"L{i}") for i in range(1, 6)]
SACRUM    = [_id("Sacrum")]
RIBS_L    = [_id(f"Rib_L_{i}") for i in range(1, 13)]
RIBS_R    = [_id(f"Rib_R_{i}") for i in range(1, 13)]
RIBS      = RIBS_L + RIBS_R
STERNUM   = [_id("Sternum")]
HIP       = [_id("Hip_L"), _id("Hip_R")]

VERTEBRAE = CERVICAL + THORACIC + LUMBAR          # individual vertebra instances
SPINE     = VERTEBRAE + SACRUM

# Region macro-groups for averaged reporting.
REGION_GROUPS = {
    "skull": SKULL, "cervical": CERVICAL, "thoracic": THORACIC, "lumbar": LUMBAR,
    "sacrum": SACRUM, "ribs": RIBS, "sternum": STERNUM, "pelvis": HIP,
}

# Left/Right pairs for swap-rate.
LR_PAIRS = [(_id(f"Rib_L_{i}"), _id(f"Rib_R_{i}")) for i in range(1, 13)] \
           + [(_id("Hip_L"), _id("Hip_R"))]

# Enumeration transition zones (off-by-one hotspots): (C7,T1), (T12,L1), (L5,Sacrum).
TRANSITION_ZONES = [(_id("C7"), _id("T1")), (_id("T12"), _id("L1")),
                    (_id("L5"), _id("Sacrum"))]

FLOATING_RIBS = [_id("Rib_L_11"), _id("Rib_L_12"), _id("Rib_R_11"), _id("Rib_R_12")]

# Difficulty strata → label-id lists (experiment_design §4.1).
DIFFICULTY_STRATA = {
    "thin_floating_ribs": FLOATING_RIBS,
    "individual_vertebrae": VERTEBRAE,
    "individual_ribs": RIBS,
    "lr_symmetric": [i for pair in LR_PAIRS for i in pair],
}
