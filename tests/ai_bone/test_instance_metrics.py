import numpy as np
from ai_bone.eval.instance_metrics import (
    instance_scores, localization_error, confusion_pairs, lr_swap_rate,
)

def _two_label_vol():
    # label 5 in block A, label 6 in block B
    v = np.zeros((10, 10, 10), int)
    v[1:4, 1:4, 1:4] = 5      # A
    v[6:9, 6:9, 6:9] = 6      # B
    return v

def test_instance_scores_perfect():
    gt = _two_label_vol()
    s = instance_scores(gt, gt, [5, 6])
    assert s["tp"] == 2 and s["fp"] == 0 and s["fn"] == 0
    assert s["id_rate"] == 1.0 and s["rq"] == 1.0
    assert s["sq"] > 0.99 and s["pq"] > 0.99

def test_instance_scores_missed_and_mislabeled():
    gt = _two_label_vol()
    pred = gt.copy()
    pred[pred == 6] = 7            # label 6 region predicted as an out-of-set id
    s = instance_scores(gt, pred, [5, 6])
    assert s["tp"] == 1 and s["fn"] == 1        # 5 hit, 6 missed
    assert s["id_rate"] == 0.5

def test_localization_error_zero_when_identical():
    gt = _two_label_vol()
    assert localization_error(gt, gt, [5, 6], (1, 1, 1)) == 0.0

def test_confusion_pairs_captures_mislabel():
    gt = _two_label_vol()
    pred = gt.copy()
    pred[gt == 5] = 6             # all of label-5 predicted as 6
    conf = confusion_pairs(gt, pred, [5, 6])
    assert conf[5].get(6, 0) > 0.99      # 5 confused as 6

def test_lr_swap_rate_detects_swap():
    gt = _two_label_vol()          # 5 = "left", 6 = "right"
    pred = np.zeros_like(gt)
    pred[gt == 5] = 6              # swap sides
    pred[gt == 6] = 5
    assert lr_swap_rate(gt, pred, [(5, 6)]) == 1.0
    assert lr_swap_rate(gt, gt, [(5, 6)]) == 0.0
