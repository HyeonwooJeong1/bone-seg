import numpy as np
from ai_bone.eval.metrics import dice, hd95

def test_dice_perfect_and_zero():
    a=np.zeros((4,4,4),int); a[1:3,1:3,1:3]=5
    assert dice(a,a,5)==1.0
    b=np.zeros((4,4,4),int)
    assert dice(a,b,5)==0.0

def test_hd95_zero_when_identical():
    a=np.zeros((5,5,5),int); a[2,2,2]=3
    assert hd95(a,a,3,(1,1,1))==0.0

def test_dice_absent_label_is_nan():
    a=np.zeros((4,4,4),int); b=np.zeros((4,4,4),int)
    v=dice(a,b,7)
    assert np.isnan(v)
