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

from ai_bone.eval.metrics import nsd, assd

def test_nsd_identical_is_one():
    a=np.zeros((10,10,10),int); a[3:7,3:7,3:7]=5
    assert nsd(a,a,5,(1,1,1),tau=3.0)==1.0
    assert assd(a,a,5,(1,1,1))==0.0

def test_nsd_disjoint_far_is_zero():
    a=np.zeros((24,24,24),int); a[2:5,2:5,2:5]=5
    b=np.zeros((24,24,24),int); b[18:21,18:21,18:21]=5
    assert nsd(a,b,5,(1,1,1),tau=3.0)==0.0

def test_nsd_absent_semantics():
    a=np.zeros((6,6,6),int); a[2:4,2:4,2:4]=5
    empty=np.zeros((6,6,6),int)
    assert np.isnan(nsd(empty,empty,5,(1,1,1)))     # both absent
    assert nsd(a,empty,5,(1,1,1))==0.0              # one absent
