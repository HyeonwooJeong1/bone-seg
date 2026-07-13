import hashlib
import numpy as np
import SimpleITK as sitk

def _grid(img, box=8):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    zi = np.linspace(0, arr.shape[0]-1, box).astype(int)
    yi = np.linspace(0, arr.shape[1]-1, box).astype(int)
    xi = np.linspace(0, arr.shape[2]-1, box).astype(int)
    g = arr[np.ix_(zi,yi,xi)]
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel()

def image_fingerprint(img, box=8) -> str:
    q = np.round(_grid(img, box), 2).tobytes()
    return hashlib.sha1(q).hexdigest()

def find_duplicates(items, thresh=0.98, box=8):
    vecs = {k: _grid(im, box) for k, im in items}
    keys = list(vecs); used=set(); groups=[]
    for i,k in enumerate(keys):
        if k in used: continue
        grp=[k]
        for k2 in keys[i+1:]:
            if k2 in used: continue
            c = float(np.corrcoef(vecs[k], vecs[k2])[0,1])
            if c >= thresh: grp.append(k2); used.add(k2)
        used.add(k); groups.append(grp)
    return groups
