import numpy as np

def pca_conflict_split(grad_vectors: dict, k: int = 2) -> dict:
    names = list(grad_vectors)
    X = np.stack([grad_vectors[n] for n in names]).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # 방향 정규화
    Xc = X - X.mean(0, keepdims=True)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt.T                        # (N, comps)
    if k <= 2:
        gid = (proj[:, 0] >= 0).astype(int)
    else:
        # k==3+: 제1축 부호 + 제2축 부호 사분면을 그룹으로 축약
        a = (proj[:, 0] >= 0).astype(int)
        b = (proj[:, 1] >= 0).astype(int) if proj.shape[1] > 1 else np.zeros(len(names), int)
        quad = a * 2 + b
        uniq = {q: i for i, q in enumerate(sorted(set(quad.tolist())))}
        gid = np.array([uniq[q] for q in quad])
    part = {}
    for name, g in zip(names, gid.tolist()):
        part.setdefault(int(g), []).append(name)
    return part
