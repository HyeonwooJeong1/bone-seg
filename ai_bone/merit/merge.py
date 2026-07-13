import numpy as np

def _norm(weights):
    s = float(sum(weights))
    return [w / s for w in weights]

def weighted_average(state_dicts, weights):
    w = _norm(weights)
    out = {}
    for k in state_dicts[0]:
        acc = None
        for sd, wi in zip(state_dicts, w):
            term = sd[k] * wi
            acc = term if acc is None else acc + term
        out[k] = acc
    return out

def ties_merge(base, state_dicts, weights, density=0.2):
    w = _norm(weights)
    out = {}
    for k in base:
        taus = [sd[k] - base[k] for sd in state_dicts]        # task vectors
        trimmed = []
        for t in taus:
            a = np.abs(np.asarray(t))
            if a.size:
                thr = np.quantile(a, 1.0 - density)
                trimmed.append(np.where(a >= thr, np.asarray(t), 0.0))
            else:
                trimmed.append(np.asarray(t))
        stack = np.stack([tw * wi for tw, wi in zip(trimmed, w)])
        sign = np.sign(stack.sum(0))                          # elect sign
        agree = np.where(np.sign(stack) == sign, stack, 0.0)
        cnt = np.sum(np.sign(stack) == sign, axis=0)
        merged = np.where(cnt > 0, agree.sum(0) / np.maximum(cnt, 1), 0.0)
        out[k] = np.asarray(base[k]) + merged
    return out
