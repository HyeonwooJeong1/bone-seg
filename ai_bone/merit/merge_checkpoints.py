"""Merge trained nnU-Net branch checkpoints into one model (MERIT weight merge).

Each branch was trained from the same shared init (CADS pretrain) on its own
dataset group, so all checkpoints share the architecture / key set. We merge the
`network_weights` state dict by case-weighted average (default) or TIES, keep the
rest of the checkpoint structure, and write a checkpoint nnU-Net can load for
inference/eval.

Dtype-safe: only floating-point tensors are averaged; integer/bool buffers (if
any) are copied from the first checkpoint. torch is imported lazily so the module
imports in a torch-free env; the merge is unit-tested in ct_env (torch present).
"""
import json


def _weights_norm(weights):
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("merge weights must sum to > 0")
    return [w / s for w in weights]


def merge_state_dicts_weighted(state_dicts, weights):
    """Case/token-weighted average of matching torch state dicts (dtype-safe)."""
    import torch
    w = _weights_norm(weights)
    keys = list(state_dicts[0])
    for sd in state_dicts[1:]:
        if list(sd) != keys:
            raise ValueError("state dicts have mismatched keys (different architectures?)")
    out = {}
    for k in keys:
        t0 = state_dicts[0][k]
        if not torch.is_floating_point(t0):
            out[k] = t0.clone()                       # int/bool buffer → take first
            continue
        acc = None
        for sd, wi in zip(state_dicts, w):
            term = sd[k].to(torch.float64) * wi
            acc = term if acc is None else acc + term
        out[k] = acc.to(t0.dtype)
    return out


def merge_state_dicts_ties(base_sd, state_dicts, weights, density=0.2):
    """TIES merge of matching torch state dicts around a base (shared init)."""
    import torch
    from ai_bone.merit.merge import ties_merge
    merged_np = ties_merge(base_sd, state_dicts, weights, density=density)
    out = {}
    for k, v in merged_np.items():
        out[k] = torch.as_tensor(v).to(state_dicts[0][k].dtype)
    return out


def merge_checkpoints(paths, weights, method="weighted", base_path=None,
                      density=0.2, weights_key="network_weights"):
    """Load nnU-Net checkpoints at `paths`, merge their `weights_key` state dicts,
    and return a checkpoint dict (structure copied from the first, weights merged)."""
    import torch
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    sds = [c[weights_key] for c in ckpts]
    if method == "weighted":
        merged = merge_state_dicts_weighted(sds, weights)
    elif method == "ties":
        if base_path is None:
            raise ValueError("ties merge needs base_path (the shared-init checkpoint)")
        base_sd = torch.load(base_path, map_location="cpu", weights_only=False)[weights_key]
        merged = merge_state_dicts_ties(base_sd, sds, weights, density=density)
    else:
        raise ValueError(f"unknown method {method!r}")
    out = dict(ckpts[0])                       # keep epoch/plans/trainer_name/etc.
    out[weights_key] = merged
    return out


def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser(description="Merge nnU-Net branch checkpoints (GPU-free).")
    ap.add_argument("--out", required=True, help="output merged checkpoint .pth")
    ap.add_argument("--method", choices=["weighted", "ties"], default="weighted")
    ap.add_argument("--base", default=None, help="shared-init checkpoint (ties)")
    ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--artifact", default=None,
                    help="orchestrate.py artifact json (reads merge_weights); "
                         "otherwise pass --weights")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("checkpoints", nargs="+", help="branch checkpoint .pth paths (in branch order)")
    args = ap.parse_args()

    weights = args.weights
    if args.artifact:
        weights = json.loads(open(args.artifact, encoding="utf-8").read())["merge_weights"]
    if weights is None:
        weights = [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        raise SystemExit(f"weights ({len(weights)}) != checkpoints ({len(args.checkpoints)})")

    merged = merge_checkpoints(args.checkpoints, weights, method=args.method,
                               base_path=args.base, density=args.density)
    torch.save(merged, args.out)
    print(f"merged {len(args.checkpoints)} checkpoints ({args.method}, weights={weights}) → {args.out}")


if __name__ == "__main__":
    main()
