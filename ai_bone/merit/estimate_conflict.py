"""Per-dataset gradient conflict estimation at the merge-ready init θ0. (SERVER, GPU)

For each source dataset, average the gradient over a small calibration set of
preprocessed cases at θ0, reduce it to a low-dim vector via a fixed random
projection (Johnson–Lindenstrauss), normalize, and save an .npz of
{dataset: vector} — consumed by `conflict_analysis.py` (cosine C, PCA, split).

Design (MERIT §4.1): gradients are taken from a curvature-bearing subset (decoder
tail + seg heads) to keep it cheap; a fixed random projection gives a stable
low-dim signature comparable across datasets. Run AFTER Stage-1 pretraining, with a
manifest mapping each preprocessed case id → its source dataset.

torch / nnU-Net are imported lazily inside main() so this file parses without them.
NOTE: nnU-Net's preprocessed-dataset loader class name can differ by version
(2.8.1 = nnUNetDataset.load_case); adjust that one import if your install differs.
"""
import argparse
import json

import numpy as np


def make_projection(dim_full, dim_low, seed=0):
    """Fixed Gaussian random projection (dim_low x dim_full), JL-scaled."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((dim_low, dim_full)).astype(np.float32)
            / np.sqrt(dim_low))


def reduce_grad(flat_grad, proj):
    """Low-dim signature of a full gradient via the fixed projection."""
    return proj @ flat_grad


def select_params(network):
    """Curvature-bearing, cheap subset: decoder + seg-head params (fallback: all)."""
    named = list(network.named_parameters())
    picked = [p for n, p in named if any(k in n.lower() for k in ("seg", "decoder"))]
    return picked or [p for _, p in named]


def _flat_grad(params):
    import torch
    gs = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return torch.cat(gs).float().cpu().numpy() if gs else None


def _random_patch(data, seg, patch, rng):
    """Random spatial patch of `patch` from (C, *spatial) arrays; pad if smaller."""
    import numpy as _np
    spatial = data.shape[1:]
    starts, slabs = [], []
    for sz, ps in zip(spatial, patch):
        if sz >= ps:
            s = int(rng.integers(0, sz - ps + 1)); slabs.append(slice(s, s + ps))
        else:
            slabs.append(slice(0, sz))
        starts.append(0)
    dsl = (slice(None),) + tuple(slabs)
    d = data[dsl]; s = seg[dsl]
    pad_d = [(0, 0)] + [(0, max(0, ps - c)) for ps, c in zip(patch, d.shape[1:])]
    if any(p[1] for p in pad_d):
        d = _np.pad(d, pad_d); s = _np.pad(s, pad_d)
    return d, s


def main():
    ap = argparse.ArgumentParser(description="Per-dataset gradient conflict at θ0 (server).")
    ap.add_argument("--dataset-id", type=int, required=True, help="merged FT nnUNet dataset id")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans", default="nnUNetPlans_iso06")
    ap.add_argument("--trainer", default="nnUNetTrainerNoMirroring_ES_PL")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--init", required=True, help="θ0 checkpoint (Stage-1 pretrain)")
    ap.add_argument("--case-datasets", required=True, help="JSON {case_id: dataset}")
    ap.add_argument("--out", default="grads.npz")
    ap.add_argument("--per-dataset", type=int, default=40, help="calibration cases/dataset")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from nnunetv2.run.run_training import get_trainer_from_args
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

    # Build network + loss exactly like training, then load the merge-ready init.
    trainer = get_trainer_from_args(str(args.dataset_id), args.config, args.fold,
                                    args.trainer, args.plans)
    trainer.initialize()
    net = trainer.network
    ckpt = torch.load(args.init, map_location=trainer.device)
    net.load_state_dict(ckpt.get("network_weights", ckpt))
    net.eval()                                   # instance-norm has no running stats

    patch = trainer.configuration_manager.patch_size
    params = select_params(net)
    dim_full = int(sum(p.numel() for p in params))
    proj = make_projection(dim_full, args.dim, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    case2ds = json.loads(open(args.case_datasets, encoding="utf-8").read())
    by_ds = {}
    for cid, ds in case2ds.items():
        by_ds.setdefault(ds, []).append(cid)

    ds_obj = nnUNetDataset(trainer.preprocessed_dataset_folder, None)
    reduced = {}
    for ds, cids in by_ds.items():
        picks = list(rng.permutation(cids))[: args.per_dataset]
        accum = np.zeros(args.dim, dtype=np.float64)
        used = 0
        for cid in picks:
            try:
                data, seg, _ = ds_obj.load_case(cid)          # (C,*sp), (1,*sp)
            except Exception:
                continue
            d, s = _random_patch(np.asarray(data), np.asarray(seg), patch, rng)
            x = torch.as_tensor(d[None]).float().to(trainer.device)      # (1,C,*sp)
            y = torch.as_tensor(s[None]).to(trainer.device)             # (1,1,*sp)
            net.zero_grad(set_to_none=True)
            out = net(x)
            loss = trainer.loss(out, y if not isinstance(out, (list, tuple))
                                else [y] * len(out))
            loss.backward()
            fg = _flat_grad(params)
            if fg is not None:
                accum += reduce_grad(fg.astype(np.float32), proj)
                used += 1
        if used:
            v = accum / used
            reduced[ds] = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
            print(f"[conflict] {ds}: {used} cases -> vector({args.dim})")

    np.savez(args.out, **reduced)
    print(f"saved {len(reduced)} dataset gradient vectors -> {args.out}")


if __name__ == "__main__":
    main()
