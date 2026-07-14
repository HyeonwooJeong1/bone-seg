"""Merge-readiness diagnostics (MERIT §3/§5; RQ3): is the CADS-pretrained init a
flat basin where independently fine-tuned branches merge without collapsing?

Three diagnostics (docs/experiment_design.md §5.3):
  1. Weight displacement  — ‖θ_k − θ0‖ and merged-vs-branch ratio (MERIT Table 1).
  2. Linear mode connectivity — loss barrier along θ_a↔θ_b interpolation (should be ~0).
  3. Perturbation robustness — loss rise under Gaussian weight noise (flatter = better).

The numeric orchestration is fully implemented and testable via an injected
`eval_fn(state_dict) -> loss`. On the server, `eval_fn` is an nnU-Net forward pass
(see `make_nnunet_eval_fn`); locally it is a lightweight stub in tests. Displacement
needs no model and runs offline on saved checkpoints.
"""
import numpy as np


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def flatten_state(sd, keys=None):
    keys = keys or list(sd)
    return np.concatenate([_np(sd[k]).ravel() for k in keys])


# ----- 1. weight displacement (offline) --------------------------------------
def displacement(theta0, theta, keys=None):
    keys = keys or list(theta0)
    return float(np.linalg.norm(flatten_state(theta, keys) - flatten_state(theta0, keys)))


def displacement_report(theta0, branch_states, merged, keys=None):
    keys = keys or list(theta0)
    ref = flatten_state(theta0, keys)
    branch = [float(np.linalg.norm(flatten_state(s, keys) - ref)) for s in branch_states]
    merged_d = float(np.linalg.norm(flatten_state(merged, keys) - ref))
    mean_branch = float(np.mean(branch)) if branch else float("nan")
    # MERIT Table 1: merged stays CLOSER to θ0 than branches → ratio > 1.
    ratio = (mean_branch / merged_d) if merged_d else float("nan")
    return {"branch_disp": branch, "merged_disp": merged_d,
            "mean_branch_disp": mean_branch, "mean_branch_over_merged": ratio}


# ----- interpolation + barrier -----------------------------------------------
def interpolate_state(a, b, alpha, keys=None):
    keys = keys or list(a)
    return {k: (1.0 - alpha) * _np(a[k]) + alpha * _np(b[k]) for k in keys}


def loss_barrier(alphas, losses):
    """Max rise above the endpoint linear interpolation along a path. >0 = barrier
    (branches NOT linearly mode-connected); ≤0 = at/below linear (ideal)."""
    a = np.asarray(alphas, float)
    L = np.asarray(losses, float)
    lin = (1.0 - a) * L[0] + a * L[-1]
    return float(np.max(L - lin))


# ----- 2. linear mode connectivity (needs eval_fn) ---------------------------
def lmc_loss_barrier(state_a, state_b, eval_fn, n=11, keys=None):
    """Sample loss along θ_a→θ_b and return the barrier. eval_fn(state)->loss."""
    alphas = np.linspace(0.0, 1.0, n)
    losses = [float(eval_fn(interpolate_state(state_a, state_b, a, keys))) for a in alphas]
    return {"alphas": alphas.tolist(), "losses": losses, "barrier": loss_barrier(alphas, losses)}


# ----- 3. perturbation robustness (needs eval_fn) ----------------------------
def gaussian_perturb_state(sd, sigma, seed=0, keys=None):
    keys = keys or list(sd)
    rng = np.random.default_rng(seed)
    out = {}
    for k in keys:
        v = _np(sd[k]).astype("float32")
        out[k] = v + rng.standard_normal(v.shape).astype("float32") * sigma
    return out


def perturbation_robustness(state, eval_fn, sigmas=(0.01, 0.05, 0.1), reps=3, seed=0, keys=None):
    base = float(eval_fn(state))
    by_sigma = {}
    for s in sigmas:
        inc = [float(eval_fn(gaussian_perturb_state(state, s, seed=seed + r, keys=keys))) - base
               for r in range(reps)]
        by_sigma[str(s)] = {"mean_increase": float(np.mean(inc)), "increases": inc}
    return {"base_loss": base, "by_sigma": by_sigma}


# ----- server: nnU-Net loss eval_fn (lazy torch/nnunetv2) ---------------------
def make_nnunet_eval_fn(trainer, n_batches=2):
    """Return eval_fn(state_dict)->mean loss over n_batches of the val loader.
    trainer = an initialized nnU-Net trainer (network, loss, dataloaders). SERVER."""
    import torch

    def eval_fn(state_dict):
        trainer.network.load_state_dict(state_dict, strict=False)
        trainer.network.eval()
        losses = []
        with torch.no_grad():
            it = iter(trainer.dataloader_val)
            for _ in range(n_batches):
                batch = next(it)
                data = batch["data"].to(trainer.device, non_blocking=True)
                target = batch["target"]
                target = [t.to(trainer.device) for t in target] if isinstance(target, list) \
                    else target.to(trainer.device)
                out = trainer.network(data)
                losses.append(float(trainer.loss(out, target).item()))
        return float(np.mean(losses))

    return eval_fn


def main():
    """Offline displacement report from saved checkpoints (no GPU). LMC/perturbation
    need a model+data eval_fn → run on the server via make_nnunet_eval_fn."""
    import argparse, json
    ap = argparse.ArgumentParser(description="Merge-readiness: weight displacement (offline).")
    ap.add_argument("--init", required=True, help="θ0 checkpoint (.pth/.npz)")
    ap.add_argument("--branches", nargs="+", required=True, help="branch checkpoints")
    ap.add_argument("--merged", required=True, help="merged checkpoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def _load(p):
        if p.endswith(".npz"):
            d = np.load(p); return {k: d[k] for k in d.files}
        import torch
        c = torch.load(p, map_location="cpu")
        return c.get("network_weights", c)

    theta0 = _load(args.init)
    branches = [_load(p) for p in args.branches]
    merged = _load(args.merged)
    rep = displacement_report(theta0, branches, merged)
    if args.out:
        json.dump(rep, open(args.out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
