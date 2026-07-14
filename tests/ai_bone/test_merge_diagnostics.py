import numpy as np
from ai_bone.merit import merge_diagnostics as md

def test_displacement_report_merged_closer():
    t0 = {"w": np.array([0.0, 0.0])}
    b1 = {"w": np.array([2.0, 0.0])}
    b2 = {"w": np.array([0.0, 2.0])}
    merged = {"w": np.array([1.0, 1.0])}          # weight-average of b1,b2
    rep = md.displacement_report(t0, [b1, b2], merged)
    assert rep["branch_disp"] == [2.0, 2.0]
    assert abs(rep["merged_disp"] - np.sqrt(2)) < 1e-9
    assert rep["mean_branch_over_merged"] > 1.0    # merged nearer θ0 (MERIT signature)

def test_interpolate_and_barrier():
    a = {"w": np.array([0.0, 0.0])}; b = {"w": np.array([2.0, 2.0])}
    assert np.allclose(md.interpolate_state(a, b, 0.5)["w"], [1.0, 1.0])
    assert md.loss_barrier([0, .5, 1], [1, 1, 1]) == 0.0
    assert md.loss_barrier([0, .5, 1], [1, 2, 1]) == 1.0     # a bump = positive barrier
    assert md.loss_barrier([0, .5, 1], [1, 0.5, 1]) == 0.0   # dip below linear = no barrier

def test_lmc_barrier_convex_eval_is_connected():
    a = {"w": np.array([-1.0])}; b = {"w": np.array([1.0])}
    eval_fn = lambda s: float(np.linalg.norm(s["w"]))       # convex → below-linear path
    res = md.lmc_loss_barrier(a, b, eval_fn, n=11)
    assert res["barrier"] <= 1e-9

def test_perturbation_reproducible_and_summarized():
    s = {"w": np.zeros(50, dtype="float32")}
    p1 = md.gaussian_perturb_state(s, 0.1, seed=7)
    p2 = md.gaussian_perturb_state(s, 0.1, seed=7)
    assert np.array_equal(p1["w"], p2["w"])                 # reproducible
    eval_fn = lambda st: float(np.linalg.norm(st["w"]))
    rob = md.perturbation_robustness(s, eval_fn, sigmas=(0.05, 0.1), reps=2, seed=0)
    assert rob["base_loss"] == 0.0
    assert set(rob["by_sigma"]) == {"0.05", "0.1"}
    assert all(np.isfinite(rob["by_sigma"][k]["mean_increase"]) for k in rob["by_sigma"])
