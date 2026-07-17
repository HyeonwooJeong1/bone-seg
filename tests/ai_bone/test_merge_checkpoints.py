import torch
from ai_bone.merit.merge_checkpoints import (
    merge_state_dicts_weighted, merge_state_dicts_ties, merge_checkpoints,
)


def _sd(w_val, buf_val):
    return {"conv.weight": torch.full((2, 2), float(w_val)),
            "norm.count": torch.tensor([int(buf_val)], dtype=torch.long)}


def test_weighted_average_matches_weights_and_keeps_int_buffers():
    sds = [_sd(2.0, 5), _sd(4.0, 9)]
    out = merge_state_dicts_weighted(sds, weights=[3, 1])   # 0.75*2 + 0.25*4 = 2.5
    assert torch.allclose(out["conv.weight"], torch.full((2, 2), 2.5))
    assert out["norm.count"].item() == 5                    # int buffer → from first
    assert out["conv.weight"].dtype == torch.float32        # dtype preserved


def test_weighted_rejects_mismatched_keys():
    a = _sd(1.0, 1); b = {"other": torch.zeros(2)}
    try:
        merge_state_dicts_weighted([a, b], [1, 1]); assert False
    except ValueError:
        pass


def test_ties_merge_returns_torch_tensors():
    base = {"w": torch.zeros(4)}
    sds = [{"w": torch.tensor([1.0, 0.0, 2.0, 0.0])},
           {"w": torch.tensor([1.0, 0.0, 0.0, 3.0])}]
    out = merge_state_dicts_ties(base, sds, [1, 1], density=1.0)
    assert isinstance(out["w"], torch.Tensor) and out["w"].shape == (4,)


def test_merge_checkpoints_preserves_structure(tmp_path):
    c0 = {"network_weights": _sd(2.0, 5), "current_epoch": 100, "trainer_name": "X"}
    c1 = {"network_weights": _sd(6.0, 9), "current_epoch": 100, "trainer_name": "X"}
    p0 = tmp_path / "b0.pth"; p1 = tmp_path / "b1.pth"
    torch.save(c0, p0); torch.save(c1, p1)
    merged = merge_checkpoints([str(p0), str(p1)], weights=[1, 1])   # mean → 4.0
    assert torch.allclose(merged["network_weights"]["conv.weight"], torch.full((2, 2), 4.0))
    assert merged["current_epoch"] == 100 and merged["trainer_name"] == "X"  # structure kept
