import numpy as np
import pytest
from ai_bone.merit import conflict_analysis as ca


def test_cosine_matrix_values():
    grads = {"a": [1, 0], "b": [1, 0], "c": [-1, 0], "d": [0, 1]}
    names, C = ca.cosine_matrix(grads)
    i = {n: k for k, n in enumerate(names)}
    assert abs(C[i["a"], i["b"]] - 1.0) < 1e-9      # aligned
    assert abs(C[i["a"], i["c"]] + 1.0) < 1e-9      # opposed
    assert abs(C[i["a"], i["d"]]) < 1e-9            # orthogonal


def test_pca_embed_shape():
    grads = {f"d{i}": np.random.default_rng(i).normal(size=6) for i in range(5)}
    _, C = ca.cosine_matrix(grads)
    assert ca.pca_embed(C, r=2).shape == (5, 2)


def test_ari_relabel_invariant_and_independent():
    assert abs(ca.adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert ca.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.5   # independent


def test_nmi_identical_is_one():
    assert abs(ca.normalized_mutual_info([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-9


def test_agreement_perfect():
    part = {0: ["a", "b"], 1: ["c", "d"]}
    region = {"a": "x", "b": "x", "c": "y", "d": "y"}
    agr = ca.agreement(part, region)
    assert abs(agr["ari"] - 1.0) < 1e-9 and abs(agr["nmi"] - 1.0) < 1e-9


def test_infer_regions_from_labelmaps():
    reg = ca.infer_regions(["ribseg", "mug500", "ctpelvic1k"])
    assert reg["ribseg"] == "ribs"
    assert reg["mug500"] == "skull"
    assert reg["ctpelvic1k"] == "pelvis"


def test_plots_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    grads = {f"d{i}": np.random.default_rng(i).normal(size=8) for i in range(4)}
    names, C = ca.cosine_matrix(grads)
    embed = ca.pca_embed(C, r=2)
    part = ca.pca_conflict_split(grads, k=2)
    p1 = ca.plot_conflict_matrix(names, C, str(tmp_path / "C.png"))
    p2 = ca.plot_pca_embedding(embed, names, part, str(tmp_path / "pca.png"))
    assert (tmp_path / "C.png").exists() and (tmp_path / "pca.png").exists()
