import pathlib


def test_docker_train_wraps_nnunet_in_container():
    t = pathlib.Path("ai_bone/train/docker_train.sh").read_text(encoding="utf-8")
    assert "docker run --rm --gpus all" in t
    assert "CUDA_VISIBLE_DEVICES" in t
    assert "bone-pipeline:latest" in t          # trainers + code baked in
    assert "nnUNetv2_train" in t
    assert "--c" in t                            # checkpoint resume
    assert "nnUNet_compile=f" in t


def test_run_in_docker_mounts_data1():
    t = pathlib.Path("ai_bone/run_in_docker.sh").read_text(encoding="utf-8")
    assert "docker run --rm" in t
    assert "-v /data1:/data1" in t
    assert "bone-pipeline" in t
