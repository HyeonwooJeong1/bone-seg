import pathlib

SCRIPT = "ai_bone/train/docker_train.sh"

def test_docker_train_wraps_nnunet_in_container():
    t = pathlib.Path(SCRIPT).read_text(encoding="utf-8")
    assert "docker run --rm --gpus all" in t
    assert "CUDA_VISIBLE_DEVICES" in t
    assert "bone-nnunet:2.8.1" in t
    assert "nnUNetv2_train" in t
    assert "--c" in t                       # checkpoint resume
    # both custom trainers bind-mounted into the image's nnunetv2 trainer dir
    assert "partial_label_trainer.py:" in t
    assert "merit_finetune_trainer.py:" in t
    assert "nnUNet_compile=f" in t
