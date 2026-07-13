import pathlib
SCRIPTS = ["ai_bone/train/stage1_pretrain.sh","ai_bone/train/stage2_baseline.sh",
           "ai_bone/merit/merit_train_partition.sh"]
def test_scripts_have_required_env():
    for s in SCRIPTS:
        t = pathlib.Path(s).read_text(encoding='utf-8')
        assert "LD_LIBRARY_PATH" in t
        assert "nnUNet_compile=f" in t
        assert "CUDA_VISIBLE_DEVICES" in t
