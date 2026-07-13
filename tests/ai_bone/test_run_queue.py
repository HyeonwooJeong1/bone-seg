import pathlib

SCRIPT = "ai_bone/train/run_queue.sh"

def test_run_queue_has_required_env_and_logic():
    t = pathlib.Path(SCRIPT).read_text()
    # server env / robustness knobs
    assert "LD_LIBRARY_PATH" in t
    assert "nnUNet_compile=f" in t
    assert "nnUNet_n_proc_DA" in t          # feed the GPU
    assert "CUDA_VISIBLE_DEVICES" in t
    # per-GPU round-robin queue + resume
    assert "idx % NG" in t
    assert "--c" in t
    assert "nnUNetv2_train" in t
