"""데이터셋별 gradient 방향 추정 → PCA split 입력용 g.npz. (서버 GPU 실행)"""
import argparse, numpy as np

def reduce_grad(flat_grad: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """고정 랜덤투영으로 저차원화. proj: (D_low, D_full)."""
    return proj @ flat_grad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--datasets", required=True)   # comma-sep
    ap.add_argument("--out", default="g.npz")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()

    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa
    # 아래는 서버에서 nnU-Net predictor/trainer 내부를 재사용해 grad를 뽑는다.
    # 핵심: 각 dataset 배치로 forward+backward → seg head+디코더말단 grad flatten
    #       → reduce_grad(proj) 저차원 → dataset별 평균 벡터.
    # (실제 nnU-Net 내부 연결은 서버 환경에서 확정: Task16 runbook에 실행법 명시)
    raise SystemExit("run on server; see ai_bone/runbook.md §MERIT")

if __name__ == "__main__":
    main()
