"""
nnUNetTrainerNoMirroring_ES — NoMirroring + validation dice 기반 early stopping.

좌우 라벨 때문에 mirroring 끄고(NoMirroring 상속), validation EMA dice가
es_patience epoch 동안 개선되지 않으면 학습을 조기 종료한다.
초기 dice=0 구간을 통과하도록 최소 es_min_epochs는 보장.

서버 배치 위치:
  .../site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerNoMirroring_ES.py
사용: nnUNetv2_train <d> 3d_fullres <fold> -p nnUNetPlans_iso06 -tr nnUNetTrainerNoMirroring_ES
"""
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import (
    nnUNetTrainerNoMirroring,
)


class nnUNetTrainerNoMirroring_ES(nnUNetTrainerNoMirroring):
    # 부모와 동일한 명시적 시그니처 필수 (nnUNetTrainer가 my_init_kwargs를 locals()로 캡처)
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000          # 상한 (수렴하면 그 전에 멈춤)
        self.es_patience = 75           # 개선 없이 이만큼 지나면 종료
        self.es_min_epochs = 200        # 최소 보장 (초기 dice=0 구간 통과)
        self._best_ema = -1.0
        self._stale = 0

    def on_epoch_end(self):
        super().on_epoch_end()
        try:
            ema = float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
        except Exception:
            return
        if np.isnan(ema):
            return
        if ema > self._best_ema + 1e-4:
            self._best_ema = ema
            self._stale = 0
        else:
            self._stale += 1
        if (self.current_epoch + 1) >= self.es_min_epochs and self._stale >= self.es_patience:
            self.print_to_log_file(
                f"[EarlyStop] EMA dice가 {self.es_patience} epoch 동안 개선 없음 "
                f"(best={self._best_ema:.4f}) → epoch {self.current_epoch}에서 조기 종료"
            )
            self.num_epochs = self.current_epoch + 1
