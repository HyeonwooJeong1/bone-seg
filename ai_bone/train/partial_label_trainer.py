"""Partial-label trainer. nnU-Net v2는 dataset.json의 ignore label(255)을 네이티브
지원하므로 loss 마스킹은 프레임워크가 처리한다. 여기선 좌우 NoMirroring + ES를
상속하고, 통합셋의 데이터셋 불균형 완화를 위해 foreground oversample을 올린다."""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerNoMirroring_ES import (
    nnUNetTrainerNoMirroring_ES,
)

class nnUNetTrainerNoMirroring_ES_PL(nnUNetTrainerNoMirroring_ES):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 0.5   # 희소 뼈 대비 상향
