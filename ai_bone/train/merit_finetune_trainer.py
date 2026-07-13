"""MERIT 파티션 fine-tune: shared init에서 짧고 낮은 LR (병합 basin 유지).

주의: nnU-Net이 `-tr`로 찾으려면 이 파일과 partial_label_trainer.py가 모두
`nnunetv2/training/nnUNetTrainer/`에 놓여야 한다(conda=복사, docker=마운트).
따라서 형제 모듈 import도 그 경로를 기준으로 한다(ai_bone.train 아님)."""
from nnunetv2.training.nnUNetTrainer.partial_label_trainer import (
    nnUNetTrainerNoMirroring_ES_PL,
)

class nnUNetTrainerMERITFinetune(nnUNetTrainerNoMirroring_ES_PL):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300
