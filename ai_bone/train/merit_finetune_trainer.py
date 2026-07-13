"""MERIT 파티션 fine-tune: shared init에서 짧고 낮은 LR (병합 basin 유지)."""
from ai_bone.train.partial_label_trainer import nnUNetTrainerNoMirroring_ES_PL

class nnUNetTrainerMERITFinetune(nnUNetTrainerNoMirroring_ES_PL):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.num_epochs = 300
