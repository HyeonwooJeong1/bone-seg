"""nnU-Net trainer applying the marginal (partial-label) loss. (SERVER)

Loads per-case `present_labels` from the merged dataset's labelsTr/*.present.json,
maps case id → present class-id mask, and computes the marginal Dice+CE so that
partial datasets (e.g. CTPelvic1K) do not supervise their unannotated classes.

torch / nnunetv2 are imported lazily so this parses without them. The exact
train_step override depends on the installed nnU-Net version (batch key access,
deep supervision, AMP); the structure below is the reference to finalize on the
server (runbook §7)."""
import glob
import json
import os

import numpy as np

from ai_bone.train.partial_label_trainer import nnUNetTrainerNoMirroring_ES_PL
from ai_bone.train.marginal_loss import MarginalDiceCELoss, present_mask_from_ids


class nnUNetTrainerMarginal(nnUNetTrainerNoMirroring_ES_PL):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._present_ids = {}          # {case_id: [present fg class ids]}
        self._num_classes = len([k for k in dataset_json["labels"] if k != "ignore"])

    def _load_present_map(self):
        from ai_bone import taxonomy_v1 as tx
        lab_dir = os.path.join(self.preprocessed_dataset_folder_base
                               if hasattr(self, "preprocessed_dataset_folder_base") else "",
                               "labelsTr")
        # fall back to the raw labelsTr (present sidecars live there)
        candidates = glob.glob(os.path.join(lab_dir, "*.present.json"))
        for p in candidates:
            case = os.path.basename(p)[: -len(".present.json")]
            names = json.loads(open(p, encoding="utf-8").read())["present_labels"]
            self._present_ids[case] = [tx.NAME_TO_ID[n] for n in names if n in tx.NAME_TO_ID]

    def _batch_present_masks(self, case_keys):
        """(B,) case ids → (B, num_classes) bool present mask (torch tensor)."""
        import torch
        rows = [present_mask_from_ids(self._present_ids.get(k, []), self._num_classes)
                for k in case_keys]
        return torch.as_tensor(np.stack(rows), device=self.device)

    def initialize(self):
        super().initialize()
        self._load_present_map()
        # Replace the loss with the marginal loss. The trainer's train_step must
        # call: self.loss(logits, target, self._batch_present_masks(batch['keys'])).
        # For deep supervision, apply per-resolution and combine with the usual
        # deep-supervision weights (see nnUNetTrainer._build_loss).
        self.loss = MarginalDiceCELoss()


if __name__ == "__main__":
    raise SystemExit("server trainer; use via nnUNetv2_train -tr nnUNetTrainerMarginal")
