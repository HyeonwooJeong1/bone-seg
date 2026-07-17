"""nnU-Net trainer applying the marginal (partial-label) loss. (SERVER)

Design (minimizes nnU-Net-version coupling):
  - `_build_loss` reuses the parent's deep-supervision wrapper but swaps the inner
    base loss for our marginal loss (`make_marginal_ds_loss`), so nnU-Net's DS
    weights and iteration are untouched and the loss keeps the standard
    `loss(output, target)` signature.
  - The marginal loss pulls the CURRENT batch's per-case present masks from a
    getter; `train_step`/`validation_step` only stash those masks then delegate to
    the parent (no copy of the training loop internals).
  - Per-case `present_labels` are read from the MERGED RAW dataset's
    `labelsTr/*.present.json` (preprocessing does not copy the sidecars), keyed by
    the same case identifiers nnU-Net uses in `batch['keys']`.

Two partial-label mechanisms combine: nnU-Net's native ignore label (taxonomy 54,
per-voxel) and the marginal collapse (per-class, this trainer). TotalSeg (all
classes annotated) reduces to the standard loss.

torch / nnunetv2 are imported by the parent; this module is server-only. Finalize
with a 1-epoch GPU smoke test (runbook §7): confirm `batch['keys']` carries the
case ids and DeepSupervisionWrapper exposes `.loss` in the installed nnU-Net.
"""
import glob
import json
import os

import numpy as np

from ai_bone.train.partial_label_trainer import nnUNetTrainerNoMirroring_ES_PL
from ai_bone.train.marginal_loss import make_marginal_ds_loss, present_mask_from_ids


class nnUNetTrainerMarginal(nnUNetTrainerNoMirroring_ES_PL):
    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._present_ids = {}          # {case_id: [present fg class ids]}
        self._current_present = None    # (B, C) bool tensor for the active batch

    # ---- present-label bookkeeping -------------------------------------------
    def _raw_labels_dir(self):
        """MERGED RAW labelsTr, where the *.present.json sidecars live."""
        raw = os.environ.get("nnUNet_raw", "")
        ds = os.path.basename(self.preprocessed_dataset_folder_base)
        return os.path.join(raw, ds, "labelsTr")

    def _load_present_map(self):
        from ai_bone import taxonomy_v1 as tx
        lab_dir = self._raw_labels_dir()
        found = glob.glob(os.path.join(lab_dir, "*.present.json"))
        for p in found:
            case = os.path.basename(p)[: -len(".present.json")]
            names = json.loads(open(p, encoding="utf-8").read())["present_labels"]
            self._present_ids[case] = [tx.NAME_TO_ID[n] for n in names if n in tx.NAME_TO_ID]
        self.print_to_log_file(
            f"[marginal] loaded present-label sidecars for {len(self._present_ids)} "
            f"cases from {lab_dir}")
        if not self._present_ids:
            self.print_to_log_file(
                "[marginal] WARNING: no present.json found — marginal loss will act "
                "as the standard loss (all classes treated as annotated).")

    def _num_seg_heads(self):
        return self.label_manager.num_segmentation_heads

    def _batch_present_masks(self, case_keys):
        """(B,) case ids → (B, C) bool present mask on the trainer device.
        A case with no sidecar → fully annotated (all True) = standard loss."""
        import torch
        C = self._num_seg_heads()
        rows = [present_mask_from_ids(self._present_ids.get(k, []), C)
                for k in case_keys]
        return torch.as_tensor(np.stack(rows), device=self.device)

    # ---- nnU-Net hooks --------------------------------------------------------
    def initialize(self):
        super().initialize()
        self._load_present_map()

    def _build_loss(self):
        """Reuse the parent's (deep-supervision-wrapped) loss but swap the base
        loss for the marginal one, keeping nnU-Net's DS weights."""
        loss = super()._build_loss()
        marg = make_marginal_ds_loss(
            get_present=lambda: self._current_present,
            ignore_label=self.label_manager.ignore_label,
        )
        try:
            from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        except Exception:
            DeepSupervisionWrapper = None
        if DeepSupervisionWrapper is not None and isinstance(loss, DeepSupervisionWrapper):
            loss.loss = marg            # keep DS weight_factors, replace inner loss
            return loss
        return marg

    def train_step(self, batch):
        self._current_present = self._batch_present_masks(batch["keys"])
        return super().train_step(batch)

    def validation_step(self, batch):
        self._current_present = self._batch_present_masks(batch["keys"])
        return super().validation_step(batch)


if __name__ == "__main__":
    raise SystemExit("server trainer; use via nnUNetv2_train -tr nnUNetTrainerMarginal")
