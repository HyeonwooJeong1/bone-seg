# Whole-pipeline image for the unified-bone project: nnU-Net base + our data-prep /
# download / eval dependencies + the ai_bone code + custom trainers baked in.
# Everything (download, combine, build_raw, preprocess, train, eval) runs inside
# this one container. GPU is exposed only for training/inference (docker --gpus).
FROM bone-nnunet:2.8.1

# Extra deps not in the nnU-Net base (SimpleITK/scipy/numpy/nibabel already present).
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Project code on the path.
COPY ai_bone /opt/ai_bone/ai_bone
ENV PYTHONPATH=/opt/ai_bone

# Make the custom nnU-Net trainers discoverable by `-tr` (copy into nnunetv2).
RUN NND=$(python -c "import nnunetv2,os;print(os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer'))") && \
    cp /opt/ai_bone/ai_bone/train/partial_label_trainer.py  "$NND"/ && \
    cp /opt/ai_bone/ai_bone/train/marginal_trainer.py       "$NND"/ && \
    cp /opt/ai_bone/ai_bone/train/merit_finetune_trainer.py "$NND"/ && \
    (cp /opt/ai_bone/ai_bone/nnUNetTrainerNoMirroring_ES.py "$NND"/ || true)

# nnU-Net data roots default to the mounted /data1 (override with -e as needed).
ENV nnUNet_raw=/data1/hyeonwoo/bone/nnunet/raw \
    nnUNet_preprocessed=/data1/hyeonwoo/bone/nnunet/preprocessed \
    nnUNet_results=/data1/hyeonwoo/bone/nnunet/results \
    nnUNet_compile=f

WORKDIR /workspace
