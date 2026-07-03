# nnU-Net v2 (2.8.1) 학습/추론용 이미지
# 호스트에서 만든 전처리 데이터·플랜(nnunetv2 2.8.1)과 100% 호환되도록 버전 고정.
# torch 2.5.1+cu124 → H100(sm_90)/드라이버 580에서 동작. 데이터 포맷은 nnunetv2 버전에 의존.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir nnunetv2==2.8.1 pydicom nibabel

# 좌우 라벨용 커스텀 트레이너(NoMirroring + early stopping)를 설치 패키지에 삽입
# (build context = ai_bone/ 이므로 파일이 최상위에 있음)
COPY nnUNetTrainerNoMirroring_ES.py /tmp/nnUNetTrainerNoMirroring_ES.py
RUN cp /tmp/nnUNetTrainerNoMirroring_ES.py \
    "$(python -c 'import nnunetv2,os;print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/"

# nnU-Net 경로 (docker run -v 마운트와 일치) + 자원 최대 활용 기본값
ENV nnUNet_raw=/data1/bone/ai_bone/nnunet/raw \
    nnUNet_preprocessed=/nnunet_pre \
    nnUNet_results=/data1/bone/ai_bone/nnunet/results \
    nnUNet_compile=f \
    nnUNet_n_proc_DA=24

WORKDIR /workspace
