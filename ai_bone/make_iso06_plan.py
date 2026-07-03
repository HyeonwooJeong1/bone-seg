"""
make_iso06_plan.py — Dataset490의 등방 0.6mm 플랜(nnUNetPlans_iso06) 생성.

방식: 490의 기본 nnUNetPlans.json(490 자체의 CT 강도/정규화 통계 보존)을 베이스로,
3d_fullres 설정만 기존 476의 검증된 iso06 설정(spacing 0.6등방, patch [224,80,128],
batch 2, 그에 맞는 architecture)으로 교체한다. → 476/481과 동일한 학습 조건 보장.
(476 iso06가 없으면 fallback: 490 기본 3d_fullres의 spacing만 0.6등방으로 override.)

실행(서버):
  nnUNet_preprocessed=/home/ubuntu/nnunet_pre python ai_bone/make_iso06_plan.py 490
"""
import sys, os, json, copy, glob

did = int(sys.argv[1]) if len(sys.argv) > 1 else 490
pre = os.environ["nnUNet_preprocessed"]

ds = next(d for d in os.listdir(pre) if d.startswith(f"Dataset{did:03d}_"))
base_path = os.path.join(pre, ds, "nnUNetPlans.json")
plans = json.load(open(base_path))

# 476의 검증된 iso06 3d_fullres 설정을 템플릿으로 사용
ref = glob.glob(os.path.join(pre, "Dataset476*", "nnUNetPlans_iso06.json"))
if ref:
    ref_cfg = json.load(open(ref[0]))["configurations"]["3d_fullres"]
    cfg = copy.deepcopy(ref_cfg)          # spacing/patch/architecture 통째로 계승
    src = f"476 iso06 템플릿 계승 (patch={cfg.get('patch_size')})"
else:
    cfg = copy.deepcopy(plans["configurations"]["3d_fullres"])
    cfg["spacing"] = [0.6, 0.6, 0.6]      # fallback: spacing만 등방 0.6
    src = f"fallback spacing override (patch={cfg.get('patch_size')})"

plans["plans_name"] = "nnUNetPlans_iso06"
plans["configurations"] = {"3d_fullres": cfg}   # 3d_fullres만 학습

out = os.path.join(pre, ds, "nnUNetPlans_iso06.json")
json.dump(plans, open(out, "w"), indent=2)
print(f"저장: {out}")
print(f"  {src}, spacing={cfg['spacing']}, batch={cfg.get('batch_size')}")
