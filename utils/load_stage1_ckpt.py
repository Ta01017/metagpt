# utils/load_stage1_ckpt.py
import torch

def load_stage1_into_stage2(model, ckpt_path: str):
    print(f"[Stage2] Loading Stage1 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    # 过滤掉 prefix encoder + 与 Stage1 不匹配的 key
    filtered = {}
    for k, v in state.items():
        if k.startswith("prefix."):
            continue  # Stage1 没有 prefix，跳过
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print(f"[Stage2] Loaded params: {len(filtered)}")
    print(f"[Stage2] Missing keys: {len(missing)} (OK)")
    print(f"[Stage2] Unexpected keys: {len(unexpected)} (OK)")
    print("[Stage2] Load done.\n")
