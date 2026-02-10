# utils.py
import os
import json
import time
from typing import Any, Dict, Optional, Tuple

import torch

def save_checkpoint(path: str, model, optimizer=None, scaler=None, step: int = 0, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)

def load_checkpoint(path: str, model, optimizer=None, scaler=None, strict: bool = True) -> Tuple[int, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt.get("step", 0))
    extra = ckpt.get("extra", {})
    return step, extra

def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class SimpleLogger:
    def __init__(self):
        self.t0 = time.time()

    def log(self, msg: str):
        dt = time.time() - self.t0
        print(f"[{dt:8.1f}s] {msg}")

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
