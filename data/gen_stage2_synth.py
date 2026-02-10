# data/gen_stage2_synth.py
import random
import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure project root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator


# -----------------------------------------------------------
# 辅助函数：从 tokenizer 离散值集合里，选取指定范围的值
# -----------------------------------------------------------
def _choose_from_vals(vals: List[int], lo: int, hi: int) -> int:
    cand = [v for v in vals if lo <= v <= hi]
    if len(cand) == 0:
        cand = sorted(vals, key=lambda v: abs(v - (lo + hi) / 2))
        return cand[0]
    return random.choice(cand)


# ===========================================================
#                     Stage2 结构生成器
# ===========================================================
class Stage2StructureGenerator:
    """
    Stage2 的结构合成器（与 Stage1 完全对齐写法）
    - 使用 tokenizer 的离散值集合
    - 使用 validator 做合法性过滤
    - 比 Stage1 可生成更多 token（比如两层结构/更多形状...可扩展）
    """

    def __init__(
        self,
        tokenizer: StructureTokenizer,
        parser: StructureParser,
        validator: StructureValidator,

        P_range_nm: Tuple[int, int] = (400, 900),
        H_range_nm: Tuple[int, int] = (80, 1000),
        R_range_nm: Tuple[int, int] = (40, 300),
        W_range_nm: Tuple[int, int] = (60, 600),

        use_materials: List[str] | None = None,
        p_cyl: float = 0.5,    # 50% 生成 CYL，其余 RECT
        seed: int = 42,
    ):
        self.tk = tokenizer
        self.parser = parser
        self.val = validator

        self.P_range = P_range_nm
        self.H_range = H_range_nm
        self.R_range = R_range_nm
        self.W_range = W_range_nm
        self.p_cyl = float(p_cyl)

        self.use_materials = use_materials if use_materials else tokenizer.materials
        random.seed(seed)

    def sample_one_tokens(self) -> List[str]:
        # Pixel pitch
        P = _choose_from_vals(self.tk.P_vals, *self.P_range)
        H = _choose_from_vals(self.tk.H_vals, *self.H_range)
        mat = random.choice(self.use_materials)

        if random.random() < self.p_cyl:
            # CYL shape
            R = _choose_from_vals(self.tk.R_vals, *self.R_range)
            toks = [
                f"PX_{P}", f"PY_{P}",
                "SUB_Glass_Substrate",
                f"L1_MAT_{mat}",
                "L1_SHAPE_CYL",
                f"L1_H_{H}",
                f"L1_R_{R}",
            ]
        else:
            # RECT shape
            W = _choose_from_vals(self.tk.W_vals, *self.W_range)
            L = _choose_from_vals(self.tk.W_vals, *self.W_range)
            toks = [
                f"PX_{P}", f"PY_{P}",
                "SUB_Glass_Substrate",
                f"L1_MAT_{mat}",
                "L1_SHAPE_RECT",
                f"L1_H_{H}",
                f"L1_W_{W}",
                f"L1_L_{L}",
            ]

        return toks

    def sample_one_ids(self, max_tries=200) -> List[int]:
        """
        返回无 BOS/EOS 的结构 token ID 列表
        """
        for _ in range(max_tries):
            toks = self.sample_one_tokens()
            struct = self.parser.parse(["[BOS]"] + toks + ["[EOS]"])

            ok, reason = self.val.validate(struct)
            if not ok:
                continue

            try:
                return self.tk.encode(toks)
            except KeyError:
                continue

        raise RuntimeError("Stage2StructureGenerator: failed to sample valid structure.")

    def make_dataset_ids(self, N: int) -> List[List[int]]:
        return [self.sample_one_ids() for _ in range(N)]


# ===========================================================
#                     Stage2 光谱生成器
# ===========================================================
class Stage2SpectrumGenerator:
    """
    无需 Meep/TMM 的模拟光谱生成器
    - Smooth + 多峰
    - 和真实光谱行为类似
    - 可直接作为 prefix 训练用数据
    """

    def __init__(self, spec_dim=300, seed=0):
        self.dim = spec_dim
        np.random.seed(seed)
        random.seed(seed)

    def _multi_peak(self):
        x = np.linspace(0, 1, self.dim)
        y = np.zeros(self.dim)

        num_peaks = random.choice([1, 2, 3])
        for _ in range(num_peaks):
            c = random.uniform(0.1, 0.9)      # center
            w = random.uniform(0.02, 0.15)    # width
            h = random.uniform(0.2, 1.0)      # height
            y += h * np.exp(-((x - c)**2) / (2 * w**2))
        return y

    def _smooth_noise(self):
        n = np.random.randn(self.dim)
        kernel = np.array([0.25, 0.5, 0.25])
        return np.convolve(n, kernel, mode="same")

    def sample_one(self):
        y = self._multi_peak() + 0.07 * self._smooth_noise()
        y = np.clip(y, 0, None)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        return y.astype(np.float32)

    def make_dataset(self, N: int) -> List[np.ndarray]:
        return [self.sample_one() for _ in range(N)]


# ===========================================================
#                   对外统一接口（用于 Stage2）
# ===========================================================
def make_stage2_synth_dataset(
    N: int,
    tokenizer: StructureTokenizer,
    parser: StructureParser,
    validator: StructureValidator,
    spec_dim: int = 300,
    seed: int = 0,
) -> Tuple[List[List[int]], List[np.ndarray]]:
    """
    返回：
        structures_ids: List[List[int]]   # 结构 token id 序列
        spectra:        List[np.ndarray]  # 对应的光谱向量
    """

    struct_gen = Stage2StructureGenerator(
        tokenizer=tokenizer,
        parser=parser,
        validator=validator,
        seed=seed,
    )

    spec_gen = Stage2SpectrumGenerator(spec_dim=spec_dim, seed=seed)

    structures = struct_gen.make_dataset_ids(N)
    spectra = spec_gen.make_dataset(N)

    return structures, spectra


# -----------------------------------------------------------
# 简单测试
# -----------------------------------------------------------
if __name__ == "__main__":
    tk = StructureTokenizer()
    parser = StructureParser()
    validator = StructureValidator()

    S, Y = make_stage2_synth_dataset(
        N=5,
        tokenizer=tk,
        parser=parser,
        validator=validator,
        spec_dim=300,
        seed=0
    )

    for i in range(5):
        print("struct ids:", S[i])
        print("decode:", tk.decode(S[i]))
        print("spectrum:", Y[i][:10], "...")
