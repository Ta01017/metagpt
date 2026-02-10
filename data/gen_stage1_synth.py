# data/gen_stage1_synth.py
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure project root is on sys.path so absolute imports work from any cwd.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from structure_lang.tokenizer import StructureTokenizer
from structure_lang.parser import StructureParser
from structure_lang.validator import StructureValidator


def _choose_from_vals(vals: List[int], lo: int, hi: int) -> int:
    """从 tokenizer 离散值中选取落在 [lo, hi] 的一个值"""
    cand = [v for v in vals if lo <= v <= hi]
    if not cand:
        # fallback：取最接近区间的值
        cand = sorted(vals, key=lambda x: abs(x - (lo + hi) / 2))
        return cand[0]
    return random.choice(cand)


class Stage1SynthGenerator:
    """
    生成 Stage1 结构 token 序列（单层 metasurface unit cell：CYL/RECT）
    - 使用 tokenizer 的离散取值集合（确保 token 都在 vocab）
    - 使用 validator 做合法性过滤（确保后续 Meep 能建模）
    """

    def __init__(
        self,
        tokenizer: StructureTokenizer,
        parser: StructureParser,
        validator: StructureValidator,
        # 你可以按项目需要收紧范围（更“像论文/像工程”）
        P_range_nm: Tuple[int, int] = (400, 900),
        H_range_nm: Tuple[int, int] = (80, 1000),
        R_range_nm: Tuple[int, int] = (40, 300),
        W_range_nm: Tuple[int, int] = (60, 600),
        use_materials: List[str] | None = None,
        p_cyl: float = 0.5,
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

        self.use_materials = use_materials if use_materials is not None else tokenizer.materials
        random.seed(seed)

    def sample_one_tokens(self) -> List[str]:
        P = _choose_from_vals(self.tk.P_vals, *self.P_range)
        H = _choose_from_vals(self.tk.H_vals, *self.H_range)

        mat = random.choice(self.use_materials)

        if random.random() < self.p_cyl:
            shape = "CYL"
            # radius 上限要考虑 margin（validator 会二次过滤）
            R = _choose_from_vals(self.tk.R_vals, *self.R_range)
            tokens = [
                "PX_%d" % P, "PY_%d" % P,
                "SUB_Glass_Substrate",
                "L1_MAT_%s" % mat,
                "L1_SHAPE_CYL",
                "L1_H_%d" % H,
                "L1_R_%d" % R,
            ]
        else:
            shape = "RECT"
            W = _choose_from_vals(self.tk.W_vals, *self.W_range)
            L = _choose_from_vals(self.tk.W_vals, *self.W_range)
            tokens = [
                "PX_%d" % P, "PY_%d" % P,
                "SUB_Glass_Substrate",
                "L1_MAT_%s" % mat,
                "L1_SHAPE_RECT",
                "L1_H_%d" % H,
                "L1_W_%d" % W,
                "L1_L_%d" % L,
            ]

        return tokens

    def sample_one_ids(self, max_tries: int = 200) -> List[int]:
        """
        返回一条“合法”的 token-id 序列（不含 BOS/EOS）
        """
        for _ in range(max_tries):
            toks = self.sample_one_tokens()
            struct = self.parser.parse(["[BOS]"] + toks + ["[EOS]"])
            ok, reason = self.val.validate(struct)
            if not ok:
                continue
            # 确保 token 都在 vocab（避免你未来扩展时漏定义）
            try:
                ids = self.tk.encode(toks)
            except KeyError:
                continue
            return ids
        raise RuntimeError("Failed to sample a valid structure within max_tries.")

    def make_dataset_ids(self, N: int) -> List[List[int]]:
        out = []
        for _ in range(N):
            out.append(self.sample_one_ids())
        return out


if __name__ == "__main__":
    tk = StructureTokenizer()
    parser = StructureParser()
    val = StructureValidator(min_feature_nm=20, margin_nm=30)

    gen = Stage1SynthGenerator(
        tokenizer=tk, parser=parser, validator=val,
        P_range_nm=(400, 900),
        H_range_nm=(80, 1000),
        R_range_nm=(40, 250),
        W_range_nm=(60, 550),
        use_materials=["SiO2", "TiO2", "Ta2O5", "HfO2"],  # 先介质-only更稳
        p_cyl=0.5,
        seed=0
    )

    seqs = gen.make_dataset_ids(5)
    for s in seqs:
        print(s, "->", tk.decode(s))
