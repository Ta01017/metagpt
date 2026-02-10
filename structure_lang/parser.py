# structure_lang/parser.py
from typing import List, Dict


class StructureParser:
    """
    将 token 序列（字符串列表）解析为结构参数 dict
    用于后续：
      - Meep 3D 建模
      - 保存结构记录
      - 可视化
      - RL 奖励计算
    """

    def __init__(self):
        pass

    def parse(self, tokens: List[str]) -> Dict:
        # 去掉 PAD/BOS/EOS
        toks = [t for t in tokens if not t.startswith("[")]

        out = {
            "P": [None, None],
            "substrate": "Glass_Substrate",
            "layer1": {
                "mat": None,
                "shape": None,
                "h_nm": None,
                "r_nm": None,         # CYL
                "w_nm": None,         # RECT
                "l_nm": None,
                "center": (0,0),
                "rot_deg": 0
            }
        }

        for t in toks:
            if t.startswith("PX_"):
                out["P"][0] = int(t.split("_")[1])
            elif t.startswith("PY_"):
                out["P"][1] = int(t.split("_")[1])

            elif t.startswith("L1_MAT_"):
                out["layer1"]["mat"] = t.split("L1_MAT_")[1]

            elif t.startswith("L1_SHAPE_"):
                out["layer1"]["shape"] = t.split("L1_SHAPE_")[1]

            elif t.startswith("L1_H_"):
                out["layer1"]["h_nm"] = int(t.split("_")[2])

            elif t.startswith("L1_R_"):
                out["layer1"]["r_nm"] = int(t.split("_")[2])

            elif t.startswith("L1_W_"):
                out["layer1"]["w_nm"] = int(t.split("_")[2])

            elif t.startswith("L1_L_"):
                out["layer1"]["l_nm"] = int(t.split("_")[2])

        return out
