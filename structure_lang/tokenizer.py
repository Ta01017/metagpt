# structure_lang/tokenizer.py
import json


class StructureTokenizer:
    """
    将 metasurface 单元结构序列映射到 token id（整数）
    支持：
      - encode: tokens → ids
      - decode: ids → tokens
      - 保存与加载词表
    """

    def __init__(self):
        self.materials = [
            'MgF2','SiO2','ZnO','MgO','Si3N4','HfO2',
            'TiO2','Ta2O5','AlN','Nb2O5','ZnS','ZnSe'
        ]

        # 参数空间（可调）
        self.P_vals = list(range(300, 1001, 10))    # PX/PY
        self.H_vals = list(range(50, 1201, 10))     # height
        self.R_vals = list(range(20, 501, 10))      # radius
        self.W_vals = list(range(20, 801, 10))      # width/length

        self.special_tokens = ["[PAD]","[BOS]","[EOS]"]

        self.vocab = {}
        self.inv_vocab = {}
        self._build_vocab()

        # expose common ids/sizes
        self.pad_id = self.vocab["[PAD]"]
        self.bos_id = self.vocab["[BOS]"]
        self.eos_id = self.vocab["[EOS]"]
        self.vocab_size = len(self.vocab)

    # --------------------------------------------------------
    # Build vocab
    # --------------------------------------------------------
    def _build_vocab(self):
        idx = 0
        for t in self.special_tokens:
            self.vocab[t] = idx; idx += 1

        # PX, PY
        for P in self.P_vals:
            self.vocab[f"PX_{P}"] = idx; idx += 1
            self.vocab[f"PY_{P}"] = idx; idx += 1

        # substrate
        self.vocab["SUB_Glass_Substrate"] = idx; idx += 1

        # materials
        for m in self.materials:
            self.vocab[f"L1_MAT_{m}"] = idx; idx += 1

        # shapes
        shapes = ["CYL", "RECT"]
        for sh in shapes:
            self.vocab[f"L1_SHAPE_{sh}"] = idx; idx += 1

        # height
        for H in self.H_vals:
            self.vocab[f"L1_H_{H}"] = idx; idx += 1

        # CYL radius
        for R in self.R_vals:
            self.vocab[f"L1_R_{R}"] = idx; idx += 1

        # RECT width/length
        for W in self.W_vals:
            self.vocab[f"L1_W_{W}"] = idx; idx += 1
            self.vocab[f"L1_L_{W}"] = idx; idx += 1

        # CoT tokens (append to preserve existing ids)
        self.cot_tokens = ["[COT]"]
        self.cot_tokens += [f"COT_MAT_{m}" for m in self.materials]
        self.cot_tokens += ["COT_SHAPE_CYL", "COT_SHAPE_RECT"]
        for t in self.cot_tokens:
            if t not in self.vocab:
                self.vocab[t] = idx; idx += 1

        self.inv_vocab = {v:k for k,v in self.vocab.items()}

    # --------------------------------------------------------
    # Encode / Decode
    # --------------------------------------------------------
    def encode(self, tokens):
        return [self.vocab[t] for t in tokens]

    def decode(self, ids):
        return [self.inv_vocab[i] for i in ids]

    # --------------------------------------------------------
    # Save / Load
    # --------------------------------------------------------
    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    @staticmethod
    def load_vocab(path):
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tk = StructureTokenizer()
        tk.vocab = vocab
        tk.inv_vocab = {v:k for k,v in vocab.items()}
        return tk
