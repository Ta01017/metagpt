class StructureTokenizer:
    def __init__(self):
        self.materials = [
            'MgF2','SiO2','ZnO','MgO','Si3N4','HfO2',
            'TiO2','Ta2O5','AlN','Nb2O5','ZnS','ZnSe'
        ]

        # discrete ranges
        self.P_vals = list(range(300, 1001, 10))    # PX/PY
        self.H_vals = list(range(50, 1201, 10))     # height
        self.R_vals = list(range(20, 501, 10))      # radius
        self.W_vals = list(range(20, 801, 10))      # width/length

        self.special_tokens = ["[PAD]","[BOS]","[EOS]"]

        self.vocab = {}
        self.inv_vocab = {}
        self._build_vocab()

    def _build_vocab(self):
        idx = 0
        for t in self.special_tokens:
            self.vocab[t] = idx; idx += 1

        for P in self.P_vals:
            self.vocab[f"PX_{P}"] = idx; idx += 1
            self.vocab[f"PY_{P}"] = idx; idx += 1

        self.vocab["SUB_Glass_Substrate"] = idx; idx += 1

        for m in self.materials:
            self.vocab[f"L1_MAT_{m}"] = idx; idx += 1
        self.vocab["L1_SHAPE_CYL"] = idx; idx += 1
        self.vocab["L1_SHAPE_RECT"] = idx; idx += 1

        for H in self.H_vals:
            self.vocab[f"L1_H_{H}"] = idx; idx += 1
        for R in self.R_vals:
            self.vocab[f"L1_R_{R}"] = idx; idx += 1
        for W in self.W_vals:
            self.vocab[f"L1_W_{W}"] = idx; idx += 1
            self.vocab[f"L1_L_{W}"] = idx; idx += 1

        self.inv_vocab = {v:k for k,v in self.vocab.items()}

    def encode(self, struct_tokens: list) -> list:
        return [self.vocab[t] for t in struct_tokens]

    def decode(self, ids: list) -> list:
        return [self.inv_vocab[i] for i in ids]
