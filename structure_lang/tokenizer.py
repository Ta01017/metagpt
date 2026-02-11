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

class StructureTokenizerExtended(StructureTokenizer):
    """扩展的StructureTokenizer，支持实际数据参数范围"""
    
    def __init__(self):
        super().__init__()
        
        # 扩展参数范围以适配实际数据
        self._extend_parameter_ranges()
        self._rebuild_vocab_with_extended_ranges()
    
    def _extend_parameter_ranges(self):
        """扩展参数范围"""
        # 基于失败样本分析，扩展参数范围
        print("🔧 扩展参数范围以适配实际数据...")
        
        # 扩展P范围：支持更小的周期
        original_P_min = min(self.P_vals)
        if original_P_min > 50:  # 如果最小P大于50nm，添加更小的值
            self.P_vals = [50] + self.P_vals
            print(f"  P范围扩展: 添加50nm")
        
        # 扩展R范围：支持更小的半径
        original_R_min = min(self.R_vals) 
        if original_R_min > 30:  # 支持小到30nm的半径
            new_R_vals = [30, 40] + self.R_vals
            self.R_vals = sorted(list(set(new_R_vals)))  # 去重并排序
            print(f"  R范围扩展: 添加30nm, 40nm")
        
        # 扩展H范围：支持更大的高度
        original_H_max = max(self.H_vals)
        if original_H_max < 800:  # 如果最大H小于800nm，添加更大的值
            additional_H = [h for h in range(original_H_max + 50, 1001, 50)]
            self.H_vals.extend(additional_H)
            self.H_vals = sorted(list(set(self.H_vals)))
            print(f"  H范围扩展: 最大到{max(self.H_vals)}nm")
        
        print(f"  最终参数范围:")
        print(f"    P_vals: {self.P_vals}")
        print(f"    R_vals: {self.R_vals}") 
        print(f"    H_vals: {self.H_vals}")
    
    def _rebuild_vocab_with_extended_ranges(self):
        """使用扩展的参数范围重新构建词表"""
        # 保存特殊token的ID
        special_ids = {token: self.vocab[token] for token in self.special_tokens}
        
        # 重新构建词表
        self.vocab = {}
        self.inv_vocab = {}
        idx = 0
        

        # 特殊token
        for t in self.special_tokens:
            self.vocab[t] = idx; idx += 1

        # PX, PY (使用扩展后的P_vals)
        for P in self.P_vals:
            self.vocab[f"PX_{P}"] = idx; idx += 1
            self.vocab[f"PY_{P}"] = idx; idx += 1

        # substrate
        self.vocab["SUB_Glass_Substrate"] = idx; idx += 1

        # materials (包含Si-Alpha)
        self.materials = ["SiO2", "TiO2", "Si-Alpha"]  # 确保包含Si-Alpha
        for m in self.materials:
            self.vocab[f"L1_MAT_{m}"] = idx; idx += 1

        # shapes
        shapes = ["CYL", "RECT"]
        for sh in shapes:
            self.vocab[f"L1_SHAPE_{sh}"] = idx; idx += 1

        # height (使用扩展后的H_vals)
        for H in self.H_vals:
            self.vocab[f"L1_H_{H}"] = idx; idx += 1

        # CYL radius (使用扩展后的R_vals)
        for R in self.R_vals:
            self.vocab[f"L1_R_{R}"] = idx; idx += 1

        # RECT width/length
        for W in self.W_vals:
            self.vocab[f"L1_W_{W}"] = idx; idx += 1
            self.vocab[f"L1_L_{W}"] = idx; idx += 1

        # CoT tokens
        self.cot_tokens = ["[COT]"]
        self.cot_tokens += [f"COT_MAT_{m}" for m in self.materials]
        self.cot_tokens += ["COT_SHAPE_CYL", "COT_SHAPE_RECT"]
        for t in self.cot_tokens:
            if t not in self.vocab:
                self.vocab[t] = idx; idx += 1

        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        
        # 重新设置常用ID
        self.pad_id = self.vocab["[PAD]"]
        self.bos_id = self.vocab["[BOS]"]
        self.eos_id = self.vocab["[EOS]"]
        self.vocab_size = len(self.vocab)
        
        print(f"  词表大小: {self.vocab_size}")


        # 特殊token
        for t in self.special_tokens:
            self.vocab[t] = idx; idx += 1

        # PX, PY (使用扩展后的P_vals)
        for P in self.P_vals:
            self.vocab[f"PX_{P}"] = idx; idx += 1
            self.vocab[f"PY_{P}"] = idx; idx += 1

        # substrate
        self.vocab["SUB_Glass_Substrate"] = idx; idx += 1

        # materials (包含Si-Alpha)
        self.materials = ["SiO2", "TiO2", "Si-Alpha"]  # 确保包含Si-Alpha
        for m in self.materials:
            self.vocab[f"L1_MAT_{m}"] = idx; idx += 1

        # shapes
        shapes = ["CYL", "RECT"]
        for sh in shapes:
            self.vocab[f"L1_SHAPE_{sh}"] = idx; idx += 1

        # height (使用扩展后的H_vals)
        for H in self.H_vals:
            self.vocab[f"L1_H_{H}"] = idx; idx += 1

        # CYL radius (使用扩展后的R_vals)
        for R in self.R_vals:
            self.vocab[f"L1_R_{R}"] = idx; idx += 1

        # RECT width/length
        for W in self.W_vals:
            self.vocab[f"L1_W_{W}"] = idx; idx += 1
            self.vocab[f"L1_L_{W}"] = idx; idx += 1

        # CoT tokens
        self.cot_tokens = ["[COT]"]
        self.cot_tokens += [f"COT_MAT_{m}" for m in self.materials]
        self.cot_tokens += ["COT_SHAPE_CYL", "COT_SHAPE_RECT"]
        for t in self.cot_tokens:
            if t not in self.vocab:
                self.vocab[t] = idx; idx += 1

        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        
        # 重新设置常用ID
        self.pad_id = self.vocab["[PAD]"]
        self.bos_id = self.vocab["[BOS]"]
        self.eos_id = self.vocab["[EOS]"]
        self.vocab_size = len(self.vocab)
        
        print(f"  词表大小: {self.vocab_size}")


    def analyze_parameter_distribution(self, folder_path: str):
        """分析实际数据中的参数分布"""
        import glob
        import re
        from pathlib import Path
        
        mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
        print(f"\n📊 分析 {len(mat_files)} 个文件的参数分布...")
        
        all_P = []
        all_R = []  # 半径 = D/2
        all_H = []
        
        for file_path in mat_files[:1000]:  # 抽样分析前1000个文件
            try:
                filename = Path(file_path).name
                pattern = r'T_P([\d\.e+-]+)_D([\d\.e+-]+)_H([\d\.e+-]+)_num-idx(\d+)\.mat'
                match = re.match(pattern, filename)
                
                if match:
                    P, D, H, idx = match.groups()
                    P_nm = float(P) * 1e9
                    R_nm = (float(D) / 2) * 1e9  # 直径转半径
                    H_nm = float(H) * 1e9
                    
                    all_P.append(P_nm)
                    all_R.append(R_nm)
                    all_H.append(H_nm)
            except:
                continue
        
        if all_P:
            print(f"  P范围: {min(all_P):.1f} - {max(all_P):.1f} nm")
            print(f"  R范围: {min(all_R):.1f} - {max(all_R):.1f} nm") 
            print(f"  H范围: {min(all_H):.1f} - {max(all_H):.1f} nm")
            
            # 检查超出范围的样本
            p_out_of_range = [p for p in all_P if p < min(self.P_vals) or p > max(self.P_vals)]
            r_out_of_range = [r for r in all_R if r < min(self.R_vals) or r > max(self.R_vals)]
            h_out_of_range = [h for h in all_H if h < min(self.H_vals) or h > max(self.H_vals)]
            
            print(f"  超出当前范围的样本:")
            print(f"    P: {len(p_out_of_range)}/{len(all_P)}")
            print(f"    R: {len(r_out_of_range)}/{len(all_R)}")
            print(f"    H: {len(h_out_of_range)}/{len(all_H)}")