# fixed_dataset_builder.py
import scipy.io
import numpy as np
import pickle
import re
import os
from pathlib import Path
import glob
from structure_lang.tokenizer import StructureTokenizerExtended

class FixedSiAlphaDatasetBuilder:
    def __init__(self, target_spec_dim: int = 128):
        self.target_spec_dim = target_spec_dim
        self.tokenizer = StructureTokenizerExtended()
        self.material = "Si-Alpha"
        
        print(f"✅ 使用扩展的tokenizer，支持更广的参数范围")
        print(f"  材料: {self.material}")
        print(f"  词表大小: {self.tokenizer.vocab_size}")
    
    def parse_filename(self, filename: str) -> dict:
        """从文件名解析结构参数"""
        pattern = r'T_P([\d\.e+-]+)_D([\d\.e+-]+)_H([\d\.e+-]+)_num-idx(\d+)\.mat'
        match = re.match(pattern, filename)
        
        if match:
            P, D, H, idx = match.groups()
            return {
                'P': float(P),
                'D': float(D), 
                'H': float(H),
                'idx': int(idx)
            }
        else:
            raise ValueError(f"无法解析文件名: {filename}")
    
    def process_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """处理光谱数据"""
        original_dim = spectrum.shape[0]
        
        if original_dim == self.target_spec_dim:
            interpolated = spectrum
        else:
            original_x = np.linspace(0, 1, original_dim)
            target_x = np.linspace(0, 1, self.target_spec_dim)
            interpolated = np.interp(target_x, original_x, spectrum)
        
        # 归一化
        spec_min, spec_max = interpolated.min(), interpolated.max()
        if spec_max - spec_min > 1e-10:
            normalized = (interpolated - spec_min) / (spec_max - spec_min)
        else:
            normalized = np.zeros_like(interpolated)
        
        return normalized.astype(np.float32)
    
    def parameters_to_tokens(self, P: float, D: float, H: float) -> list:
        """将结构参数转换为token序列"""
        # 单位转换：米 → 纳米
        P_nm = P * 1e9
        H_nm = H * 1e9
        R_nm = (D / 2) * 1e9  # 直径转半径
        
        # 量化到最接近的离散值
        P_quantized = self._quantize_to_nearest(P_nm, self.tokenizer.P_vals)
        H_quantized = self._quantize_to_nearest(H_nm, self.tokenizer.H_vals)
        R_quantized = self._quantize_to_nearest(R_nm, self.tokenizer.R_vals)
        
        # 构建token序列
        tokens = [
            f"PX_{P_quantized}",
            f"PY_{P_quantized}", 
            "SUB_Glass_Substrate",
            f"L1_MAT_{self.material}",
            "L1_SHAPE_CYL",
            f"L1_H_{H_quantized}",
            f"L1_R_{R_quantized}"
        ]
        
        token_ids = self.tokenizer.encode(tokens)
        
        # 验证量化误差（降低警告阈值到20%）
        p_error = abs(P_nm - P_quantized) / P_nm * 100 if P_nm > 0 else 0
        r_error = abs(R_nm - R_quantized) / R_nm * 100 if R_nm > 0 else 0
        h_error = abs(H_nm - H_quantized) / H_nm * 100 if H_nm > 0 else 0
        
        if max(p_error, r_error, h_error) > 20:  # 误差大于20%时警告
            print(f"⚠️  较大量化误差: P={p_error:.1f}%, R={r_error:.1f}%, H={h_error:.1f}%")
        
        return token_ids
    
    def _quantize_to_nearest(self, value: float, allowed_values: list) -> int:
        """将连续值量化到最接近的离散值"""
        return min(allowed_values, key=lambda x: abs(x - value))
    
    def validate_parameters(self, P: float, D: float, H: float) -> bool:
        """验证参数是否在扩展后的范围内"""
        P_nm = P * 1e9
        H_nm = H * 1e9
        R_nm = (D / 2) * 1e9
        
        P_valid = min(self.tokenizer.P_vals) <= P_nm <= max(self.tokenizer.P_vals)
        H_valid = min(self.tokenizer.H_vals) <= H_nm <= max(self.tokenizer.H_vals)
        R_valid = min(self.tokenizer.R_vals) <= R_nm <= max(self.tokenizer.R_vals)
        
        return P_valid and H_valid and R_valid
    
   def build_dataset(self, folder_path: str, output_prefix: str, max_samples: int = None):
        """构建数据集"""
        mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
        
        if max_samples:
            mat_files = mat_files[:max_samples]
            
        print(f"\n📁 处理文件夹: {folder_path}")
        print(f"📊 找到 {len(mat_files)} 个MAT文件")
        
        all_spectra = []
        all_tokens = []
        failed_files = []
        
        for i, file_path in enumerate(mat_files):
            if i % 1000 == 0 and i > 0:
                print(f"进度: {i}/{len(mat_files)}")
            
            try:
                filename = Path(file_path).name
                params = self.parse_filename(filename)
                
                # 验证参数范围
                if not self.validate_parameters(params['P'], params['D'], params['H']):
                    # 详细错误信息
                    P_nm = params['P'] * 1e9
                    R_nm = (params['D'] / 2) * 1e9
                    H_nm = params['H'] * 1e9
                    
                    error_msg = f"参数超出范围: P={P_nm:.1f}nm, R={R_nm:.1f}nm, H={H_nm:.1f}nm"
                    failed_files.append((filename, error_msg))
                    continue
                
                # 加载和处理数据
                mat_data = scipy.io.loadmat(file_path)
                spectrum = mat_data['T'].flatten()

                processed_spectrum = self.process_spectrum(spectrum)
                tokens = self.parameters_to_tokens(params['P'], params['D'], params['H'])
                
                all_spectra.append(processed_spectrum)
                all_tokens.append(tokens)
                
            except Exception as e:
                failed_files.append((filename, str(e)))
                continue
        
        # 保存数据
        spectra_array = np.array(all_spectra, dtype=np.float32)
        np.save(f"{output_prefix}_spec.npy", spectra_array)
        with open(f"{output_prefix}_struct.pkl", 'wb') as f:
            pickle.dump(all_tokens, f)
        
        # 统计信息
        print(f"\n✅ 处理完成:")
        print(f"   成功: {len(all_spectra)} 个样本")
        print(f"   失败: {len(failed_files)} 个样本")
        
        if failed_files:
            print(f"\n❌ 失败样本示例:")
            for filename, error in failed_files[:5]:
                print(f"   {filename}: {error}")
        
        return spectra_array, all_tokens, failed_files, len(mat_files)  # 返回失败信息和文件总数
# 使用修复后的构建器
def build_fixed_datasets(train_folder: str, val_folder: str, output_dir: str = ".", 
                        spec_dim: int = 128, test_mode: bool = False):
    """使用修复后的构建器构建数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    builder = FixedSiAlphaDatasetBuilder(target_spec_dim=spec_dim)
    
    max_samples = 100 if test_mode else None
    
    print("=" * 60)
    print("🚀 开始构建修复后的数据集")
    print("=" * 60)
    
    # 处理训练集
    train_spec, train_struct, train_failed, train_total = builder.build_dataset(
        train_folder, 
        os.path.join(output_dir, "spec_train"),
        max_samples=max_samples
    )
    
    # 处理验证集
    val_spec, val_struct, val_failed, val_total = builder.build_dataset(
        val_folder, 
        os.path.join(output_dir, "spec_val"),
        max_samples=max_samples
    )
    
    print(f"\n🎉 数据集构建成功!")
    print(f"   训练集: {train_spec.shape[0]:,} 样本")
    print(f"   验证集: {val_spec.shape[0]:,} 样本")
    print(f"   训练集失败率: {len(train_failed)/train_total*100:.1f}%")
    print(f"   验证集失败率: {len(val_failed)/val_total*100:.1f}%")
    
    return train_spec, train_struct, val_spec, val_struct

# 修复tokenizer的参数离散值设置
def create_better_tokenizer():
    """创建更合理的离散值设置"""
    
    # 基于实际数据分析，创建更密集的离散值
    print("🔧 创建更合理的离散值设置...")
    
    # 更密集的P值（周期）：从50nm到1000nm，步长10nm
    P_vals = list(range(50, 1001, 10))
    
    # 更密集的R值（半径）：从20nm到500nm，步长5nm  
    R_vals = list(range(20, 501, 5))
    
    # 更密集的H值（高度）：从50nm到1200nm，步长10nm
    H_vals = list(range(50, 1201, 10))
    
    print(f"  改进的离散值设置:")
    print(f"  P_vals: {len(P_vals)}个值, {P_vals[0]}nm - {P_vals[-1]}nm")
    print(f"  R_vals: {len(R_vals)}个值, {R_vals[0]}nm - {R_vals[-1]}nm")
    print(f"  H_vals: {len(H_vals)}个值, {H_vals[0]}nm - {H_vals[-1]}nm")
    
    return P_vals, R_vals, H_vals

# 改进的tokenizer扩展类
class ImprovedStructureTokenizerExtended(StructureTokenizerExtended):
    """改进的tokenizer，使用更合理的离散值"""


     
    def __init__(self):
        super().__init__()
        
        # 使用改进的离散值
        self.P_vals, self.R_vals, self.H_vals = create_better_tokenizer()
        
        # 重新构建词表
        self._rebuild_vocab_with_improved_ranges()
    
    def _rebuild_vocab_with_improved_ranges(self):
        """使用改进的离散值重新构建词表"""
        self.vocab = {}
        self.inv_vocab = {}
        idx = 0
        
        # 特殊token
        for t in self.special_tokens:
            self.vocab[t] = idx; idx += 1

        # PX, PY (使用改进的P_vals)
        for P in self.P_vals:
            self.vocab[f"PX_{P}"] = idx; idx += 1
            self.vocab[f"PY_{P}"] = idx; idx += 1

        # substrate
        self.vocab["SUB_Glass_Substrate"] = idx; idx += 1

        # materials
        self.materials = ["SiO2", "TiO2", "Si-Alpha"]
        for m in self.materials:
            self.vocab[f"L1_MAT_{m}"] = idx; idx += 1

        # shapes
        shapes = ["CYL", "RECT"]
        for sh in shapes:
            self.vocab[f"L1_SHAPE_{sh}"] = idx; idx += 1


        # height (使用改进的H_vals)
        for H in self.H_vals:
            self.vocab[f"L1_H_{H}"] = idx; idx += 1

        # CYL radius (使用改进的R_vals)
        for R in self.R_vals:
            self.vocab[f"L1_R_{R}"] = idx; idx += 1

        # RECT width/length (使用与R相同的范围)
        for W in self.R_vals:  # 重用R_vals的范围
            self.vocab[f"L1_W_{W}"] = idx; idx += 1
            self.vocab[f"L1_L_{W}"] = idx; idx += 1

        # CoT tokens
        self.cot_tokens = ["[COT]"]
        self.cot_tokens += [f"COT_MAT_{m}" for m in self.materials]
        self.cot_tokens += ["COT_SHAPE_CYL", "COT_SHAPE_RECT"]
        for t in self.cot_tokens:
            if t not in self.vocab:
                self.vocab[t] = idx; idx += 1

        self.inv_vocab = {v:k for k,v in self.voca