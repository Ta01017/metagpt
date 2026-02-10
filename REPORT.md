# MetaGPT项目 （微纳光谱反演）

## 1. 项目概述
目标：从光谱生成结构（MetaGPT），采用三阶段训练：
- Stage-1：结构语言建模（不使用光谱）
- Stage-2：光谱条件生成（SFT）
- Stage-3：强化学习优化（以光谱重建为奖励）

## 2. 基本模型（MetaGPT）
整体是 **Decoder-only Transformer**，输入为结构 token 序列，输出为下一 token 概率。  
为支持“光谱 -> 结构”，在输入端增加 **prefix 条件编码**：
- **SpectrumEncoder**：将光谱向量编码为若干 prefix token（长度 `prefix_len`）
- 结构生成时把 prefix 与结构 token 拼接，再进入 Transformer Decoder
- 注意力掩码支持 prefix 全可见 + 结构 token 因果自回归

核心模块组成：
- `SpectrumEncoder`：光谱编码成 prefix
- `Transformer Decoder`：生成结构 token
- `LM Head`：预测下一个 token

## 3. 编码方式（Encoding）
### 3.1 结构编码
- 使用离散 token 表示结构元素（见下表）
- 以 `[BOS] ... [EOS]` 包裹序列
- 可选加入 CoT token 作为条件先验（见下表）

**结构 token 含义对照表**

| Token 类别 | 含义 |
|-----------|------|
| `PX_*` | 结构的 x 方向周期/单元尺寸（pitch-x） |
| `PY_*` | 结构的 y 方向周期/单元尺寸（pitch-y） |
| `SUB_*` | 衬底材料类型（substrate） |
| `L1_MAT_*` | 第一层材料类型 |
| `L1_SHAPE_*` | 第一层几何形状（如 RECT/CYL） |
| `L1_H_*` | 第一层高度/厚度 |
| `L1_W_*` | 第一层宽度（矩形时） |
| `L1_L_*` | 第一层长度（矩形时） |
| `L1_R_*` | 第一层半径（圆柱时） |
| `[COT]` | Chain-of-Thought 标记，表示后续为结构先验提示 |
| `COT_MAT_*` | CoT 中的材料先验（用于引导生成） |
| `COT_SHAPE_*` | CoT 中的形状先验（用于引导生成） |

### 3.2 光谱编码
- 光谱为连续向量（长度 `spec_dim`）
- 通过 `SpectrumEncoder` 映射成 `prefix_len` 个 prefix token
- prefix 与结构 token 拼接后输入 Transformer Decoder

## 4. 三阶段训练流程

### Stage-1：结构语言预训练
**目标**：学习结构 token 的语法与基本统计分布（不依赖光谱）。  
**输入**：结构 token 序列  
**输出**：下一 token 预测  
**作用**：打好结构语言基础，为后续条件生成提供可用的语言能力。

### Stage-2：光谱条件生成（SFT）
**目标**：学习从光谱条件生成结构序列。  
**输入**：光谱 + 结构 token  
**输出**：结构 token  
**特点**：
- 使用 `SpectrumEncoder` 将光谱编码为 prefix
- 训练时引入 grammar 约束与 CoT（可选）
**作用**：建立“光谱 -> 结构”的主映射。

### Stage-3：强化学习优化（RL）
**目标**：在结构合法的前提下，让生成结构的光谱匹配更好。  
**输入**：光谱  
**输出**：结构序列  
**奖励**：
- 光谱重建误差（fake forward MSE）
- 结构合法性惩罚
- 结构长度/重复/缺失 EOS 等惩罚
**作用**：进一步减少光谱误差，提高结构质量。

## 5. 实验设置
- 数据：`dataset_stage2/spec_train.npy`，`dataset_stage2/struct_train.pkl`
- 验证脚本：`compare_stage2_stage3.py`
- 评测指标：合法率、MSE、MAE、Corr（fake forward）
- Oracle：GT结构 -> fake forward 的指标，作为理论下限

## 6. 对比实验结果

**对比汇总表（单表展示）**

| 轮次 | 方式 | 模型 | Valid | MSE | MAE | Corr |
|------|------|------|-------|-----|-----|------|
| 第一轮 | Sample | Stage-2 | 98% | 0.0283 | 0.1205 | 0.9651 |
| 第一轮 | Sample | Stage-3 | 100% | 0.0281 | 0.1186 | 0.9667 |
| 第一轮 | Greedy | Stage-2 | 100% | 0.0242 | 0.1148 | 0.9686 |
| 第一轮 | Greedy | Stage-3 | 100% | 0.0200 | 0.1013 | 0.9756 |
| 第一轮 | Oracle | GT 结构 | - | 0.0050 | 0.0499 | 0.9942 |
| 第二轮 | Sample | Stage-2 | 98% | 0.0284 | 0.1212 | 0.9646 |
| 第二轮 | Sample | Stage-3 | 100% | 0.0263 | 0.1201 | 0.9674 |
| 第二轮 | Greedy | Stage-2 | 100% | 0.0240 | 0.1148 | 0.9690 |
| 第二轮 | Greedy | Stage-3 | 100% | 0.0157 | 0.0933 | 0.9814 |
| 第二轮 | Oracle | GT 结构 | - | 0.0051 | 0.0498 | 0.9939 |

备注：第二轮加入 reward shaping（MSE + Corr + 结构长度惩罚 + 重复惩罚 + 非法原因惩罚）。

---

## 7. 当前结论
1. Stage-3 RL 能显著降低 MAE（尤其是 greedy decoding）。
2. Oracle MAE 约 0.05，当前还有可提升空间，主要差异来自结构偏差。
3. Reward shaping 对 MAE 改善有效，下一步继续 Stage‑3 训练可能进一步降低。

## 8. 后续计划
- 继续 Stage‑3 RL 训练（不改结构）
- 若提升趋缓，再考虑提升 SpectrumEncoder 或重训 Stage‑2

## 9. 实验记录（追加）
| 日期 | 训练/对比 | 主要结果 | 备注 |
|------|-----------|----------|------|
| 待补充 | Stage‑3 + 对比 | MAE 0.093 (greedy) | reward shaping |
