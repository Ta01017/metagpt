# 周报（Stage12 OptoGPT / Toy 验证）
日期：2026-02-10

## 1. 本周目标
验证 OptoGPT 风格的 Stage12（SFT + RL）在 toy 数据上能稳定达到 MAE≈0.05，确保端到端流程正确，为后续真实数据对接做准备。

## 2. 本周完成
- 统一 toy 数据集与 SFT/RL/验证的 `max_len`（结构长度 8，加 BOS/EOS 后为 10）。
- 修复验证逻辑：解码时使用 `model.generator` 投影到词表（与 OptoGPT 设计一致）。
- RL 稳定化：降低学习率、加入熵正则、启用语法约束，减少无效序列。

## 3. 实验设置
- 数据：`dataset_stage2_toy/spec_train.npy` + `dataset_stage2_toy/struct_train.pkl`
- 模型：OptoGPT `Transformer_I`（`d_model=256, n_layers=1, n_heads=8, d_ff=1024, dropout=0.05`）
- 解码：greedy，`n=50`
- 评估指标：fake forward 的 MSE / MAE / Corr

## 4. 关键结果（Toy）

| 模型 | MSE | MAE | Corr |
|------|-----|-----|------|
| Stage12-SFT | 0.005202 | 0.051040 | 0.993884 |
| Stage12-RL  | 0.003439 | 0.045071 | 0.995790 |

结论：SFT 与 RL 的 MAE 均稳定到 ~0.05 左右，RL 进一步优于 SFT，说明当前 pipeline 正确可用。

## 5. 风险与问题
- 当前结果仅在 toy 数据验证，真实数据仍需进一步确认与调参。
- 训练速度与真实数据规模可能成为瓶颈。

## 6. 下周计划
- 将相同流程迁移到真实数据评估。
- 进一步对比 Stage2/Stage3 与 Stage12 的差距，定位误差来源。
