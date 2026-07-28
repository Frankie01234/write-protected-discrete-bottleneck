# 写保护离散瓶颈：复现笔记

> 论文：https://arxiv.org/abs/2607.08312  
> 等级：B（论文子集，按论文自实现；无官方代码）

## 实现口径

| 模块 | 本仓库 |
|------|--------|
| 环境 | 彩色物体 Grid World（5×5 / 7×7），非 MuJoCo |
| 编码器 | 轻量 CNN + TinyVAE（确定性 μ） |
| 负结果 | 可训练 Gumbel-softmax + 语言 grounding 头 |
| Layer1 | `z.detach()` + 冻结正交投影瓶颈 |
| Layer2 | Memory Table：`Dict[symbol → Counter[label]]` |
| Layer3 | 简化 DP-Means 冲突分裂 |
| 明确不做 | V-JEPA/CLIP、真实机器人、MuJoCo 全家桶 |

## 超参（默认）

- codebook / 符号数：64
- z 维：32
- Gumbel τ：vanilla=0.5；high-temp=2.0；entropy λ=0.1
- 物理预训练：位置分类 + 平均色回归（无语言标签）
- 交互步数：400–800（随 run 配置）

## 三组实测摘要（本机 CPU）

| Run | 设置 | Gumbel/anti 最高 acc | Full Memory | 无 DP-Means |
|-----|------|----------------------|-------------|-------------|
| run01 | 5×5, 6 obj, seed7 | 0.333 | **1.000** | 0.500 |
| run02 | 7×7, 6 obj, seed42 | 0.167 | **1.000** | 0.167 |
| run03 | 7×7, 24 obj, seed123 | 0.083 | **1.000** | 0.125 |

解读：

1. **负结果方向成立**：端到端 Gumbel / anti-collapse 的 grounding 接近随机或远低于写保护路径；本子集未稳定复现「vanilla → 2/64 符号」极端坍缩，但清晰复现「有多样性仍难语义化」。
2. **三层齐全有效**：Memory Table + 写保护在三组数据上均达 100% holdout grounding。
3. **Layer3 必要**：去掉 DP-Means 后准确率大幅下降（尤其 24 物体：1.000 → 0.125），方向对齐论文 97.2% vs 22.2%。

绝对数值**不可**与论文表格直接对比。

## 设计原则

语言可命名与绑定符号，但不得反向写坏物理符号形成层——「写保护边界」。
