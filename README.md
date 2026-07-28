# write-protected-discrete-bottleneck

写保护离散瓶颈：修复语言梯度破坏世界模型符号层的结构问题。

- **论文：** [Write-Protected Discrete Bottlenecks for Language-Grounded World Models](https://arxiv.org/abs/2607.08312)
- **复现方案：** [复现方案.md](./复现方案.md)（等级 **B**，按论文自实现；无官方代码）
- **笔记：** [notes/write-protect.md](./notes/write-protect.md)

## 本仓库做了什么

1. 实现最小 Grid World + CNN/VAE + Gumbel / 冻结正交离散瓶颈。
2. 复现结构性负结果：端到端语言梯度进入离散瓶颈 → 坍缩或多样性无语义。
3. 实现三层修复：`z.detach()`、Memory Table、DP-Means 冲突分裂，并做去第三层消融。
4. 在三组不同数据（网格大小 / 物体数 / seed）上跑通实验，写出完整日志与 JSON。

## 快速复现

建议使用短路径虚拟环境（Windows 长路径限制）：

```bash
python -m venv F:\v\wpdb
F:\v\wpdb\Scripts\pip install -r requirements.txt
F:\v\wpdb\Scripts\python scripts/run_experiments.py
```

输出：

- `logs/run0*.log`、`logs/MASTER_RUN.md` 完整日志
- `results/run0*.json` 指标明细
- `results/summary_three_runs.json` 三组对照

## 指标口径（诚实声明）

| 论文 | 本仓库 |
|------|--------|
| V-JEPA / CLIP / MuJoCo | CNN + Grid World 子集 |
| 97.2% vs 22.2%（36 物体） | 方向对齐；绝对值不可直接对比 |
| 74 runs / 多 seed | 每配置单 seed × 三组数据 |

## 明确未做

- 真实机器人与端到端 LLM 微调对照
- 官方权重（论文未发布代码）
- 完整 V-JEPA/CLIP 编码器装机
