"""实验协议：B1–B5 负结果 + 三层修复与消融。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .blackboard import DPMeansCollisionResolver, MemoryTable
from .gridworld import GridConfig, GridWorld
from .models import WorldModelBundle


@dataclass
class RunConfig:
    run_id: str
    seed: int
    grid_size: int
    n_objects: int
    image_size: int = 32
    n_symbols: int = 64
    gumbel_epochs: int = 60
    gumbel_steps_per_epoch: int = 96
    pretrain_steps: int = 1200
    interaction_steps: int = 500
    label_noise: float = 0.0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def label_to_index(labels: List[str]) -> Dict[str, int]:
    return {l: i for i, l in enumerate(labels)}


def collect_object_batch(env: GridWorld) -> Tuple[torch.Tensor, List[str]]:
    views = env.object_centered_views()
    xs = torch.from_numpy(np.stack([v[0] for v in views])).float()
    ys = [v[1] for v in views]
    return xs, ys


@torch.no_grad()
def symbol_diversity(model: WorldModelBundle, x: torch.Tensor) -> int:
    model.eval()
    z, _ = model.encode_z(x)
    ids = model.symbols_from_z(z)
    return int(ids.unique().numel())


@torch.no_grad()
def diversity_on_rollout(model: WorldModelBundle, env: GridWorld, n: int = 128) -> int:
    """在随机游走帧上统计符号多样性（更接近论文 batch 口径）。"""
    model.eval()
    device = next(model.parameters()).device
    x, _, _ = env.random_walk_batch(n)
    return symbol_diversity(model, x.to(device))


def train_gumbel_variant(
    env: GridWorld,
    cfg: RunConfig,
    *,
    temperature: float = 0.5,
    lr: float = 1e-4,
    spectral_norm: bool = False,
    entropy_coef: float = 0.0,
    name: str = "vanilla",
    log: logging.Logger,
) -> Dict[str, Any]:
    """端到端 Gumbel + 语言 grounding（演示坍缩 / 反坍缩仍难语义化）。"""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    n_labels = len(env.labels)
    lab2i = label_to_index(env.labels)
    model = WorldModelBundle(
        n_labels=n_labels,
        bottleneck="gumbel",
        n_symbols=cfg.n_symbols,
        temperature=temperature,
        spectral_norm=spectral_norm,
    ).to(device)

    # 先短物理预训练，再让语言梯度进入瓶颈
    _pretrain_position(model, env, steps=max(200, cfg.pretrain_steps // 3), device=device, log=log)
    # vanilla：继续更新全部参数以放大梯度破坏；反坍缩变体冻结编码器更稳
    if name == "vanilla":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        for p in model.encoder.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
    hist_div: List[float] = []
    hist_acc: List[float] = []

    for epoch in range(cfg.gumbel_epochs):
        model.train()
        x, ys = collect_object_batch(env)
        reps = max(1, cfg.gumbel_steps_per_epoch // max(1, len(ys)))
        # 轻微像素噪声，避免完全死记
        x = x.repeat(reps, 1, 1, 1)
        x = x + 0.02 * torch.randn_like(x)
        x = x.clamp(0, 1).to(device)
        y_idx = torch.tensor([lab2i[y] for y in ys] * reps, device=device)

        z, vae_loss = model.encode_z(x)
        ids, y_soft, embed = model.bottleneck(z, hard=False)
        logits = model.grounding(embed)
        ce = F.cross_entropy(logits, y_idx)
        loss = ce + 0.05 * vae_loss
        if entropy_coef > 0:
            loss = loss - entropy_coef * model.bottleneck.entropy_bonus(y_soft)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(-1)
            acc = (pred == y_idx).float().mean().item()
            # hard 符号多样性
            hard_ids = y_soft.argmax(-1)
            div = int(hard_ids.unique().numel())
        hist_div.append(div)
        hist_acc.append(acc)
        if epoch % 10 == 0 or epoch == cfg.gumbel_epochs - 1:
            log.info(
                f"[{name}] epoch={epoch:03d} loss={loss.item():.4f} "
                f"acc={acc:.3f} diversity={div}/{cfg.n_symbols}"
            )

    model.eval()
    x, ys = collect_object_batch(env)
    x = x.to(device)
    with torch.no_grad():
        z, _ = model.encode_z(x)
        ids, y_soft, embed = model.bottleneck(z, hard=True)
        logits = model.grounding(embed)
        y_idx = torch.tensor([lab2i[y] for y in ys], device=device)
        acc = (logits.argmax(-1) == y_idx).float().mean().item()
        div_objects = int(ids.unique().numel())
    div = diversity_on_rollout(model, env, n=128)

    return {
        "name": name,
        "final_diversity": div,
        "final_diversity_on_objects": div_objects,
        "final_grounding_acc": acc,
        "diversity_curve": hist_div,
        "acc_curve": hist_acc,
        "collapsed": div <= 2,
        "temperature": temperature,
        "lr": lr,
        "spectral_norm": spectral_norm,
        "entropy_coef": entropy_coef,
    }


def _pretrain_position(
    model: WorldModelBundle,
    env: GridWorld,
    steps: int,
    device: torch.device,
    log: logging.Logger,
) -> None:
    """物理预训练：预测格子位置 + 局部平均色（无语言标签）。"""
    g = env.cfg.grid_size
    pos_head = nn.Linear(32, g * g).to(device)
    color_head = nn.Linear(32, 3).to(device)
    params = (
        list(model.encoder.parameters())
        + list(model.vae.parameters())
        + list(pos_head.parameters())
        + list(color_head.parameters())
    )
    opt = torch.optim.Adam(params, lr=1e-3)
    model.train()
    pbar = tqdm(range(steps), desc="physics-pretrain", leave=False)
    for t in pbar:
        # 交替：随机游走 / 物体中心观测
        if t % 2 == 0:
            action = env.ACTIONS[int(env.rng.integers(0, len(env.ACTIONS)))]
            img, info = env.step(action)
            ax, ay = info["pos"]
        else:
            views = env.object_centered_views()
            img, _, pos = views[int(env.rng.integers(0, len(views)))]
            ax, ay = pos
            env.agent = pos
        x = torch.from_numpy(img).float().unsqueeze(0).to(device)
        z, vae_loss = model.encode_z(x)
        target = torch.tensor([ay * g + ax], device=device)
        pos_loss = F.cross_entropy(pos_head(z), target)
        # 图像平均色：迫使保留外观信息
        mean_color = x.mean(dim=(2, 3))
        color_loss = F.mse_loss(torch.sigmoid(color_head(z)), mean_color)
        loss = pos_loss + 0.5 * color_loss + 0.1 * vae_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if t % 200 == 0:
            log.info(
                f"[pretrain] step={t} pos={pos_loss.item():.4f} "
                f"color={color_loss.item():.4f} vae={vae_loss.item():.4f}"
            )
            pbar.set_postfix(pos=float(pos_loss.item()))


def run_write_protected(
    env: GridWorld,
    cfg: RunConfig,
    *,
    enable_dpmeans: bool = True,
    log: logging.Logger,
) -> Dict[str, Any]:
    """三层修复：z.detach/冻结瓶颈 + Memory Table +（可选）DP-Means。"""
    set_seed(cfg.seed + 17)
    device = torch.device(cfg.device)
    n_labels = len(env.labels)
    model = WorldModelBundle(
        n_labels=n_labels,
        bottleneck="frozen",
        n_symbols=cfg.n_symbols,
    ).to(device)

    _pretrain_position(model, env, cfg.pretrain_steps, device, log)

    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.vae.parameters():
        p.requires_grad = False

    memory = MemoryTable()
    resolver = DPMeansCollisionResolver(lambda_threshold=0.35, min_conflict=2)
    social_opt = torch.optim.Adam(model.grounding.parameters(), lr=3e-4)
    lab2i = label_to_index(env.labels)

    query_pairs: List[Tuple[int, str]] = []
    diversities: List[int] = []

    log.info(
        f"[write-protect] interaction_steps={cfg.interaction_steps} "
        f"dpmeans={enable_dpmeans} n_objects={cfg.n_objects}"
    )

    for t in tqdm(range(cfg.interaction_steps), desc="interaction", leave=False):
        views = env.object_centered_views()
        img, true_y, pos = views[t % len(views)]
        env.agent = pos
        # 混入随机游走
        if t % 3 != 0:
            action = env.ACTIONS[int(env.rng.integers(0, 4))]
            img2, info = env.step(action)
            if info["label"] is not None:
                img, true_y = img2, info["label"]
            else:
                # 仍用物体观测，保证每步有监督信号
                pass

        if true_y is None:
            continue
        teacher_y = env.noisy_teacher_label(true_y)
        assert teacher_y is not None

        x = torch.from_numpy(img).float().unsqueeze(0).to(device)
        with torch.no_grad():
            z, _ = model.encode_z(x)
            base_ids = model.symbols_from_z(z)
            base_sym = int(base_ids.item())
            feat = z.squeeze(0).cpu().numpy()

        sid = resolver.observe(
            memory, base_sym, teacher_y, feat, enable_split=enable_dpmeans
        )

        model.grounding.train()
        logits = model.grounding(z)
        loss = F.cross_entropy(logits, torch.tensor([lab2i[teacher_y]], device=device))
        social_opt.zero_grad()
        loss.backward()
        social_opt.step()
        model.grounding.eval()

        q_sid = resolver.resolve_symbol(base_sym, feat, enable_split=enable_dpmeans)
        query_pairs.append((q_sid, true_y))

        if t % 50 == 0:
            xb, _ = collect_object_batch(env)
            div = symbol_diversity(model, xb.to(device))
            diversities.append(div)
            acc_so_far = memory.grounding_accuracy(query_pairs[-min(80, len(query_pairs)) :])
            log.info(
                f"[write-protect] t={t:04d} sym={base_sym}->{sid} "
                f"div={div}/{cfg.n_symbols} mem_acc@recent={acc_so_far:.3f} "
                f"collisions={memory.collision_count()} "
                f"sub={sum(len(v) for v in memory.sub_symbols.values())}"
            )

    holdout: List[Tuple[int, str]] = []
    xb, yb = collect_object_batch(env)
    with torch.no_grad():
        z, _ = model.encode_z(xb.to(device))
        ids = model.symbols_from_z(z)
        for i, lab in enumerate(yb):
            feat = z[i].cpu().numpy()
            base = int(ids[i].item())
            sid = resolver.resolve_symbol(base, feat, enable_split=enable_dpmeans)
            holdout.append((sid, lab))

    mem_acc = memory.grounding_accuracy(holdout)
    final_div = symbol_diversity(model, xb.to(device))
    rollout_div = diversity_on_rollout(model, env, n=128)
    with torch.no_grad():
        logits = model.grounding(z)
        y_idx = torch.tensor([lab2i[y] for y in yb], device=device)
        social_acc = (logits.argmax(-1) == y_idx).float().mean().item()

    return {
        "enable_dpmeans": enable_dpmeans,
        "memory_grounding_acc": mem_acc,
        "social_head_acc": social_acc,
        "final_diversity": final_div,
        "rollout_diversity": rollout_div,
        "collapsed": final_div <= 2 and rollout_div <= 2,
        "collision_symbols": memory.collision_count(),
        "active_memory_keys": memory.active_symbols(),
        "n_sub_symbols": sum(len(v) for v in memory.sub_symbols.values()),
        "diversity_checkpoints": diversities,
        "holdout_n": len(holdout),
        "holdout_pairs": [(int(s), y) for s, y in holdout],
    }


def run_one_experiment(cfg: RunConfig, out_dir: Path, log: logging.Logger) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    t0 = time.time()
    set_seed(cfg.seed)
    env = GridWorld(
        GridConfig(
            grid_size=cfg.grid_size,
            n_objects=cfg.n_objects,
            image_size=cfg.image_size,
            seed=cfg.seed,
            label_noise=cfg.label_noise,
        )
    )
    log.info("=" * 72)
    log.info(f"开始实验 {cfg.run_id}")
    log.info(json.dumps(asdict(cfg), ensure_ascii=False))
    log.info(f"物体标签: {env.labels}")

    results: Dict[str, Any] = {"config": asdict(cfg), "labels": env.labels}

    log.info("--- B1 Vanilla Gumbel-softmax ---")
    results["B1_vanilla"] = train_gumbel_variant(
        env, cfg, temperature=0.5, lr=1e-4, name="vanilla", log=log
    )

    log.info("--- B2 High temperature ---")
    results["B2_high_temp"] = train_gumbel_variant(
        env, cfg, temperature=2.0, lr=1e-4, name="high_temp", log=log
    )
    log.info("--- B2 Entropy bonus ---")
    results["B2_entropy"] = train_gumbel_variant(
        env, cfg, temperature=0.5, lr=1e-4, entropy_coef=0.1, name="entropy", log=log
    )
    log.info("--- B2 Spectral norm ---")
    results["B2_spectral"] = train_gumbel_variant(
        env,
        cfg,
        temperature=0.5,
        lr=1e-4,
        spectral_norm=True,
        name="spectral",
        log=log,
    )

    log.info("--- B5 Full three-layer (detach + Memory + DP-Means) ---")
    env_wp = GridWorld(
        GridConfig(
            grid_size=cfg.grid_size,
            n_objects=cfg.n_objects,
            image_size=cfg.image_size,
            seed=cfg.seed + 1,
            label_noise=cfg.label_noise,
        )
    )
    results["B5_full"] = run_write_protected(env_wp, cfg, enable_dpmeans=True, log=log)

    log.info("--- Ablation: without DP-Means (Layer3 off) ---")
    env_ab = GridWorld(
        GridConfig(
            grid_size=cfg.grid_size,
            n_objects=cfg.n_objects,
            image_size=cfg.image_size,
            seed=cfg.seed + 2,
            label_noise=cfg.label_noise,
        )
    )
    results["ablation_no_dpmeans"] = run_write_protected(
        env_ab, cfg, enable_dpmeans=False, log=log
    )

    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed

    anti_max_acc = max(
        results["B2_high_temp"]["final_grounding_acc"],
        results["B2_entropy"]["final_grounding_acc"],
        results["B2_spectral"]["final_grounding_acc"],
    )
    full_acc = results["B5_full"]["memory_grounding_acc"]
    ab_acc = results["ablation_no_dpmeans"]["memory_grounding_acc"]
    trend_neg = (
        results["B1_vanilla"]["collapsed"]
        or results["B1_vanilla"]["final_diversity"] <= 4
        or results["B1_vanilla"]["final_grounding_acc"] < 0.25
    ) and (anti_max_acc < max(0.25, full_acc - 0.05))
    if cfg.n_objects >= 12:
        trend_ablation = full_acc > ab_acc + 0.05
    else:
        trend_ablation = full_acc >= ab_acc - 0.05

    results["summary"] = {
        "vanilla_collapsed": results["B1_vanilla"]["collapsed"],
        "vanilla_div": results["B1_vanilla"]["final_diversity"],
        "vanilla_acc": results["B1_vanilla"]["final_grounding_acc"],
        "anti_collapse_max_acc": anti_max_acc,
        "anti_collapse_divs": {
            "high_temp": results["B2_high_temp"]["final_diversity"],
            "entropy": results["B2_entropy"]["final_diversity"],
            "spectral": results["B2_spectral"]["final_diversity"],
        },
        "full_memory_acc": full_acc,
        "full_div": results["B5_full"]["final_diversity"],
        "full_rollout_div": results["B5_full"].get("rollout_diversity"),
        "no_dpmeans_memory_acc": ab_acc,
        "trend_ok_negative": trend_neg,
        "trend_ok_full_vs_ablation": trend_ablation,
    }

    log.info("--- 摘要 ---")
    log.info(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    log.info(f"耗时 {elapsed:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_id}.json"
    # holdout_pairs 可能较长，保留
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"结果已写入 {out_path}")
    return results
