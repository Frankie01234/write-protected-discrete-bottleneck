"""编码器、Gumbel 瓶颈与冻结正交瓶颈。"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """轻量 CNN 编码器 → 特征向量。"""

    def __init__(self, in_ch: int = 3, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class TinyVAE(nn.Module):
    """特征 → 32D 潜变量 z；训练用重参数，推理/写保护路径用确定性 μ。"""

    def __init__(self, feat_dim: int = 64, z_dim: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(), nn.Linear(64, z_dim * 2))
        self.dec = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, feat_dim))
        # 直连残差，避免早期训练把颜色信息压没
        self.skip = nn.Linear(feat_dim, z_dim)
        self.z_dim = z_dim

    def encode(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(feat)
        mu, logvar = h.chunk(2, dim=-1)
        mu = mu + 0.5 * self.skip(feat)
        std = torch.exp(0.5 * logvar.clamp(-8, 4))
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(
        self, feat: torch.Tensor, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(feat)
        out = mu if deterministic else z
        recon = self.dec(out)
        recon_loss = F.mse_loss(recon, feat.detach())
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return out, recon_loss + 0.1 * kl


class GumbelBottleneck(nn.Module):
    """可训练 Gumbel-softmax 离散瓶颈（语言梯度可回传）。"""

    def __init__(
        self,
        z_dim: int = 32,
        n_symbols: int = 64,
        temperature: float = 0.5,
        spectral_norm: bool = False,
    ):
        super().__init__()
        proj = nn.Linear(z_dim, n_symbols, bias=False)
        if spectral_norm:
            # 兼容不同 torch 版本的谱归一化 API
            if hasattr(nn.utils, "parametrizations"):
                proj = nn.utils.parametrizations.spectral_norm(proj)
            else:
                proj = nn.utils.spectral_norm(proj)
        self.proj = proj
        self.n_symbols = n_symbols
        self.temperature = temperature

    def forward(
        self, z: torch.Tensor, hard: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.proj(z)
        # soft one-hot（可微）
        y_soft = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        symbol_ids = y_soft.argmax(dim=-1)
        # 连续嵌入供下游 grounding head
        embed = y_soft @ self.proj.weight  # (B, z_dim) 近似
        return symbol_ids, y_soft, embed

    def entropy_bonus(self, y_soft: torch.Tensor) -> torch.Tensor:
        p = y_soft.clamp_min(1e-8)
        ent = -(p * p.log()).sum(dim=-1).mean()
        return ent


class FrozenOrthogonalBottleneck(nn.Module):
    """冻结正交随机投影离散瓶颈（写保护）。"""

    def __init__(self, z_dim: int = 32, n_symbols: int = 64):
        super().__init__()
        W = torch.empty(n_symbols, z_dim)
        nn.init.orthogonal_(W)
        self.register_buffer("W", W)
        self.n_symbols = n_symbols

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 写保护：切断对上游的语言梯度
        z_stop = z.detach()
        scores = F.linear(z_stop, self.W)
        symbol_ids = scores.argmax(dim=-1)
        return symbol_ids, scores


class GroundingHead(nn.Module):
    """符号/嵌入 → 标签分类头（端到端对照用）。"""

    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SocialHead(nn.Module):
    """写保护路径上的社交预测头：只读 detach(z)。"""

    def __init__(self, z_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.detach())


class WorldModelBundle(nn.Module):
    """物理引擎最小子集：CNN + VAE + 瓶颈。"""

    def __init__(
        self,
        n_labels: int,
        bottleneck: str = "frozen",
        n_symbols: int = 64,
        temperature: float = 0.5,
        spectral_norm: bool = False,
        feat_dim: int = 64,
        z_dim: int = 32,
    ):
        super().__init__()
        self.encoder = SmallCNN(feat_dim=feat_dim)
        self.vae = TinyVAE(feat_dim=feat_dim, z_dim=z_dim)
        self.bottleneck_type = bottleneck
        if bottleneck == "gumbel":
            self.bottleneck = GumbelBottleneck(
                z_dim=z_dim,
                n_symbols=n_symbols,
                temperature=temperature,
                spectral_norm=spectral_norm,
            )
            self.grounding = GroundingHead(z_dim, n_labels)
        else:
            self.bottleneck = FrozenOrthogonalBottleneck(z_dim=z_dim, n_symbols=n_symbols)
            self.grounding = SocialHead(z_dim, n_labels)
        self.n_symbols = n_symbols
        self.n_labels = n_labels

    def encode_z(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        z, vae_loss = self.vae(feat)
        return z, vae_loss

    def symbols_from_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.bottleneck_type == "gumbel":
            ids, _, _ = self.bottleneck(z, hard=True)
            return ids
        ids, _ = self.bottleneck(z)
        return ids
