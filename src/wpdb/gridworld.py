"""最小彩色物体网格世界（Grid World）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


# 颜色 × 形状 → 标签名
COLOR_RGB = {
    "red": (1.0, 0.15, 0.15),
    "blue": (0.15, 0.35, 1.0),
    "green": (0.15, 0.85, 0.25),
    "yellow": (0.95, 0.85, 0.15),
    "purple": (0.65, 0.2, 0.85),
    "cyan": (0.15, 0.85, 0.9),
}
SHAPES = ("cube", "ball", "cylinder", "capsule")


def default_object_pool(n_objects: int) -> List[str]:
    """构造可扩展物体标签池。"""
    colors = list(COLOR_RGB.keys())
    labels: List[str] = []
    for shape in SHAPES:
        for color in colors:
            labels.append(f"{color}_{shape}")
            if len(labels) >= n_objects:
                return labels
    # 超出基础池时用编号扩展
    i = 0
    while len(labels) < n_objects:
        labels.append(f"extra_{i}")
        i += 1
    return labels


@dataclass
class GridConfig:
    grid_size: int = 5
    n_objects: int = 6
    image_size: int = 32
    seed: int = 0
    label_noise: float = 0.0  # 教师标签错误率（模拟 70% 准确时可设 0.3）


class GridWorld:
    """离散网格 + 固定放置的彩色物体；观测为 RGB 小图。"""

    ACTIONS = ("up", "down", "left", "right", "stay")

    def __init__(self, cfg: GridConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.labels = default_object_pool(cfg.n_objects)
        self.positions: Dict[Tuple[int, int], str] = {}
        self.agent: Tuple[int, int] = (0, 0)
        self.reset()

    def reset(self) -> np.ndarray:
        g = self.cfg.grid_size
        cells = [(x, y) for x in range(g) for y in range(g)]
        self.rng.shuffle(cells)
        self.positions = {}
        for i, lab in enumerate(self.labels):
            self.positions[cells[i]] = lab
        # agent 放在无物体格
        for c in cells[len(self.labels) :]:
            self.agent = c
            break
        else:
            self.agent = (0, 0)
        return self.render()

    def step(self, action: str) -> Tuple[np.ndarray, Dict]:
        x, y = self.agent
        g = self.cfg.grid_size
        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(g - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(g - 1, x + 1)
        self.agent = (x, y)
        info = {
            "pos": self.agent,
            "label": self.positions.get(self.agent),
            "on_object": self.agent in self.positions,
        }
        return self.render(), info

    def label_at(self, pos: Tuple[int, int] | None = None) -> str | None:
        p = self.agent if pos is None else pos
        return self.positions.get(p)

    def noisy_teacher_label(self, true_label: str | None) -> str | None:
        """脚本教师：可注入标签噪声。"""
        if true_label is None:
            return None
        if self.rng.random() < self.cfg.label_noise:
            others = [l for l in self.labels if l != true_label]
            return self.rng.choice(others)
        return true_label

    def render(self) -> np.ndarray:
        """渲染整图 RGB，形状 (3, H, W)，值域 [0,1]。"""
        h = w = self.cfg.image_size
        g = self.cfg.grid_size
        img = np.ones((h, w, 3), dtype=np.float32) * 0.92
        cell = h // g
        # 网格线
        for i in range(g + 1):
            img[i * cell : i * cell + 1, :, :] = 0.75
            img[:, i * cell : i * cell + 1, :] = 0.75
        for (cx, cy), lab in self.positions.items():
            color_name = lab.split("_")[0]
            rgb = COLOR_RGB.get(color_name, (0.4, 0.4, 0.4))
            y0, x0 = cy * cell + 2, cx * cell + 2
            y1, x1 = min(h, (cy + 1) * cell - 2), min(w, (cx + 1) * cell - 2)
            shape = lab.split("_")[1] if "_" in lab else "cube"
            patch = img[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            ph, pw, _ = patch.shape
            yy, xx = np.mgrid[0:ph, 0:pw]
            cy_p, cx_p = ph / 2, pw / 2
            if shape == "ball":
                mask = (yy - cy_p) ** 2 + (xx - cx_p) ** 2 <= (min(ph, pw) * 0.4) ** 2
            elif shape == "cylinder":
                mask = (np.abs(xx - cx_p) < pw * 0.28) & (np.abs(yy - cy_p) < ph * 0.42)
            elif shape == "capsule":
                mask = (np.abs(yy - cy_p) / max(ph, 1) + np.abs(xx - cx_p) / max(pw, 1)) < 0.55
            else:  # cube
                mask = np.ones((ph, pw), dtype=bool)
            for c in range(3):
                patch[..., c][mask] = rgb[c]
        # agent 标记
        ax, ay = self.agent
        y0, x0 = ay * cell + cell // 3, ax * cell + cell // 3
        y1, x1 = y0 + max(2, cell // 3), x0 + max(2, cell // 3)
        img[y0:y1, x0:x1] = (0.05, 0.05, 0.05)
        return np.transpose(img, (2, 0, 1))

    def object_centered_views(self) -> List[Tuple[np.ndarray, str, Tuple[int, int]]]:
        """每个物体的局部观测（把 agent 临时放到该格渲染）。"""
        out = []
        saved = self.agent
        for pos, lab in self.positions.items():
            self.agent = pos
            out.append((self.render(), lab, pos))
        self.agent = saved
        return out

    def random_walk_batch(
        self, n_steps: int
    ) -> Tuple[torch.Tensor, List[str | None], List[Tuple[int, int]]]:
        """随机游走采集观测。"""
        imgs = []
        labels = []
        poss = []
        for _ in range(n_steps):
            action = self.ACTIONS[int(self.rng.integers(0, len(self.ACTIONS)))]
            img, info = self.step(action)
            imgs.append(img)
            labels.append(info["label"])
            poss.append(info["pos"])
        x = torch.from_numpy(np.stack(imgs)).float()
        return x, labels, poss
