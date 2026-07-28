"""梯度隔离黑板（Memory Table）与 DP-Means 冲突分裂。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryTable:
    """非参数 Dict[symbol → Counter[label]]，零梯度。"""

    table: Dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    sub_symbols: Dict[int, List[int]] = field(default_factory=dict)
    next_sub_id: int = 10_000

    def write(self, symbol: int, label: str) -> None:
        self.table[int(symbol)][label] += 1

    def predict(self, symbol: int) -> Optional[str]:
        c = self.table.get(int(symbol))
        if not c:
            return None
        return c.most_common(1)[0][0]

    def grounding_accuracy(self, pairs: List[Tuple[int, str]]) -> float:
        if not pairs:
            return 0.0
        ok = sum(1 for s, y in pairs if self.predict(s) == y)
        return ok / len(pairs)

    def collision_count(self, min_count: int = 2) -> int:
        n = 0
        for c in self.table.values():
            active = [k for k, v in c.items() if v >= min_count]
            if len(active) >= 2:
                n += 1
        return n

    def active_symbols(self) -> int:
        return sum(1 for c in self.table.values() if sum(c.values()) > 0)


class DPMeansCollisionResolver:
    """
    简化 DP-Means：同一 base symbol 上多标签冲突时，按特征距离分裂子簇。

    实现要点：
    - 每个 (base_symbol, label) 维护一个质心
    - 冲突后查询走最近质心对应的子符号
    - 无训练参数；特征固定时结果近似确定性
    """

    def __init__(self, lambda_threshold: float = 0.35, min_conflict: int = 2):
        self.lambda_threshold = lambda_threshold
        self.min_conflict = min_conflict
        # base_symbol -> list[(centroid, sub_id, label)]
        self.clusters: Dict[int, List[Tuple[np.ndarray, int, str]]] = {}

    def _norm(self, feature: np.ndarray) -> np.ndarray:
        feat = np.asarray(feature, dtype=np.float64).ravel()
        return feat / (np.linalg.norm(feat) + 1e-8)

    def _new_sub(self, memory: MemoryTable) -> int:
        sid = memory.next_sub_id
        memory.next_sub_id += 1
        return sid

    def observe(
        self,
        memory: MemoryTable,
        symbol: int,
        label: str,
        feature: np.ndarray,
        enable_split: bool = True,
    ) -> int:
        symbol = int(symbol)
        feat = self._norm(feature)

        if not enable_split:
            memory.write(symbol, label)
            return symbol

        clusters = self.clusters.setdefault(symbol, [])

        # 若该 label 已有簇：更新质心并写入子符号
        for i, (cent, sid, lab) in enumerate(clusters):
            if lab == label:
                new_cent = 0.85 * cent + 0.15 * feat
                new_cent = new_cent / (np.linalg.norm(new_cent) + 1e-8)
                clusters[i] = (new_cent, sid, lab)
                memory.write(sid, label)
                return sid

        # 尚无该 label：先看 base 是否已有冲突压力
        memory.write(symbol, label)
        counts = memory.table[symbol]
        active_labels = [k for k, v in counts.items() if v >= self.min_conflict]

        if len(active_labels) >= 2 or len(clusters) >= 1:
            # 为当前 label 开新子符号（冲突分裂）
            sid = self._new_sub(memory)
            clusters.append((feat.copy(), sid, label))
            memory.sub_symbols.setdefault(symbol, []).append(sid)
            memory.write(sid, label)
            # 若其它已有标签还没有子簇，也补建（用当前特征作占位会被后续更新）
            for lab2, cnt in list(counts.items()):
                if lab2 == label or cnt < self.min_conflict:
                    continue
                if any(l == lab2 for _, _, l in clusters):
                    continue
                sid2 = self._new_sub(memory)
                clusters.append((feat.copy() * 0.0, sid2, lab2))  # 占位，等该标签到来更新
                # 用微扰避免零向量
                clusters[-1] = (self._norm(feat + 0.05 * np.random.randn(*feat.shape)), sid2, lab2)
                memory.sub_symbols.setdefault(symbol, []).append(sid2)
                memory.write(sid2, lab2)
            return sid

        # 无冲突：记录单簇（仍用 base id）
        if not clusters:
            clusters.append((feat.copy(), symbol, label))
        return symbol

    def resolve_symbol(
        self, symbol: int, feature: np.ndarray, enable_split: bool = True
    ) -> int:
        if not enable_split or symbol not in self.clusters or not self.clusters[symbol]:
            return int(symbol)
        feat = self._norm(feature)
        clusters = self.clusters[symbol]
        # 若只有指向 base 的单簇，直接返回
        if len(clusters) == 1 and clusters[0][1] == symbol:
            return int(symbol)
        best_sid, best_d = int(symbol), 1e9
        for cent, sid, _ in clusters:
            d = float(np.linalg.norm(feat - cent))
            if d < best_d:
                best_d, best_sid = d, sid
        return int(best_sid)
