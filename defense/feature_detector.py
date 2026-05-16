# -*- coding: utf-8 -*-
"""
Feature anomaly detector.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch_geometric.data import HeteroData

from .utils import normalize_scores


class FeatureAnomalyDetector:
    def __init__(
        self,
        clean_data: HeteroData,
        poison_data: HeteroData,
        target_type: str,
    ):
        self.clean_data = clean_data
        self.poison_data = poison_data
        self.target_type = target_type

    def score_target_nodes(self, train_idx: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        train_idx = train_idx.detach().cpu().long()

        clean_x = self.clean_data[self.target_type].x.detach().float().cpu()
        poison_x = self.poison_data[self.target_type].x.detach().float().cpu()

        n = min(clean_x.size(0), poison_x.size(0))
        valid = train_idx[train_idx < n]

        scores = torch.zeros(train_idx.numel(), dtype=torch.float)
        if valid.numel() == 0:
            return scores, {"feature_diff_nonzero": 0}

        diff = (poison_x[:n] - clean_x[:n]).norm(p=2, dim=1)

        pos = {int(node): i for i, node in enumerate(train_idx.tolist())}
        for node in valid.tolist():
            scores[pos[int(node)]] = diff[int(node)]

        raw_scores = scores.clone()
        scores = normalize_scores(scores)

        return scores, {
            "feature_diff_nonzero": int((raw_scores > 1e-12).sum().item()),
            "raw_scores": raw_scores,
        }
