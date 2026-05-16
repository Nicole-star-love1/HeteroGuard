# -*- coding: utf-8 -*-
"""
Relation-aware structural detector.

This detector no longer assumes a fixed backdoor edge direction or target-type
trigger nodes only. It detects new nodes by node type and new edges by edge type,
then scores target training nodes by adjacency to any newly added trigger node.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData

from .utils import (
    EdgeType,
    edge_index_tail,
    infer_new_node_ranges,
    normalize_scores,
)


class RelationAwareStructuralDetector:
    def __init__(
        self,
        clean_data: HeteroData,
        poison_data: HeteroData,
        target_type: str,
    ):
        self.clean_data = clean_data
        self.poison_data = poison_data
        self.target_type = target_type
        self.new_node_ranges = infer_new_node_ranges(clean_data, poison_data)

    def find_new_nodes_by_type(self) -> Dict[str, torch.Tensor]:
        out = {}
        for nt, (clean_n, poison_n) in self.new_node_ranges.items():
            if poison_n > clean_n:
                out[nt] = torch.arange(clean_n, poison_n, dtype=torch.long)
            else:
                out[nt] = torch.empty(0, dtype=torch.long)
        return out

    def find_new_edges_by_type(self) -> Dict[EdgeType, torch.Tensor]:
        out = {}
        for et in self.poison_data.edge_types:
            new_edges = edge_index_tail(self.clean_data, self.poison_data, et)
            if new_edges.numel() > 0 and new_edges.size(1) > 0:
                out[et] = new_edges.detach().cpu()
        return out

    def score_target_nodes(
        self,
        train_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Direction-agnostic structural score for target training nodes.

        A target node gets score if it is connected to:
            - a newly added trigger node of any type, or
            - any newly added edge incident to itself.
        """
        train_idx = train_idx.detach().cpu().long()
        pos = {int(n): i for i, n in enumerate(train_idx.tolist())}
        scores = torch.zeros(train_idx.numel(), dtype=torch.float)

        new_nodes = self.find_new_nodes_by_type()
        new_edges = self.find_new_edges_by_type()

        trigger_sets = {
            nt: set(int(x) for x in idx.tolist())
            for nt, idx in new_nodes.items()
            if idx.numel() > 0
        }

        edge_hits = 0
        trigger_adj_hits = 0

        for et, edges in new_edges.items():
            src_type, _, dst_type = et
            if edges.numel() == 0:
                continue

            src = edges[0].tolist()
            dst = edges[1].tolist()

            for s, d in zip(src, dst):
                s = int(s)
                d = int(d)

                src_is_trigger = src_type in trigger_sets and s in trigger_sets[src_type]
                dst_is_trigger = dst_type in trigger_sets and d in trigger_sets[dst_type]

                # Target node at source side.
                if src_type == self.target_type and s in pos:
                    edge_hits += 1
                    if dst_is_trigger:
                        trigger_adj_hits += 1
                        scores[pos[s]] += 5.0
                    else:
                        scores[pos[s]] += 1.0

                # Target node at destination side.
                if dst_type == self.target_type and d in pos:
                    edge_hits += 1
                    if src_is_trigger:
                        trigger_adj_hits += 1
                        scores[pos[d]] += 5.0
                    else:
                        scores[pos[d]] += 1.0

        raw_scores = scores.clone()
        scores = normalize_scores(scores)

        info = {
            "new_nodes_by_type": {
                nt: idx.tolist() for nt, idx in new_nodes.items() if idx.numel() > 0
            },
            "new_edges_by_type": {
                str(et): int(edges.size(1)) for et, edges in new_edges.items()
            },
            "edge_hits": int(edge_hits),
            "trigger_adj_hits": int(trigger_adj_hits),
            "raw_scores": raw_scores,
        }
        return scores, info

    def detect_trigger_nodes_by_type(self) -> Dict[str, torch.Tensor]:
        """
        Returns newly added nodes by type. In the rewritten attack package,
        these correspond to trigger nodes.
        """
        return self.find_new_nodes_by_type()

    def detect_trigger_edges(self) -> Dict[EdgeType, torch.Tensor]:
        return self.find_new_edges_by_type()
