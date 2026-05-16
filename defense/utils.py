# -*- coding: utf-8 -*-
"""
Utilities for HeteroGuard defense package.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch_geometric.data import HeteroData


EdgeType = Tuple[str, str, str]


def clone_hetero_data(data: HeteroData) -> HeteroData:
    out = HeteroData()

    for nt in data.node_types:
        for key, value in data[nt].items():
            out[nt][key] = _clone_value(value)
        if "x" in out[nt] and out[nt].x is not None:
            out[nt].num_nodes = int(out[nt].x.size(0))
        elif getattr(data[nt], "num_nodes", None) is not None:
            out[nt].num_nodes = int(data[nt].num_nodes)

    for et in data.edge_types:
        for key, value in data[et].items():
            out[et][key] = _clone_value(value)

    return out


def _clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone().contiguous()
    return deepcopy(value)


def get_target_train_idx(data: HeteroData, target_type: str) -> torch.Tensor:
    return torch.where(data[target_type].train_mask.detach().cpu())[0].long()


def get_target_test_idx(data: HeteroData, target_type: str) -> torch.Tensor:
    if "test_mask" not in data[target_type]:
        return torch.empty(0, dtype=torch.long)
    return torch.where(data[target_type].test_mask.detach().cpu())[0].long()


def normalize_scores(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    scores = scores.detach().float().cpu()
    if scores.numel() == 0:
        return scores
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    min_v = scores.min()
    scores = scores - min_v
    max_v = scores.max()
    if max_v <= eps:
        return torch.zeros_like(scores)
    return scores / max_v


def zscore_clip(scores: torch.Tensor, clip: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
    scores = scores.detach().float().cpu()
    if scores.numel() == 0:
        return scores
    mean = scores.mean()
    std = scores.std().clamp(min=eps)
    z = (scores - mean) / std
    return z.clamp(min=0.0, max=clip) / clip


def compute_binary_metrics(
    predicted_nodes: Union[torch.Tensor, Sequence[int]],
    true_nodes: Union[torch.Tensor, Sequence[int]],
) -> Dict[str, float]:
    if isinstance(predicted_nodes, torch.Tensor):
        pred = set(int(x) for x in predicted_nodes.detach().cpu().view(-1).tolist())
    else:
        pred = set(int(x) for x in predicted_nodes)

    if isinstance(true_nodes, torch.Tensor):
        true = set(int(x) for x in true_nodes.detach().cpu().view(-1).tolist())
    else:
        true = set(int(x) for x in true_nodes)

    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def infer_new_node_ranges(
    clean_data: HeteroData,
    poison_data: HeteroData,
) -> Dict[str, Tuple[int, int]]:
    """
    Returns node_type -> (clean_num_nodes, poison_num_nodes).
    If poison has more nodes than clean, nodes in [clean_n, poison_n) are
    considered newly added candidate trigger nodes.
    """
    out = {}
    for nt in poison_data.node_types:
        poison_n = int(poison_data[nt].x.size(0))
        clean_n = int(clean_data[nt].x.size(0)) if nt in clean_data.node_types else 0
        out[nt] = (clean_n, poison_n)
    return out


def align_clean_to_poison_metadata(
    clean_data: HeteroData,
    poison_data: HeteroData,
) -> HeteroData:
    """
    Create a clean graph with the same node/edge metadata as poison_data.

    This is important for models such as HAN/HGT whose layers are lazy-built
    from metadata. If poison_data contains new backdoor edge types, the same
    model cannot be evaluated on clean_data unless clean_data also has empty
    stores for those edge types.
    """
    aligned = clone_hetero_data(clean_data)

    for nt in poison_data.node_types:
        if nt not in aligned.node_types:
            raise ValueError(f"clean_data lacks node type {nt} required by poison_data")

    for et in poison_data.edge_types:
        if et not in aligned.edge_types or "edge_index" not in aligned[et]:
            src_type, _, _ = et
            device = aligned[src_type].x.device
            aligned[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return aligned


def edge_index_tail(
    clean_data: HeteroData,
    poison_data: HeteroData,
    edge_type: EdgeType,
) -> torch.Tensor:
    """
    Returns newly appended edge tail if poison edge count >= clean edge count.
    This matches our rewritten attack package, which appends new trigger edges.
    """
    p_ei = poison_data[edge_type].edge_index
    if edge_type not in clean_data.edge_types or "edge_index" not in clean_data[edge_type]:
        return p_ei

    c_ei = clean_data[edge_type].edge_index
    if p_ei.size(1) >= c_ei.size(1):
        return p_ei[:, c_ei.size(1):]

    # Fallback: if edge order is not append-only, return all poison edges.
    return p_ei


def make_target_weights(
    train_idx: torch.Tensor,
    suspicious_scores: torch.Tensor,
    min_weight: float = 0.1,
    max_downweight: float = 0.9,
) -> torch.Tensor:
    """
    Converts suspicious scores into CE loss weights.

    Higher suspicious score -> lower training weight.
    """
    s = normalize_scores(suspicious_scores)
    weights = 1.0 - float(max_downweight) * s
    return weights.clamp(min=float(min_weight), max=1.0)


def filter_edges_involving_nodes(
    data: HeteroData,
    suspicious_nodes_by_type: Dict[str, torch.Tensor],
) -> HeteroData:
    """
    Purify graph by removing edges incident to suspicious nodes.

    Nodes are not reindexed or physically deleted; they are isolated. This
    avoids expensive and error-prone heterogeneous reindexing.
    """
    out = clone_hetero_data(data)

    suspicious_sets = {
        nt: set(int(x) for x in nodes.detach().cpu().view(-1).tolist())
        for nt, nodes in suspicious_nodes_by_type.items()
        if nodes is not None and nodes.numel() > 0
    }

    if not suspicious_sets:
        return out

    for et in list(out.edge_types):
        src_type, _, dst_type = et
        ei = out[et].edge_index
        if ei.numel() == 0:
            continue

        keep = torch.ones(ei.size(1), dtype=torch.bool, device=ei.device)

        if src_type in suspicious_sets:
            bad_src = torch.tensor(
                [int(x) in suspicious_sets[src_type] for x in ei[0].detach().cpu().tolist()],
                dtype=torch.bool,
                device=ei.device,
            )
            keep &= ~bad_src

        if dst_type in suspicious_sets:
            bad_dst = torch.tensor(
                [int(x) in suspicious_sets[dst_type] for x in ei[1].detach().cpu().tolist()],
                dtype=torch.bool,
                device=ei.device,
            )
            keep &= ~bad_dst

        out[et].edge_index = ei[:, keep].contiguous()

    return out


def summarize_scores(name: str, scores: torch.Tensor) -> Dict:
    scores = scores.detach().float().cpu()
    if scores.numel() == 0:
        return {"name": name, "num": 0}
    return {
        "name": name,
        "num": int(scores.numel()),
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
        "mean": float(scores.mean().item()),
        "std": float(scores.std().item()) if scores.numel() > 1 else 0.0,
        "nonzero": int((scores > 0).sum().item()),
    }
