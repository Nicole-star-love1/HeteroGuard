# -*- coding: utf-8 -*-
"""
Shared utilities for heterogeneous graph backdoor experiments.

This version is aligned with the rewritten packages:

    data/hetero_dataset.py
    models/hetero_gnn.py
    attack/
    defense/

Core assumptions:
    - model(data) returns {node_type: logits}
    - attacker.poison(clean_data) returns training poisoned graph
    - attacker.inject_trigger(poison_data, inject_nodes) returns ASR evaluation graph
"""

from __future__ import annotations

import json
import logging
import os
import random
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# HeteroData helpers
# ---------------------------------------------------------------------

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


def align_clean_to_reference_metadata(clean_data: HeteroData, reference_data: HeteroData) -> HeteroData:
    """
    Add empty edge stores so a model initialized on reference_data metadata can
    also evaluate clean_data.
    """
    aligned = clone_hetero_data(clean_data)

    for et in reference_data.edge_types:
        if et not in aligned.edge_types or "edge_index" not in aligned[et]:
            src_type, _, _ = et
            device = aligned[src_type].x.device
            aligned[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return aligned


def validate_edge_index(data: HeteroData, raise_error: bool = True) -> bool:
    ok = True
    for et in data.edge_types:
        src_type, _, dst_type = et
        ei = data[et].edge_index

        if ei.numel() == 0:
            continue

        n_src = data[src_type].x.size(0)
        n_dst = data[dst_type].x.size(0)

        bad = (
            int(ei[0].min().item()) < 0 or
            int(ei[1].min().item()) < 0 or
            int(ei[0].max().item()) >= n_src or
            int(ei[1].max().item()) >= n_dst
        )
        if bad:
            ok = False
            msg = (
                f"Edge index out of bounds for {et}: "
                f"src=[{int(ei[0].min().item())},{int(ei[0].max().item())}]/{n_src}, "
                f"dst=[{int(ei[1].min().item())},{int(ei[1].max().item())}]/{n_dst}"
            )
            if raise_error:
                raise ValueError(msg)
            logger.warning(msg)

    return ok


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def train_node_classifier(
    model,
    data: HeteroData,
    target_node_type: str,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    verbose: bool = False,
):
    """
    Supervised node classifier training.

    The model must follow:
        out_dict = model(data)
        logits = out_dict[target_node_type]
    """
    validate_edge_index(data, raise_error=True)

    model.eval()
    with torch.no_grad():
        _ = model(data)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = data[target_node_type].train_mask
    val_mask = data[target_node_type].val_mask
    labels = data[target_node_type].y

    best_val_acc = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        out = model(data)[target_node_type]
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)[target_node_type]
            preds = logits.argmax(dim=1)
            if val_mask.sum().item() > 0:
                val_acc = (preds[val_mask] == labels[val_mask]).float().mean().item()
            else:
                val_acc = 0.0

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            logger.info(
                f"    Epoch {epoch + 1}/{epochs}, "
                f"Loss={loss.item():.4f}, ValAcc={val_acc:.4f}"
            )

        if patience is not None and patience > 0 and (epoch - best_epoch) >= patience:
            if verbose:
                logger.info(f"    Early stop at epoch {epoch + 1}, best_epoch={best_epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Ensure all lazily-created modules stay on the same device as the input graph.
    try:
        first_x = next(iter(data.x_dict.values()))
        model.to(first_x.device)
    except Exception:
        pass

    return model


@torch.no_grad()
def measure_clean_acc(
    model,
    data: HeteroData,
    target_type: str,
    exclude_indices: Optional[Union[torch.Tensor, Sequence[int]]] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    model.to(device)
    model.eval()
    # Avoid mutating caller-owned HeteroData via in-place PyG .to().
    data = clone_hetero_data(data).to(device)

    test_mask = data[target_type].test_mask.detach().cpu().clone()
    if exclude_indices is not None and len(exclude_indices) > 0:
        if isinstance(exclude_indices, torch.Tensor):
            exclude_set = set(int(x) for x in exclude_indices.detach().cpu().view(-1).tolist())
        else:
            exclude_set = set(int(x) for x in exclude_indices)

        test_indices = torch.where(test_mask)[0].tolist()
        keep = [idx for idx in test_indices if idx not in exclude_set]
        test_mask = torch.zeros_like(test_mask)
        if keep:
            test_mask[torch.tensor(keep, dtype=torch.long)] = True

    if test_mask.sum().item() == 0:
        return torch.tensor(0.0)

    labels = data[target_type].y.detach().cpu()
    logits = model(data)[target_type].detach().cpu()
    preds = logits.argmax(dim=1)

    return (preds[test_mask] == labels[test_mask]).float().mean()


def measure_asr(
    model,
    clean_data: HeteroData,
    poison_data: HeteroData,
    target_type: str,
    target_class: int = 0,
    num_inject: int = 200,
    device: Union[str, torch.device] = "cpu",
    attacker=None,
    seed: int = 42,
    exclude_target_class: bool = True,
):
    """
    Measure attack success rate.

    Preferred path:
        attacker.inject_trigger(poison_data, inject_indices)

    This guarantees that ASR evaluation uses the same trigger pattern as the
    attack implementation. Manual trigger-direction guessing is intentionally
    removed.
    """
    model.to(device)
    model.eval()

    # Avoid mutating caller-owned HeteroData via in-place PyG .cpu().
    clean_cpu = clone_hetero_data(clean_data).cpu()
    poison_cpu = clone_hetero_data(poison_data).cpu()

    test_mask = clean_cpu[target_type].test_mask
    test_indices = torch.where(test_mask)[0].long()

    if test_indices.numel() == 0:
        return None, None, None

    labels = clean_cpu[target_type].y.detach().cpu()

    if exclude_target_class:
        candidate_indices = test_indices[labels[test_indices] != int(target_class)]
        if candidate_indices.numel() == 0:
            candidate_indices = test_indices
    else:
        candidate_indices = test_indices

    num_inject = min(int(num_inject), int(candidate_indices.numel()))
    if num_inject <= 0:
        return None, None, None

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    perm = torch.randperm(candidate_indices.numel(), generator=gen)
    inject_indices = candidate_indices[perm[:num_inject]].long()

    all_test_set = set(int(x) for x in test_indices.tolist())
    inject_set = set(int(x) for x in inject_indices.tolist())
    other_indices = torch.tensor(
        sorted(list(all_test_set - inject_set)),
        dtype=torch.long,
    )

    if attacker is not None and hasattr(attacker, "inject_trigger"):
        injected_data = attacker.inject_trigger(poison_cpu, inject_indices)
    else:
        logger.warning(
            "measure_asr called without attacker.inject_trigger. "
            "Falling back to poison_data without additional test-time injection."
        )
        injected_data = clone_hetero_data(poison_cpu)

    injected_data = clone_hetero_data(injected_data).to(device)

    with torch.no_grad():
        logits = model(injected_data)[target_type].detach().cpu()
        preds = logits.argmax(dim=1)

    asr = (preds[inject_indices] == int(target_class)).float().mean().item()

    if other_indices.numel() > 0:
        clean_acc = (preds[other_indices] == labels[other_indices]).float().mean().item()
    else:
        clean_acc = None

    natural_ratio = (labels[candidate_indices] == int(target_class)).float().mean().item()

    return asr, clean_acc, natural_ratio


# ---------------------------------------------------------------------
# Attacker factory
# ---------------------------------------------------------------------

def create_attacker(attack_name: str, target_type: str, num_classes: int, args: Optional[Dict] = None):
    """
    Compatibility wrapper around attack.hetero_attack.create_attacker.

    Supports names:
        feature, sba, uba, relation, clean_label, cba, grad/gba/gcba
    """
    from attack.hetero_attack import create_attacker as _create_attacker

    args = dict(args or {})

    # Accept legacy aliases.
    attack_key = str(attack_name).strip().lower()
    if attack_key == "gba" or attack_key == "gcba":
        attack_key = "grad"
    if attack_key == "metapath":
        attack_key = "relation"

    return _create_attacker(
        attack_key,
        target_node_type=target_type,
        num_classes=num_classes,
        attacker_kwargs=args,
    )


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def save_checkpoint(checkpoint_path: str, checkpoint_data: Dict):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(checkpoint_data), f, indent=2, ensure_ascii=False)
    logger.info(f"检查点已保存: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> Dict:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"检查点已加载: {checkpoint_path}")
    return data


def get_checkpoint_path(
    checkpoint_dir: str,
    dataset: str,
    attack: str,
    poison_rate: float,
    phase: str = "train",
    model: Optional[str] = None,
) -> str:
    if model is None:
        filename = f"{dataset}_{attack}_r{float(poison_rate):.2f}_{phase}.json"
    else:
        filename = f"{dataset}_{model}_{attack}_r{float(poison_rate):.2f}_{phase}.json"
    return os.path.join(checkpoint_dir, filename)


def get_model_checkpoint_path(
    checkpoint_dir: str,
    dataset: str,
    model: str,
    tag: str,
    attack: Optional[str] = None,
    poison_rate: Optional[float] = None,
) -> str:
    if attack is None:
        filename = f"{dataset}_{model}_{tag}.pt"
    else:
        filename = f"{dataset}_{model}_{attack}_r{float(poison_rate):.2f}_{tag}.pt"
    return os.path.join(checkpoint_dir, "models", filename)
